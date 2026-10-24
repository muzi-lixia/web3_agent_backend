"""user_memory 读写。

同一 subject 只保留一条 active：写入前先把旧的标 superseded，靠部分唯一索引兜底。
读取整批 active，按写入时间倒序 —— 注入时取前几条，不做相似度筛选：这个用户的
active 集就是「记忆主题数」，排序在内存里完成，比一次 query embedding 便宜得多。

写入只有 `commit_extraction` 一个入口，且必须带读取时的水位线、生命周期版本
和抽取租约。三个条件一起构成「这份结果还有效吗」，任一不成立就整份丢弃。
分开校验、或者只用 GREATEST() 推进水位线都不够：水位线只能防回退，
挡不住旧任务用自己的旧结论 supersede 掉新结论。
"""

from __future__ import annotations

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.domain.memory.memory import MemoryItem
from web3_agent.persistence.pg.orm_models import UserMemory, UserMemoryState
from web3_agent.persistence.pg.repositories import conversation_repo


async def active_items(db: AsyncSession, siwe_user_id: str) -> list[UserMemory]:
    stmt = (
        select(UserMemory)
        .where(UserMemory.siwe_user_id == siwe_user_id, UserMemory.status == "active")
        .order_by(UserMemory.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def supersede(db: AsyncSession, siwe_user_id: str, subjects: list[str]) -> int:
    if not subjects:
        return 0
    result = await db.execute(
        update(UserMemory)
        .where(
            UserMemory.siwe_user_id == siwe_user_id,
            UserMemory.status == "active",
            UserMemory.subject.in_(subjects),
        )
        .values(status="superseded", superseded_at=func.now())
    )
    return result.rowcount or 0


async def claim(db: AsyncSession, siwe_user_id: str, token: str, ttl_seconds: int) -> bool:
    """抢占该用户的抽取任务：同一时刻只允许一个有效任务。

    一行带条件的 upsert：行不存在就建；已存在时只有「没人持有」或「持有者已过期」
    才更新。并发的第二个任务会阻塞在这一行的锁上，等第一个提交后重新判定条件，
    拿不到就返回 False —— 不调模型、不写库，省掉一次白跑。

    行在 claim 阶段就建出来（水位线 0）：状态行必须存在，后续的条件更新才有行可锁。
    """
    expires = func.now() + func.make_interval(0, 0, 0, 0, 0, 0, float(ttl_seconds))
    stmt = (
        insert(UserMemoryState)
        .values(
            siwe_user_id=siwe_user_id,
            last_extracted_message_id=0,
            fail_count=0,
            claim_token=token,
            claim_expires_at=expires,
        )
        .on_conflict_do_update(
            index_elements=["siwe_user_id"],
            set_={"claim_token": token, "claim_expires_at": expires},
            where=or_(
                UserMemoryState.claim_token.is_(None),
                UserMemoryState.claim_expires_at.is_(None),
                UserMemoryState.claim_expires_at < func.now(),
            ),
        )
        .returning(UserMemoryState.claim_token)
    )
    return (await db.execute(stmt)).scalar_one_or_none() == token


async def release_claim(db: AsyncSession, siwe_user_id: str, token: str) -> None:
    """还回租约。只释放自己持有的那一个 —— 过期后被别人接手，这里不能把人家的清掉。"""
    await db.execute(
        update(UserMemoryState)
        .where(
            UserMemoryState.siwe_user_id == siwe_user_id,
            UserMemoryState.claim_token == token,
        )
        .values(claim_token=None, claim_expires_at=None)
    )


async def commit_extraction(
    db: AsyncSession,
    siwe_user_id: str,
    *,
    until_id: int,
    items: list[MemoryItem],
    expected_watermark: int,
    expected_generation: int,
    claim_token: str,
) -> bool:
    """替换记忆、插入新记忆、推进水位线 —— 一次条件事务，要么全成，要么全不成。

    三个条件必须同时成立：
    - 生命周期版本没变（用户没在抽取期间清空或彻底删除）；
    - 水位线仍是读取时的值（没有别的任务先跑完这一段）；
    - 租约还在自己手里（没有被过期接管）。

    条件判断放在事务最前面：拿到 user_memory_state 的行锁之后再动记忆表，
    后面这段写入不可能被另一个抽取任务插进来。整套失败时返回 False，
    调用方不需要也不应该再补写 —— 事务里一个字都没落。
    """
    if not await conversation_repo.assert_generation(db, siwe_user_id, expected_generation):
        # 清空只推进版本、不删记忆行，租约还在这一行上；不还回去，
        # 这个用户的抽取要等到租约过期才恢复。
        await release_claim(db, siwe_user_id, claim_token)
        return False
    if not await _advance_gate(db, siwe_user_id, until_id, expected_watermark, claim_token):
        await release_claim(db, siwe_user_id, claim_token)
        return False
    if items:
        # 顺序不能反：同 subject 只允许一条 active，先插入会撞部分唯一索引。
        await supersede(db, siwe_user_id, [item.subject for item in items])
        await _insert_rows(db, siwe_user_id, items)
    return True


async def _advance_gate(
    db: AsyncSession,
    siwe_user_id: str,
    until_id: int,
    expected_watermark: int,
    claim_token: str,
) -> bool:
    """带条件的推进：只有「水位线还是读到的那个值」且「租约还是我的」才写得进去。

    这条 UPDATE 同时做三件事：占住状态行的锁（后续写入的串行点）、
    校验水位线、释放租约。返回 False 说明这份结果基于已经过期的读取。
    """
    stmt = (
        update(UserMemoryState)
        .where(
            UserMemoryState.siwe_user_id == siwe_user_id,
            UserMemoryState.last_extracted_message_id == expected_watermark,
            UserMemoryState.claim_token == claim_token,
        )
        .values(
            last_extracted_message_id=until_id,
            fail_count=0,
            claim_token=None,
            claim_expires_at=None,
        )
        .returning(UserMemoryState.last_extracted_message_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None


async def _insert_rows(
    db: AsyncSession,
    siwe_user_id: str,
    items: list[MemoryItem],
) -> None:
    for item in items:
        db.add(
            UserMemory(
                siwe_user_id=siwe_user_id,
                text=item.text,
                memory_type=item.memory_type,
                subject=item.subject,
                confidence=item.confidence,
            )
        )
    await db.flush()


async def get_state(db: AsyncSession, siwe_user_id: str) -> UserMemoryState | None:
    return await db.get(UserMemoryState, siwe_user_id)


async def purge(db: AsyncSession, siwe_user_id: str) -> int:
    """彻底删除：superseded 的历史行一起删，水位线也删 —— 留着水位线等于记住
    用户聊过多少条消息。返回删掉的记忆条数。"""
    result = await db.execute(delete(UserMemory).where(UserMemory.siwe_user_id == siwe_user_id))
    await db.execute(delete(UserMemoryState).where(UserMemoryState.siwe_user_id == siwe_user_id))
    return result.rowcount or 0


async def mark_failed(
    db: AsyncSession, siwe_user_id: str, *, claim_token: str, expected_generation: int
) -> bool:
    """失败不动水位线：下一轮还会重试同一段，不会漏掉。

    走条件 UPDATE 而不是 upsert：状态行由 claim 建出来，这里只增计数并还回租约。
    行被清空删掉时这里什么都不做 —— 不会凭空造出一条「这个用户抽过 0 条、
    失败 1 次」的记录。
    """
    if not await conversation_repo.assert_generation(db, siwe_user_id, expected_generation):
        await release_claim(db, siwe_user_id, claim_token)
        return False
    stmt = (
        update(UserMemoryState)
        .where(
            UserMemoryState.siwe_user_id == siwe_user_id,
            UserMemoryState.claim_token == claim_token,
        )
        .values(
            fail_count=UserMemoryState.fail_count + 1,
            claim_token=None,
            claim_expires_at=None,
        )
        .returning(UserMemoryState.fail_count)
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None
