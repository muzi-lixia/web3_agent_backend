"""user_memory 读写。

同一 subject 只保留一条 active：写入前先把旧的标 superseded，靠部分唯一索引兜底。
检索走余弦距离，距离上限由调用方给 —— 阈值不确定时宁可不召回，也不要塞错记忆。
"""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.domain.memory.memory import MemoryItem
from web3_agent.persistence.pg.orm_models import UserMemory, UserMemoryState


async def active_items(db: AsyncSession, siwe_user_id: str) -> list[UserMemory]:
    stmt = (
        select(UserMemory)
        .where(UserMemory.siwe_user_id == siwe_user_id, UserMemory.status == "active")
        .order_by(UserMemory.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def recall(
    db: AsyncSession,
    siwe_user_id: str,
    *,
    embedding: list[float],
    limit: int,
    max_distance: float,
) -> list[UserMemory]:
    """constraint 不参与这里 —— 它们全量注入，不靠相似度。"""
    distance = UserMemory.embedding.cosine_distance(embedding)
    stmt = (
        select(UserMemory)
        .where(
            UserMemory.siwe_user_id == siwe_user_id,
            UserMemory.status == "active",
            UserMemory.memory_type != "constraint",
            distance <= max_distance,
        )
        .order_by(distance)
        .limit(limit)
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


async def insert_many(
    db: AsyncSession,
    siwe_user_id: str,
    items: list[MemoryItem],
    embeddings: list[list[float]],
) -> None:
    for item, vec in zip(items, embeddings, strict=True):
        db.add(
            UserMemory(
                siwe_user_id=siwe_user_id,
                text=item.text,
                memory_type=item.memory_type,
                subject=item.subject,
                embedding=vec,
                confidence=item.confidence,
            )
        )
    await db.flush()


async def get_state(db: AsyncSession, siwe_user_id: str) -> UserMemoryState | None:
    return await db.get(UserMemoryState, siwe_user_id)


async def advance_watermark(db: AsyncSession, siwe_user_id: str, message_id: int) -> None:
    stmt = (
        insert(UserMemoryState)
        .values(
            siwe_user_id=siwe_user_id,
            last_extracted_message_id=message_id,
            fail_count=0,
        )
        .on_conflict_do_update(
            index_elements=["siwe_user_id"],
            set_={"last_extracted_message_id": message_id, "fail_count": 0},
        )
    )
    await db.execute(stmt)


async def mark_failed(db: AsyncSession, siwe_user_id: str) -> None:
    """失败不动水位线：下一轮还会重试同一段，不会漏掉。"""
    stmt = (
        insert(UserMemoryState)
        .values(siwe_user_id=siwe_user_id, last_extracted_message_id=0, fail_count=1)
        .on_conflict_do_update(
            index_elements=["siwe_user_id"],
            set_={"fail_count": UserMemoryState.fail_count + 1},
        )
    )
    await db.execute(stmt)
