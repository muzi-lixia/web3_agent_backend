"""conversation_message 读写。会话不是表，归属只有 siwe_user_id 一个来源。"""

from __future__ import annotations

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.orm_models import ConversationGeneration, ConversationMessage


async def lock_generation(db: AsyncSession, siwe_user_id: str) -> int:
    """读生命周期版本并加共享锁，直到本事务结束。行不存在时建一行 0。

    锁随事务释放，所以只能用在几百毫秒的短事务里 —— 一轮对话的模型调用
    不能包在这里面。清空接口的 bump 要拿排他锁，两者因此天然串行：
    要么写事务先完成（随后被清空删掉），要么清空先提交（写事务看到新版本）。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
    """
    stmt = (
        select(ConversationGeneration.generation)
        .where(ConversationGeneration.siwe_user_id == siwe_user_id)
        .with_for_update(read=True)
    )
    current = (await db.execute(stmt)).scalar_one_or_none()
    if current is not None:
        return current
    inserted = await db.execute(
        insert(ConversationGeneration)
        .values(siwe_user_id=siwe_user_id, generation=0)
        .on_conflict_do_nothing(index_elements=["siwe_user_id"])
        .returning(ConversationGeneration.generation)
    )
    created = inserted.scalar_one_or_none()
    if created is not None:
        return created
    # 并发下别人抢先插入：回头再锁一次，这次一定命中。
    return (await db.execute(stmt)).scalar_one()


async def current_generation(db: AsyncSession, siwe_user_id: str) -> int:
    """只读版本号，不加锁。后台任务读取时用 —— 写回那一步才需要锁。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
    """
    value = (
        await db.execute(
            select(ConversationGeneration.generation).where(
                ConversationGeneration.siwe_user_id == siwe_user_id
            )
        )
    ).scalar_one_or_none()
    return value or 0


async def assert_generation(db: AsyncSession, siwe_user_id: str, expected: int) -> bool:
    """确认版本未被推进，并在本事务内锁住它。

    必须与真正的写入在同一个事务里：读一次比对、再开事务写，中间仍会被清空插进来。
    返回 False 表示这份结果基于已被删除的数据，调用方必须放弃写入。

    走 lock_generation 而不是只做一次 SELECT：版本行不存在时普通 SELECT 锁不到
    任何东西，此时清空接口的 bump 会立刻插入并提交，而这边仍判定 0 == 0 通过 ——
    写回去的内容就落在了一次删除之后。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        expected: 取数时的会话版本，写回前必须仍然一致。
    """
    return await lock_generation(db, siwe_user_id) == expected


async def bump_generation(db: AsyncSession, siwe_user_id: str) -> int:
    """推进生命周期版本。清空 / 彻底删除的第一步，拿排他行锁。

    在建行分支写 1 而不是 0：任何携带 0 的旧任务都会被判为过期。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
    """
    stmt = (
        insert(ConversationGeneration)
        .values(siwe_user_id=siwe_user_id, generation=1)
        .on_conflict_do_update(
            index_elements=["siwe_user_id"],
            set_={"generation": ConversationGeneration.generation + 1},
        )
        .returning(ConversationGeneration.generation)
    )
    return (await db.execute(stmt)).scalar_one()


async def load_history(
    db: AsyncSession, siwe_user_id: str, limit: int, max_chars: int
) -> list[tuple[str, str]]:
    """按 id 正序返回最近 limit 条 (role, content)，再按字符预算从旧到新裁掉。

    摘要行不进对话窗口：它是压缩后的结论，由 `latest_summary` 单独取出。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        limit: 本次查询或展示返回条数上限。
        max_chars: 本段文本允许的最大字符预算。
    """
    stmt = (
        select(ConversationMessage.role, ConversationMessage.content, ConversationMessage.turn_id)
        .where(
            ConversationMessage.siwe_user_id == siwe_user_id,
            ConversationMessage.role != "summary",
        )
        .order_by(ConversationMessage.id.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    ordered = _order_turns([(r.turn_id, r.role, r.content) for r in reversed(rows)])
    return _fit_budget([(role, content) for role, content in ordered], max_chars)


async def latest_summary(db: AsyncSession, siwe_user_id: str) -> ConversationMessage | None:
    """当前生效的摘要行。它的 id 同时是摘要水位线：比它新的消息还没被压缩。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
    """
    stmt = (
        select(ConversationMessage)
        .where(
            ConversationMessage.siwe_user_id == siwe_user_id,
            ConversationMessage.role == "summary",
        )
        .order_by(ConversationMessage.id.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalars().first()


async def write_summary(
    db: AsyncSession,
    siwe_user_id: str,
    *,
    content: str,
    expected_id: int | None,
    expected_generation: int,
) -> bool:
    """用新的摘要行替换旧的。`expected_id` 是读取时那条摘要行的 id（没有则 None）。

    删旧插新而不是 UPDATE：新行的 id 天然就是新的水位线，不用另存一个字段。

    两步都是条件写，失败时不留下任何痕迹：
    - 替换：`DELETE WHERE id = expected_id`。删不到说明这条摘要已被别人改掉或被清空，
      直接放弃 —— 不能用无条件删除，那样在判定失败之前就已经把别人的结果删了。
    - 首次：`INSERT ... ON CONFLICT DO NOTHING`。`uq_conv_summary_user` 保证每用户
      最多一条摘要行，两个任务同时插只有一个能成。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        content: 消息正文、文档正文或待写入文本。
        expected_id: 取数时的旧摘要消息 ID；为空表示原先没有摘要。
        expected_generation: 取数时看到的会话生命周期版本。

    Returns:
        本份摘要成功写入为 True；版本或旧摘要冲突时为 False。
    """
    if not await assert_generation(db, siwe_user_id, expected_generation):
        return False

    if expected_id is not None:
        removed = (
            await db.execute(
                delete(ConversationMessage).where(
                    ConversationMessage.siwe_user_id == siwe_user_id,
                    ConversationMessage.role == "summary",
                    ConversationMessage.id == expected_id,
                )
            )
        ).rowcount or 0
        if not removed:
            return False

    inserted = await db.execute(
        insert(ConversationMessage)
        .values(siwe_user_id=siwe_user_id, role="summary", content=content)
        .on_conflict_do_nothing(
            index_elements=["siwe_user_id"], index_where=text("role = 'summary'")
        )
    )
    return bool(inserted.rowcount)


def _order_turns(rows: list[tuple[str | None, str, str]]) -> list[tuple[str, str]]:
    """把交错的消息重排成完整轮次：轮次之间按首条 id，轮次内 user 在前。

    两台设备同时提问时，落库顺序是 u1,u2,a1,a2 —— 原样喂给模型，a1 看起来像在回答 u2。
    没有 turn_id 的旧数据各自成组，保持 id 顺序。

    Args:
        rows: 已读取的数据库记录列表。
    """
    order: list[str] = []
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for index, (turn_id, role, content) in enumerate(rows):
        key = turn_id or f"\0{index}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((0 if role == "user" else 1, role, content))
    ordered: list[tuple[str, str]] = []
    for key in order:
        # 组内 user 在前；同一角色内保持原有 id 顺序（list.sort 稳定）。
        ordered.extend((role, content) for _, role, content in sorted(groups[key], key=lambda r: r[0]))
    return ordered


def _fit_budget(
    messages: list[tuple[str, str]], max_chars: int
) -> list[tuple[str, str]]:
    """从最新一条往回装，装不下就丢掉更早的，保留一段连续的结尾。

    最新一条永远保留：用户下一句多半就在问它，截断它等于把上下文切成两半。

    Args:
        messages: 按时间排列的 (角色, 正文) 消息列表。
        max_chars: 本段文本允许的最大字符预算。
    """
    if max_chars <= 0:
        return messages
    kept: list[tuple[str, str]] = []
    used = 0
    for role, content in reversed(messages):
        size = len(content or "")
        if kept and used + size > max_chars:
            break
        kept.append((role, content))
        used += size
    return list(reversed(kept))


async def list_page(
    db: AsyncSession, siwe_user_id: str, limit: int, before_id: int | None = None
) -> list[ConversationMessage]:
    """多取一条用来判 hasMore。摘要行不是对话内容，不进列表。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        limit: 本次查询或展示返回条数上限。
        before_id: 历史分页上界，仅返回 ID 小于该值的消息。
    """
    stmt = select(ConversationMessage).where(
        ConversationMessage.siwe_user_id == siwe_user_id,
        ConversationMessage.role != "summary",
    )
    if before_id is not None:
        stmt = stmt.where(ConversationMessage.id < before_id)
    stmt = stmt.order_by(ConversationMessage.id.desc()).limit(limit + 1)
    return list((await db.execute(stmt)).scalars().all())


async def append_message(
    db: AsyncSession,
    *,
    siwe_user_id: str,
    role: str,
    content: str,
    tool_calls: list | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    intent_preview: dict | None = None,
) -> int:
    """追加用户或助手消息并 flush，返回数据库分配的消息 ID。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        role: 消息角色，如 user、assistant 或 summary。
        content: 消息正文、文档正文或待写入文本。
        tool_calls: 随助手消息保存的工具轨迹 JSON；没有调用时为空。
        turn_id: 同一轮用户问题与助手回答共用的标识。
        task_id: 本轮创建的写操作任务标识；普通问答为空。
        intent_preview: 写操作待确认的参数快照，用于刷新后重建卡片。
    """
    row = ConversationMessage(
        siwe_user_id=siwe_user_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        turn_id=turn_id,
        task_id=task_id,
        intent_preview=intent_preview,
    )
    db.add(row)
    await db.flush()
    return row.id


async def clear_user_messages(db: AsyncSession, siwe_user_id: str) -> int:
    """删除该用户全部消息，包含摘要，返回删除行数。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
    """
    result = await db.execute(
        delete(ConversationMessage).where(ConversationMessage.siwe_user_id == siwe_user_id)
    )
    return result.rowcount or 0


async def summarizable_range(
    db: AsyncSession, siwe_user_id: str, *, window: int, since_id: int, cap: int
) -> tuple[int, int | None]:
    """返回（可摘要条数，本批压到的最大 id）。

    可摘要区间 = 已被挤出最近 window 条、且 id 大于水位线的那部分。
    窗口起点用 id 定义，不用「第几条」—— id 有空洞（清空删过、并发插入），按 offset 数会错。
    窗口只数真实消息：摘要行自己也会落在最近 window 条里，算进去会挤掉一条对话。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        window: 需要保留为近期上下文的消息条数。
        since_id: 已处理的消息水位线，本次仅选更大的 ID。
        cap: 本批处理或选取的最大条数。
    """
    sub = (
        select(ConversationMessage.id.label("id"))
        .where(
            ConversationMessage.siwe_user_id == siwe_user_id,
            ConversationMessage.role != "summary",
        )
        .order_by(ConversationMessage.id.desc())
        .limit(window)
        .subquery()
    )
    window_start = (await db.execute(select(func.min(sub.c.id)).select_from(sub))).scalar_one_or_none()
    if window_start is None:
        return 0, None

    pending = (
        ConversationMessage.siwe_user_id == siwe_user_id,
        ConversationMessage.id > since_id,
        ConversationMessage.id < window_start,
        ConversationMessage.role != "summary",
    )
    count = (
        await db.execute(select(func.count()).select_from(ConversationMessage).where(*pending))
    ).scalar_one()
    ids = (
        (
            await db.execute(
                select(ConversationMessage.id)
                .where(*pending)
                .order_by(ConversationMessage.id.asc())
                .limit(cap)
            )
        )
        .scalars()
        .all()
    )
    return count, (ids[-1] if ids else None)


async def load_range(
    db: AsyncSession, siwe_user_id: str, *, after_id: int, until_id: int
) -> list[tuple[str, str]]:
    """取待摘要消息的 (role, content)，左开右闭，按 id 正序。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        after_id: 读取区间的左边界，不包含此消息 ID。
        until_id: 本批处理区间的最大消息 ID。
    """
    stmt = (
        select(ConversationMessage.role, ConversationMessage.content)
        .where(
            ConversationMessage.siwe_user_id == siwe_user_id,
            ConversationMessage.id > after_id,
            ConversationMessage.id <= until_id,
            ConversationMessage.role != "summary",
        )
        .order_by(ConversationMessage.id.asc())
    )
    return [(r.role, r.content) for r in (await db.execute(stmt)).all()]


async def pending_for_memory(
    db: AsyncSession, siwe_user_id: str, *, since_id: int, cap: int
) -> tuple[int, list[tuple[int, str, str]]]:
    """返回（水位线之后的消息总数，本批要处理的 (id, role, content)）。

    取最早的一批而不是最新的一批：积压时按顺序吃完，水位线才能单调推进。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        since_id: 已处理的消息水位线，本次仅选更大的 ID。
        cap: 本批处理或选取的最大条数。
    """
    pending = (
        ConversationMessage.siwe_user_id == siwe_user_id,
        ConversationMessage.id > since_id,
        ConversationMessage.role != "summary",
    )
    total = (
        await db.execute(select(func.count()).select_from(ConversationMessage).where(*pending))
    ).scalar_one()
    if total == 0:
        return 0, []
    rows = (
        await db.execute(
            select(ConversationMessage.id, ConversationMessage.role, ConversationMessage.content)
            .where(*pending)
            .order_by(ConversationMessage.id.asc())
            .limit(cap)
        )
    ).all()
    return total, [(r.id, r.role, r.content) for r in rows]
