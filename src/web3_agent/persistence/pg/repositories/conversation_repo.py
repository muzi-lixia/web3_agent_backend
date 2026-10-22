"""conversation_message 读写。会话不是表，归属只有 siwe_user_id 一个来源。"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.orm_models import ConversationMessage


async def load_history(
    db: AsyncSession, siwe_user_id: str, limit: int, max_chars: int
) -> list[tuple[str, str]]:
    """按 id 正序返回最近 limit 条 (role, content)，再按字符预算从旧到新裁掉。"""
    stmt = (
        select(ConversationMessage.role, ConversationMessage.content)
        .where(ConversationMessage.siwe_user_id == siwe_user_id)
        .order_by(ConversationMessage.id.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return _fit_budget([(r.role, r.content) for r in reversed(rows)], max_chars)


def _fit_budget(
    messages: list[tuple[str, str]], max_chars: int
) -> list[tuple[str, str]]:
    """从最新一条往回装，装不下就丢掉更早的，保留一段连续的结尾。

    最新一条永远保留：用户下一句多半就在问它，截断它等于把上下文切成两半。
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
    """多取一条用来判 hasMore。"""
    stmt = select(ConversationMessage).where(ConversationMessage.siwe_user_id == siwe_user_id)
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
) -> int:
    row = ConversationMessage(
        siwe_user_id=siwe_user_id, role=role, content=content, tool_calls=tool_calls
    )
    db.add(row)
    await db.flush()
    return row.id


async def clear_user_messages(db: AsyncSession, siwe_user_id: str) -> int:
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
    """
    sub = (
        select(ConversationMessage.id.label("id"))
        .where(ConversationMessage.siwe_user_id == siwe_user_id)
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
    """取待摘要消息的 (role, content)，左开右闭，按 id 正序。"""
    stmt = (
        select(ConversationMessage.role, ConversationMessage.content)
        .where(
            ConversationMessage.siwe_user_id == siwe_user_id,
            ConversationMessage.id > after_id,
            ConversationMessage.id <= until_id,
        )
        .order_by(ConversationMessage.id.asc())
    )
    return [(r.role, r.content) for r in (await db.execute(stmt)).all()]


async def pending_for_memory(
    db: AsyncSession, siwe_user_id: str, *, since_id: int, cap: int
) -> tuple[int, list[tuple[int, str, str]]]:
    """返回（水位线之后的消息总数，本批要处理的 (id, role, content)）。

    取最早的一批而不是最新的一批：积压时按顺序吃完，水位线才能单调推进。
    """
    pending = (
        ConversationMessage.siwe_user_id == siwe_user_id,
        ConversationMessage.id > since_id,
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
