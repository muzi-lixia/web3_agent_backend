"""conversation_message 读写。会话不是表，归属只有 siwe_user_id 一个来源。"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.orm_models import ConversationMessage


async def load_history(
    db: AsyncSession, siwe_user_id: str, limit: int
) -> list[tuple[str, str]]:
    """按 id 正序返回最近 limit 条 (role, content)。"""
    stmt = (
        select(ConversationMessage.role, ConversationMessage.content)
        .where(ConversationMessage.siwe_user_id == siwe_user_id)
        .order_by(ConversationMessage.id.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [(r.role, r.content) for r in reversed(rows)]


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
