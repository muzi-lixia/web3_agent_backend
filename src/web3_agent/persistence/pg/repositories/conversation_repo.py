"""conversation_message 读写。会话不是表，归属只有 siwe_user_id 一个来源。"""

from __future__ import annotations

from sqlalchemy import delete, select
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
