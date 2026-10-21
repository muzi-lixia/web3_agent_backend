"""conversation_session / conversation_message 读写。过期是惰性判定，没有后台清理任务。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.orm_models import ConversationMessage, ConversationSession


async def touch_session(
    db: AsyncSession,
    *,
    session_id: str,
    wallet_address: str | None,
    chain_id: int | None,
    ttl_seconds: int,
    siwe_user_id: str | None = None,
) -> datetime:
    """取会话；不存在或已过期就从空历史重新开始。返回本次的 expires_at。"""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)

    row = await db.get(ConversationSession, session_id)
    if row is None:
        db.add(
            ConversationSession(
                id=session_id,
                siwe_user_id=siwe_user_id,
                wallet_address=wallet_address,
                chain_id=chain_id,
                expires_at=expires_at,
            )
        )
        await db.flush()
        return expires_at

    if row.expires_at <= now:
        await db.execute(
            delete(ConversationMessage).where(ConversationMessage.session_id == session_id)
        )
        row.created_at = now

    row.siwe_user_id = siwe_user_id or row.siwe_user_id
    row.wallet_address = wallet_address or row.wallet_address
    row.chain_id = chain_id or row.chain_id
    row.last_seen_at = now
    row.expires_at = expires_at
    await db.flush()
    return expires_at


async def load_history(
    db: AsyncSession, session_id: str, limit: int
) -> list[tuple[str, str]]:
    """按时间正序返回最近 limit 条 (role, content)。"""
    stmt = (
        select(ConversationMessage.role, ConversationMessage.content)
        .where(ConversationMessage.session_id == session_id)
        .order_by(ConversationMessage.id.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [(r.role, r.content) for r in reversed(rows)]


async def append_message(
    db: AsyncSession,
    *,
    session_id: str,
    role: str,
    content: str,
    tool_calls: list | None = None,
) -> None:
    db.add(
        ConversationMessage(
            session_id=session_id, role=role, content=content, tool_calls=tool_calls
        )
    )
    await db.flush()
