"""ConversationContext：PG 短期会话。只存对话，不存用户档案。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.repositories import conversation_repo


class ConversationContext:
    def __init__(self, ttl_seconds: int, max_messages: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_messages

    async def load(
        self,
        db: AsyncSession,
        *,
        session_id: str,
        wallet_address: str | None,
        chain_id: int | None,
    ) -> list[tuple[str, str]]:
        await conversation_repo.touch_session(
            db,
            session_id=session_id,
            wallet_address=wallet_address,
            chain_id=chain_id,
            ttl_seconds=self._ttl,
        )
        return await conversation_repo.load_history(db, session_id, self._max)

    async def save(
        self,
        db: AsyncSession,
        *,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list | None = None,
    ) -> None:
        await conversation_repo.append_message(
            db, session_id=session_id, role=role, content=content, tool_calls=tool_calls
        )
