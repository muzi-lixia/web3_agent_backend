"""ConversationContext：PG 短期会话。只存对话，不存用户档案。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.repositories import conversation_repo


class ConversationContext:
    def __init__(self, max_messages: int, max_chars: int) -> None:
        self._max = max_messages
        self._max_chars = max_chars

    async def load(
        self,
        db: AsyncSession,
        *,
        siwe_user_id: str
    ) -> list[tuple[str, str]]:
        return await conversation_repo.load_history(
            db, siwe_user_id, self._max, self._max_chars
        )

    async def save(
        self,
        db: AsyncSession,
        *,
        siwe_user_id: str,
        role: str,
        content: str,
        tool_calls: list | None = None,
    ) -> None:
        await conversation_repo.append_message(
            db, siwe_user_id=siwe_user_id, role=role, content=content, tool_calls=tool_calls
        )
