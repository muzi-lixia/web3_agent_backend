"""Small production pools must not silently disable multi-step plans."""
import asyncio
from unittest.mock import AsyncMock

from psycopg_pool import AsyncConnectionPool

from web3_agent.agent.checkpointer import AsyncPostgresSaver, init_checkpointer
from web3_agent.config.settings import Settings


def test_single_connection_pool_keeps_checkpointer_enabled(monkeypatch):
    # Keep real pool construction: it validates min_size <= max_size.
    # Replace network work only, so this test needs no running PostgreSQL.
    monkeypatch.setattr(AsyncConnectionPool, "open", AsyncMock())
    monkeypatch.setattr(AsyncPostgresSaver, "setup", AsyncMock())

    async def run():
        ready = await init_checkpointer(Settings(_env_file=None, checkpoint_pool_size=1))
        assert ready is not None
        _, pool = ready
        try:
            assert pool.max_size == 1
            assert pool.min_size == 0
        finally:
            await pool.close()

    asyncio.run(run())
