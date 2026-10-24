"""后台任务的读取顺序：版本必须早于任何用户数据。

READ COMMITTED 下每条语句取新快照。先读消息、后取版本，中间提交的清空会让
「清空前读到的旧消息」贴上清空后的新版本，写回校验就成了摆设。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

from web3_agent.config.settings import Settings
from web3_agent.domain.memory import extractor as extractor_module
from web3_agent.domain.memory import summarizer as summarizer_module
from web3_agent.domain.memory.extractor import MemoryExtractor
from web3_agent.domain.memory.summarizer import Summarizer


@asynccontextmanager
async def _fake_scope():
    yield object()


def test_summarizer_reads_generation_before_user_data() -> None:
    async def run() -> list[str]:
        order: list[str] = []

        async def fake_generation(db, siwe_user_id):
            order.append("generation")
            return 7

        async def fake_summary_get(db, siwe_user_id):
            order.append("summary")
            return None

        async def fake_range(db, siwe_user_id, **kwargs):
            order.append("range")
            return 12, 5

        async def fake_load(db, siwe_user_id, **kwargs):
            order.append("messages")
            return [("user", "hi")]

        with (
            patch.object(summarizer_module, "session_scope", _fake_scope),
            patch.object(summarizer_module.conversation_repo, "current_generation", fake_generation),
            patch.object(summarizer_module.conversation_repo, "latest_summary", fake_summary_get),
            patch.object(summarizer_module.conversation_repo, "summarizable_range", fake_range),
            patch.object(summarizer_module.conversation_repo, "load_range", fake_load),
        ):
            job = await Summarizer(
                Settings(_env_file=None), llm=None, audit=None, quota=None  # type: ignore[arg-type]
            ).collect("0xabc")

        assert job is not None and job.generation == 7
        assert order.index("generation") < order.index("range")
        assert order.index("generation") < order.index("messages")
        return order

    assert asyncio.run(run())[0] == "generation"


def test_extractor_reads_generation_before_user_data() -> None:
    async def run() -> list[str]:
        order: list[str] = []

        async def fake_claim(db, siwe_user_id, token, ttl):
            order.append("claim")
            return True

        async def fake_generation(db, siwe_user_id):
            order.append("generation")
            return 9

        async def fake_state(db, siwe_user_id):
            order.append("state")
            return None

        async def fake_pending(db, siwe_user_id, **kwargs):
            order.append("messages")
            return 100, [(1, "user", "hi"), (2, "assistant", "hello")]

        async def fake_active(db, siwe_user_id):
            order.append("items")
            return []

        with (
            patch.object(extractor_module, "session_scope", _fake_scope),
            patch.object(extractor_module.memory_repo, "claim", fake_claim),
            patch.object(extractor_module.conversation_repo, "current_generation", fake_generation),
            patch.object(extractor_module.memory_repo, "get_state", fake_state),
            patch.object(extractor_module.conversation_repo, "pending_for_memory", fake_pending),
            patch.object(extractor_module.memory_repo, "active_items", fake_active),
        ):
            job = await MemoryExtractor(
                Settings(_env_file=None), llm=None, audit=None, quota=None  # type: ignore[arg-type]
            ).collect("0xabc")

        assert job is not None and job.generation == 9
        # 租约要最先抢（抢占是唯一的写入口），但版本必须排在任何读数据之前。
        assert order[0] == "claim"
        assert order.index("generation") < order.index("state")
        assert order.index("generation") < order.index("messages")
        assert order.index("generation") < order.index("items")
        return order

    asyncio.run(run())
