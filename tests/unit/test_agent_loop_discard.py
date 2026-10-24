"""失效轮次的收尾：不落库只是底线，正文也不能交回去。

读历史之后用户点了清空，这一轮既要写不进库，也不能把基于已删内容生成的答案
交回给调用方 —— 否则界面上就是「清了又冒出来」。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import web3_agent.agent.agent_loop as agent_loop_module
from web3_agent.agent.agent_loop import AgentLoop
from web3_agent.common.types import ChatOutcome, ToolCallRecord


class _Scope:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc_info):
        return False


def _bare_loop(log: AsyncMock) -> AgentLoop:
    """`_finish` 只用到审计和会话写入，绕开需要整张依赖图的构造。"""
    loop = AgentLoop.__new__(AgentLoop)
    loop._audit = SimpleNamespace(log=log)
    return loop


def _outcome() -> ChatOutcome:
    return ChatOutcome(
        kind="answer",
        content="这是基于已删除历史生成的答案",
        tool_calls=[ToolCallRecord(name="get_balance", args={"address": "0x1"})],
        steps=2,
        citations=[{"id": "S1", "chunk_id": "c1"}],
    )


def test_stale_turn_returns_nothing_renderable(monkeypatch) -> None:
    async def run() -> None:
        save, log = AsyncMock(), AsyncMock()

        async def _stale(db, siwe_user_id, generation):
            return False

        monkeypatch.setattr(agent_loop_module, "session_scope", lambda: _Scope())
        monkeypatch.setattr(agent_loop_module.conversation_repo, "assert_generation", _stale)
        monkeypatch.setattr(agent_loop_module.conversation_repo, "append_message", save)

        result = await _bare_loop(log)._finish(
            "u-1", _outcome(), generation=3, turn_id="t-1"
        )

        assert result.discarded is True
        assert result.content == ""
        # 引用和工具轨迹也是同一次生成的产物，留着照样能拼出旧内容。
        assert result.citations == []
        assert result.tool_calls == []
        assert result.steps == 0
        assert result.kind == "answer"
        save.assert_not_awaited()
        log.assert_awaited()

    asyncio.run(run())


def test_live_turn_is_saved_unchanged(monkeypatch) -> None:
    async def run() -> None:
        save, log = AsyncMock(), AsyncMock()

        async def _fresh(db, siwe_user_id, generation):
            return True

        monkeypatch.setattr(agent_loop_module, "session_scope", lambda: _Scope())
        monkeypatch.setattr(agent_loop_module.conversation_repo, "assert_generation", _fresh)
        monkeypatch.setattr(agent_loop_module.conversation_repo, "append_message", save)

        result = await _bare_loop(log)._finish(
            "u-1", _outcome(), generation=3, turn_id="t-1"
        )

        assert result.discarded is False
        assert result.content == "这是基于已删除历史生成的答案"
        save.assert_awaited_once()

    asyncio.run(run())
