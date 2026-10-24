"""答案落库之后的辅助任务：失败只记日志，不能把这一轮变成 500。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import web3_agent.api.v1.chat_router as chat_router


def _identity() -> SimpleNamespace:
    return SimpleNamespace(user_id="user-1", address="0x" + "1" * 40)


def _run(monkeypatch, container):
    spawned: list = []
    monkeypatch.setattr(chat_router, "_spawn_background", lambda coro: spawned.append(coro))
    asyncio.run(chat_router._start_aux_tasks(container, _identity()))
    for coro in spawned:
        coro.close()
    return spawned


def test_summary_collect_failure_does_not_block_memory(monkeypatch):
    container = SimpleNamespace(
        summarizer=SimpleNamespace(collect=AsyncMock(side_effect=RuntimeError("db down")),
                                   run=AsyncMock()),
        memory=SimpleNamespace(collect=AsyncMock(return_value="job"), run=AsyncMock()),
    )
    spawned = _run(monkeypatch, container)
    assert len(spawned) == 1
    container.memory.collect.assert_awaited_once_with("user-1")


def test_memory_collect_failure_does_not_block_summary(monkeypatch):
    container = SimpleNamespace(
        summarizer=SimpleNamespace(collect=AsyncMock(return_value="job"), run=AsyncMock()),
        memory=SimpleNamespace(collect=AsyncMock(side_effect=RuntimeError("db down")),
                               run=AsyncMock()),
    )
    assert len(_run(monkeypatch, container)) == 1


def test_nothing_to_do_spawns_nothing(monkeypatch):
    container = SimpleNamespace(
        summarizer=SimpleNamespace(collect=AsyncMock(return_value=None), run=AsyncMock()),
        memory=SimpleNamespace(collect=AsyncMock(return_value=None), run=AsyncMock()),
    )
    assert _run(monkeypatch, container) == []
