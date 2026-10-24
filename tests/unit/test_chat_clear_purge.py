"""清空/删除对话必须连带丢掉幂等缓存。

幂等响应里存的是上一轮的正文和引用，也就是用户刚要求删除的内容。
只删消息不删它，客户端拿同一个 key 重试就能把删掉的答案取回来。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import web3_agent.api.v1.chat_router as chat_router
from web3_agent.persistence.pg.repositories import idempotency_repo


class _Session:
    async def __aenter__(self):
        return SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(rowcount=1)))

    async def __aexit__(self, *exc_info):
        return False


def _run(monkeypatch, handler):
    monkeypatch.setattr(chat_router, "session_scope", lambda: _Session())
    purge = AsyncMock(return_value=2)
    monkeypatch.setattr(idempotency_repo, "purge_user", purge)
    monkeypatch.setattr(chat_router.conversation_repo, "bump_generation", AsyncMock(return_value=3))
    monkeypatch.setattr(chat_router.conversation_repo, "clear_user_messages", AsyncMock(return_value=4))
    monkeypatch.setattr(chat_router.profile_repo, "purge", AsyncMock(return_value=True))
    monkeypatch.setattr(chat_router.memory_repo, "purge", AsyncMock(return_value=5))
    container = SimpleNamespace(audit=SimpleNamespace(log=AsyncMock()))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=container)))
    identity = SimpleNamespace(user_id="user-1", address="0x" + "1" * 40, chain_id=1)
    asyncio.run(handler(request, identity))
    return purge


def test_clear_drops_idempotency_records(monkeypatch):
    purge = _run(monkeypatch, chat_router.clear)
    assert purge.await_args.kwargs["siwe_user_id"] == "user-1"


def test_purge_drops_idempotency_records(monkeypatch):
    purge = _run(monkeypatch, chat_router.purge_memory)
    assert purge.await_args.kwargs["siwe_user_id"] == "user-1"
