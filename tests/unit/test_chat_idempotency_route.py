"""幂等键在路由层的分支：重放、拒绝、失败后放行。

去重的正确性靠真库（tests/integration/test_chat_idempotency.py），
这里只管路由拿到的结论有没有被正确处理。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import web3_agent.api.v1.chat_router as chat_router
from web3_agent.api.schemas.chat_schema import ChatRequest
from web3_agent.common.exceptions import AppError


class _Session:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc_info):
        return False


def _container(handle=None):
    return SimpleNamespace(
        settings=SimpleNamespace(
            allowed_chain_ids={1},
            turn_timeout_seconds=5,
            turn_max_downstream_calls=24,
        ),
        agent=SimpleNamespace(
            handle=handle
            or AsyncMock(
                return_value=SimpleNamespace(
                    kind="answer", content="你好", steps=1, tool_calls=[], citations=[],
                    discarded=False,
                )
            )
        ),
        quota=SimpleNamespace(charge=AsyncMock()),
    )


def _run(monkeypatch, claim_result, *, container=None, key="key-abcdefgh", swallow=False):
    """跑一次 /chat。返回 (响应或异常, container, complete 桩, claim 桩, settle 桩)。"""
    container = container or _container()
    claim = AsyncMock(return_value=claim_result)
    complete = AsyncMock()
    settled = AsyncMock()
    monkeypatch.setattr(chat_router, "session_scope", lambda: _Session())
    monkeypatch.setattr(chat_router.idempotency_repo, "claim", claim)
    monkeypatch.setattr(chat_router.idempotency_repo, "complete", complete)
    monkeypatch.setattr(chat_router.idempotency_repo, "release", AsyncMock())
    monkeypatch.setattr(chat_router, "_settle_turn", settled)
    monkeypatch.setattr(chat_router, "_spawn_background", lambda coro: coro.close())
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=container)),
        state=SimpleNamespace(quota_key=None),
    )
    identity = SimpleNamespace(user_id="user-1", address="0x" + "1" * 40, chain_id=1)
    payload = ChatRequest(userMessage="你好", idempotencyKey=key)
    try:
        outcome = asyncio.run(chat_router.chat(payload, request, identity))
    except BaseException as exc:  # noqa: BLE001 只有 swallow 时才留在这里
        if not swallow:
            raise
        outcome = exc
    return outcome, container, complete, claim, settled


def test_completed_request_replays_without_calling_the_model(monkeypatch):
    cached = {"type": "answer", "content": "上次的答案", "data": {}}
    response, container, complete, _, _ = _run(monkeypatch, ("done", cached))
    assert response.content == "上次的答案"
    container.agent.handle.assert_not_awaited()
    complete.assert_not_awaited()


def test_request_still_running_is_rejected(monkeypatch):
    with pytest.raises(AppError) as error:
        _run(monkeypatch, ("running", None))
    assert error.value.code == "E511"
    # 上一轮跑完再拿同一个 key 就能拿到答案：这是「等一会儿可以重试」，
    # 不是「这条请求废了」。客户端要靠 Retry-After 决定等多久。
    assert error.value.retryable is True
    assert error.value.retry_after == 3


def test_same_key_with_new_content_is_rejected(monkeypatch):
    with pytest.raises(AppError) as error:
        _run(monkeypatch, ("conflict", None))
    assert error.value.code == "E512"
    # 换 key 才能跑，同一个 key 重试多少次都是这个结果。
    assert error.value.retryable is False


def test_successful_turn_stores_the_response(monkeypatch):
    response, _, complete, _, settled = _run(monkeypatch, ("new", None))
    assert response.content == "你好"
    stored = complete.await_args.kwargs
    assert stored["key"] == "key-abcdefgh"
    assert stored["response"]["content"] == "你好"
    # 存的是别名版本，取回来直接喂给 ChatResponse 就能还原。
    assert set(stored["response"]) >= {"type", "content", "data"}
    assert settled.await_args.kwargs["release"] is None


def test_failed_turn_releases_the_key_and_stores_nothing(monkeypatch):
    container = _container(handle=AsyncMock(side_effect=RuntimeError("boom")))
    outcome, _, complete, _, settled = _run(
        monkeypatch, ("new", None), container=container, swallow=True
    )
    assert isinstance(outcome, RuntimeError)
    complete.assert_not_awaited()
    assert settled.await_args.kwargs["release"] == ("user-1", "key-abcdefgh")


def test_discarded_turn_is_not_stored_and_releases_the_key(monkeypatch):
    """失效轮次在幂等语义上等于「没跑成」：存下来会让重试永远拿到空答案。"""
    container = _container(
        handle=AsyncMock(
            return_value=SimpleNamespace(
                kind="answer", content="", steps=0, tool_calls=[], citations=[],
                discarded=True,
            )
        )
    )
    response, _, complete, _, settled = _run(monkeypatch, ("new", None), container=container)
    assert response.content == ""
    assert response.data["discarded"] is True
    complete.assert_not_awaited()
    assert settled.await_args.kwargs["release"] == ("user-1", "key-abcdefgh")
