"""无效凭据不能绕过入口限流。

`chat_rate_limit` 原来依赖 `optional_identity`：凭据无效时依赖解析直接失败，
计数一行都不执行。带一个随便编的 Bearer 就能无限打接口，限流形同不存在。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import web3_agent.api.deps as deps_module
from web3_agent.common.exceptions import E503_UNAUTHENTICATED, AppError
from web3_agent.domain.auth.tokens import SiweClaims


class _Scope:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc_info):
        return False


def _quota(check: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        key_for=lambda *, user_id, client_ip: f"u:{user_id}" if user_id else f"ip:{client_ip}",
        check_request=check,
    )


def _request(quota, forwarded: str | None = None) -> SimpleNamespace:
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(quota=quota))),
        state=SimpleNamespace(),
        client=SimpleNamespace(host="203.0.113.7"),
        headers=headers,
    )


def test_invalid_token_is_counted_as_anonymous(monkeypatch) -> None:
    async def run() -> None:
        def _reject(**kwargs):
            raise AppError(*E503_UNAUTHENTICATED)

        check = AsyncMock()
        monkeypatch.setattr(deps_module, "decode", _reject)
        monkeypatch.setattr(deps_module, "session_scope", lambda: _Scope())

        await deps_module.chat_rate_limit(_request(_quota(check)), "Bearer not-a-real-token")

        check.assert_awaited_once()
        assert check.await_args.kwargs["key"] == "ip:203.0.113.7"
        assert check.await_args.kwargs["logged_in"] is False

    asyncio.run(run())


def test_valid_token_is_counted_per_user(monkeypatch) -> None:
    async def run() -> None:
        claims = SiweClaims(
            user_id="u-9",
            address="0x" + "1" * 40,
            chain_id=1,
            expires_at=datetime.now(timezone.utc),
        )
        check = AsyncMock()
        monkeypatch.setattr(deps_module, "decode", lambda **kwargs: claims)
        monkeypatch.setattr(deps_module, "session_scope", lambda: _Scope())

        await deps_module.chat_rate_limit(_request(_quota(check)), "Bearer good")

        assert check.await_args.kwargs["key"] == "u:u-9"
        assert check.await_args.kwargs["logged_in"] is True

    asyncio.run(run())


def _proxy(hops: int, monkeypatch) -> None:
    monkeypatch.setattr(
        deps_module, "get_settings", lambda: SimpleNamespace(trusted_proxy_hops=hops)
    )


def test_forwarded_header_is_ignored_without_a_proxy_in_front(monkeypatch) -> None:
    """直连部署时 XFF 是客户端自己写的，照抄就等于把限流键交给调用方。"""
    _proxy(0, monkeypatch)
    request = _request(SimpleNamespace(), forwarded="198.51.100.9")
    assert deps_module.client_ip(request) == "203.0.113.7"


def test_forwarded_header_is_used_behind_one_proxy(monkeypatch) -> None:
    _proxy(1, monkeypatch)
    request = _request(SimpleNamespace(), forwarded="198.51.100.9")
    assert deps_module.client_ip(request) == "198.51.100.9"


def test_forwarded_chain_counts_hops_from_the_right(monkeypatch) -> None:
    """两层反代：[真实客户端, 第一层代理]，取倒数第二项。"""
    _proxy(2, monkeypatch)
    request = _request(SimpleNamespace(), forwarded="198.51.100.9, 10.0.0.2")
    assert deps_module.client_ip(request) == "198.51.100.9"


def test_malformed_forwarded_value_falls_back_to_the_peer(monkeypatch) -> None:
    _proxy(1, monkeypatch)
    request = _request(SimpleNamespace(), forwarded="not-an-ip")
    assert deps_module.client_ip(request) == "203.0.113.7"


def test_bad_token_still_reaches_the_counter_in_a_real_request(monkeypatch) -> None:
    """端到端确认顺序：限流挂在 router 上，先于 require_identity 执行。"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from web3_agent.api.v1 import chat_router

    def _reject(**kwargs):
        raise AppError(*E503_UNAUTHENTICATED)

    check = AsyncMock()
    app = FastAPI()

    @app.exception_handler(AppError)
    async def _handle(request, exc: AppError):
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.message},
        )

    app.state.container = SimpleNamespace(quota=_quota(check))
    app.include_router(chat_router.router)

    monkeypatch.setattr(deps_module, "decode", _reject)
    monkeypatch.setattr(deps_module, "session_scope", lambda: _Scope())

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"userMessage": "我的余额"},
            headers={"Authorization": "Bearer bad"},
        )

    assert response.status_code == 401
    check.assert_awaited_once()
    assert check.await_args.kwargs["logged_in"] is False
