"""SIWE 与配额的端到端验证：走真实 PG，必要时走真实 HTTP。

本机没有 PG 时整组跳过，不让本地缺服务变成红灯。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from web3_agent.common.exceptions import AppError
from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import get_settings
from web3_agent.domain.quotas import QuotaController
from web3_agent.domain.auth import siwe
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.repositories import nonce_repo, quota_repo

ACCOUNT = Account.from_key("0x" + "33" * 32)
ADDRESS = ACCOUNT.address
CHAIN_ID = 80094


def _pg_ready() -> bool:
    async def probe() -> bool:
        try:
            pg.init_engine(get_settings())
            await pg.create_all()
            return True
        except Exception:  # noqa: BLE001 探测失败即跳过
            return False
        finally:
            await pg.dispose_engine()

    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg_ready(), reason="本机 PostgreSQL 不可用")


def test_nonce_is_single_use() -> None:
    nonce = "n" + uuid.uuid4().hex

    async def flow() -> tuple[str | None, str]:
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            async with pg.session_scope() as db:
                await nonce_repo.issue(db, nonce=nonce, ttl_seconds=60, address=ADDRESS)
            async with pg.session_scope() as db:
                first = await nonce_repo.consume(db, nonce)
            async with pg.session_scope() as db:
                second = await nonce_repo.consume(db, nonce)
            return first, second
        finally:
            await pg.dispose_engine()

    first, second = asyncio.run(flow())
    assert first == ADDRESS.lower()
    assert second == ""


def test_expired_nonce_cannot_be_consumed() -> None:
    nonce = "n" + uuid.uuid4().hex

    async def flow() -> str:
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            async with pg.session_scope() as db:
                await nonce_repo.issue(db, nonce=nonce, ttl_seconds=-1, address=ADDRESS)
            async with pg.session_scope() as db:
                return await nonce_repo.consume(db, nonce)
        finally:
            await pg.dispose_engine()

    assert asyncio.run(flow()) == ""


def test_quota_counts_and_raises_at_limit() -> None:
    key = "ip:test-" + uuid.uuid4().hex

    async def flow() -> int:
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            controller = QuotaController(get_settings())
            async with pg.session_scope() as db:
                await controller.hit(db, key=key, metric=QuotaMetric.QPS, limit=2)
                await controller.hit(db, key=key, metric=QuotaMetric.QPS, limit=2)
                with pytest.raises(AppError) as exc:
                    await controller.hit(db, key=key, metric=QuotaMetric.QPS, limit=2)
                assert exc.value.code == "E501"
                return await quota_repo.read(db, user_key=key, metric=QuotaMetric.QPS)
        finally:
            await pg.dispose_engine()

    # 被拦的那次也计了数：拦截依据就是计数本身，不回滚
    assert asyncio.run(flow()) == 3


def test_auth_flow_over_http() -> None:
    from fastapi.testclient import TestClient

    from web3_agent.main import app

    with TestClient(app) as client:
        nonce_resp = client.post(
            "/api/v1/auth/nonce", json={"address": ADDRESS, "chainId": CHAIN_ID}
        )
        assert nonce_resp.status_code == 200, nonce_resp.text
        message = nonce_resp.json()["message"]

        signed = ACCOUNT.sign_message(encode_defunct(text=message))
        verify_resp = client.post(
            "/api/v1/auth/verify",
            json={"message": message, "signature": signed.signature.to_0x_hex()},
        )
        assert verify_resp.status_code == 200, verify_resp.text
        body = verify_resp.json()
        assert body["siweAddress"] == ADDRESS
        assert body["token"]

        # nonce 已核销，同一个签名不能换第二个 token
        replay = client.post(
            "/api/v1/auth/verify",
            json={"message": message, "signature": signed.signature.to_0x_hex()},
        )
        assert replay.status_code == 401
        assert replay.json()["errorCode"] == "E503"


def test_verify_rejects_message_signed_by_another_wallet() -> None:
    from fastapi.testclient import TestClient

    from web3_agent.main import app

    attacker = Account.from_key("0x" + "44" * 32)
    with TestClient(app) as client:
        resp = client.post("/api/v1/auth/nonce", json={"address": ADDRESS, "chainId": CHAIN_ID})
        message = resp.json()["message"]

        stolen = attacker.sign_message(encode_defunct(text=message))
        forged = client.post(
            "/api/v1/auth/verify",
            json={"message": message, "signature": stolen.signature.to_0x_hex()},
        )
        assert forged.status_code == 401
        assert forged.json()["errorCode"] == "E503"


def test_chat_returns_429_when_qps_bucket_is_full() -> None:
    """把当前分钟的匿名计数打满，/chat 必须在进模型之前就被 429 拦下。"""
    from fastapi.testclient import TestClient

    from web3_agent.main import app

    async def fill() -> None:
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            async with pg.session_scope() as db:
                await quota_repo.incr(
                    db, user_key="ip:testclient", metric=QuotaMetric.QPS, amount=10_000
                )
        finally:
            await pg.dispose_engine()

    asyncio.run(fill())

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat",
            json={"sessionId": "s-" + uuid.uuid4().hex, "userMessage": "查一下我的余额"},
        )
        assert resp.status_code == 429
        assert resp.json()["errorCode"] == "E501"
        assert resp.json()["retryable"] is True


def test_siwe_rejects_message_from_other_domain() -> None:
    text = siwe.build_message(
        domain="evil.example",
        address=ADDRESS,
        uri="http://localhost:5173",
        chain_id=CHAIN_ID,
        nonce="abcdef0123456789",
        issued_at=datetime.now(timezone.utc),
        expiration_time=datetime.now(timezone.utc).replace(year=2099),
    )
    parsed = siwe.parse_message(text)
    settings = get_settings()
    with pytest.raises(siwe.SiweError):
        siwe.validate_site(
            parsed, expected_domain=settings.siwe_domain, expected_uri=settings.siwe_uri
        )
