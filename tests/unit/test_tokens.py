"""SIWE 会话令牌：签发、解析、失效。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from web3_agent.common.exceptions import AppError
from web3_agent.domain.auth import tokens

SECRET = "test-secret-0123456789abcdefghij"
ALGORITHM = "HS256"
ADDRESS = "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed"


def _issue(ttl: int = 3600, now: datetime | None = None):
    return tokens.issue(
        secret=SECRET,
        algorithm=ALGORITHM,
        ttl_seconds=ttl,
        address=ADDRESS,
        chain_id=80094,
        now=now,
    )


def test_issue_and_decode_roundtrip() -> None:
    token, claims = _issue()
    decoded = tokens.decode(secret=SECRET, algorithm=ALGORITHM, token=token)

    assert decoded.address == ADDRESS
    assert decoded.chain_id == 80094
    assert decoded.user_id == claims.user_id == tokens.user_id_for(ADDRESS)


def test_user_id_is_uuid_and_leaks_no_address() -> None:
    uid = tokens.user_id_for(ADDRESS)
    assert uuid.UUID(uid)  # 合法 UUID
    assert ADDRESS not in uid
    assert "eip155" not in uid


def test_user_id_is_stable_across_checksum_case() -> None:
    mixed = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
    assert tokens.user_id_for(mixed) == tokens.user_id_for(mixed.lower())


def test_user_id_differs_per_address() -> None:
    other = "0x0000000000000000000000000000000000000001"
    assert tokens.user_id_for(ADDRESS) != tokens.user_id_for(other)


def test_decode_rejects_wrong_secret() -> None:
    token, _ = _issue()
    with pytest.raises(AppError) as exc:
        tokens.decode(secret="another-secret-3f8a19c47b0e526d7a", algorithm=ALGORITHM, token=token)
    assert exc.value.code == "E503"


def test_decode_rejects_expired_token() -> None:
    token, _ = _issue(ttl=1, now=datetime.now(timezone.utc) - timedelta(hours=1))
    with pytest.raises(AppError) as exc:
        tokens.decode(secret=SECRET, algorithm=ALGORITHM, token=token)
    assert exc.value.code == "E503"


def test_decode_rejects_tampered_token() -> None:
    token, _ = _issue()
    with pytest.raises(AppError):
        tokens.decode(secret=SECRET, algorithm=ALGORITHM, token=token[:-2] + "xy")


def test_decode_rejects_user_id_that_does_not_match_the_address() -> None:
    """签名有效但 id 和地址对不上：密钥泄漏时也不能拿 A 的 id 冒充 B。"""
    import jwt

    payload = {
        "sub": tokens.user_id_for("0x0000000000000000000000000000000000000001"),
        "siwe_user_id": tokens.user_id_for("0x0000000000000000000000000000000000000001"),
        "siwe_address": ADDRESS,
        "chain_id": 1,
        "exp": int(datetime.now(timezone.utc).timestamp()) + 600,
    }
    forged = jwt.encode(payload, SECRET, algorithm=ALGORITHM)
    with pytest.raises(AppError) as exc:
        tokens.decode(secret=SECRET, algorithm=ALGORITHM, token=forged)
    assert exc.value.code == "E503"


def test_decode_normalizes_the_address() -> None:
    """混合大小写通过 EIP-55 校验后，返回的地址统一成小写。"""
    token, _ = tokens.issue(
        secret=SECRET,
        algorithm=ALGORITHM,
        ttl_seconds=600,
        address="0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
        chain_id=1,
    )
    assert tokens.decode(secret=SECRET, algorithm=ALGORITHM, token=token).address == ADDRESS
