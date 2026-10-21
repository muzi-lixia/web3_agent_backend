"""EIP-4361 解析、时效校验、验签。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from eth_account import Account

from web3_agent.domain.auth import siwe

PRIVATE_KEY = "0x" + "11" * 32
ACCOUNT = Account.from_key(PRIVATE_KEY)
OTHER_ACCOUNT = Account.from_key("0x" + "22" * 32)
DOMAIN = "localhost"
URI = "http://localhost:5173"


def _message(
    *,
    address: str = ACCOUNT.address,
    domain: str = DOMAIN,
    uri: str = URI,
    nonce: str = "abcdef0123456789",
    expires_in: int = 300,
) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=expires_in)
    text = siwe.build_message(
        domain=domain,
        address=address,
        uri=uri,
        chain_id=80094,
        nonce=nonce,
        issued_at=now,
        expiration_time=expires_at,
    )
    return text, expires_at


def test_build_and_parse_roundtrip() -> None:
    text, _ = _message()
    parsed = siwe.parse_message(text)

    assert parsed.domain == DOMAIN
    assert parsed.address == ACCOUNT.address.lower()
    assert parsed.uri == URI
    assert parsed.version == "1"
    assert parsed.chain_id == 80094
    assert parsed.nonce == "abcdef0123456789"
    assert parsed.statement == "Sign in to Web3 Agent"


def test_parse_message_without_statement() -> None:
    text, _ = _message()
    lines = text.splitlines()
    # 去掉 statement 与其后的空行：EIP-4361 允许无 statement
    stripped = "\n".join(lines[:1] + [lines[1], ""] + lines[4:])
    parsed = siwe.parse_message(stripped)
    assert parsed.statement is None
    assert parsed.nonce == "abcdef0123456789"


def test_parse_rejects_bad_header() -> None:
    text, _ = _message()
    with pytest.raises(siwe.SiweError):
        siwe.parse_message(text.replace("wants you to sign in", "asks you to sign in"))


def test_parse_rejects_bad_address() -> None:
    text, _ = _message()
    with pytest.raises(siwe.SiweError):
        siwe.parse_message(text.replace(ACCOUNT.address, "0x" + "1" * 39, 1))


def test_parse_rejects_short_nonce() -> None:
    text, _ = _message(nonce="abc123")
    with pytest.raises(siwe.SiweError):
        siwe.parse_message(text)


def test_validate_site_accepts_same_origin() -> None:
    parsed = siwe.parse_message(_message()[0])
    siwe.validate_site(parsed, expected_domain=DOMAIN, expected_uri=URI)


def test_validate_site_rejects_other_domain() -> None:
    parsed = siwe.parse_message(_message(domain="evil.example")[0])
    with pytest.raises(siwe.SiweError):
        siwe.validate_site(parsed, expected_domain=DOMAIN, expected_uri=URI)


def test_validate_site_rejects_other_uri() -> None:
    parsed = siwe.parse_message(_message(uri="https://evil.example")[0])
    with pytest.raises(siwe.SiweError):
        siwe.validate_site(parsed, expected_domain=DOMAIN, expected_uri=URI)


def test_validate_temporal_rejects_expired() -> None:
    parsed = siwe.parse_message(_message(expires_in=-1)[0])
    with pytest.raises(siwe.SiweError):
        siwe.validate_temporal(parsed)


def test_validate_temporal_accepts_live_message() -> None:
    parsed = siwe.parse_message(_message()[0])
    siwe.validate_temporal(parsed)


def test_recover_address_matches_signer() -> None:
    text, _ = _message()
    signature = ACCOUNT.sign_message(_encode(text)).signature
    assert siwe.recover_address(text, signature) == ACCOUNT.address


def test_recover_address_differs_for_other_signer() -> None:
    text, _ = _message()
    signature = OTHER_ACCOUNT.sign_message(_encode(text)).signature
    assert siwe.recover_address(text, signature) != ACCOUNT.address


def test_recover_address_rejects_garbage_signature() -> None:
    text, _ = _message()
    with pytest.raises(siwe.SiweError):
        siwe.recover_address(text, "0x" + "00" * 65)


def _encode(text: str):
    from eth_account.messages import encode_defunct

    return encode_defunct(text=text)
