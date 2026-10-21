"""SIWE 鉴权接口（PRD 7.4）：签发 nonce、验签建会话。

失败一律返回 E503，不区分「domain 不对」「nonce 用过」「签名不匹配」——
区分开等于给攻击者一个探测接口。具体原因进审计日志。
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request

from web3_agent.api.deps import auth_rate_limit
from web3_agent.api.schemas.auth_schema import (
    NonceRequest,
    NonceResponse,
    VerifyRequest,
    VerifyResponse,
)
from web3_agent.common.evm_address import normalize_address, to_checksum_address
from web3_agent.common.exceptions import E102_CHAIN_NOT_ALLOWED, E503_UNAUTHENTICATED, AppError
from web3_agent.domain.auth import siwe, tokens
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import conversation_repo, nonce_repo

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(auth_rate_limit)])


def _short(address: str) -> str:
    return f"{address[:8]}…{address[-4:]}"


@router.post("/nonce", response_model=NonceResponse, response_model_by_alias=True)
async def issue_nonce(payload: NonceRequest, request: Request) -> NonceResponse:
    settings = request.app.state.container.settings
    address = normalize_address(payload.address)
    if address is None:
        raise AppError(*E503_UNAUTHENTICATED)
    if payload.chain_id not in settings.allowed_chain_ids:
        raise AppError(*E102_CHAIN_NOT_ALLOWED)

    nonce = secrets.token_hex(16)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=settings.siwe_nonce_ttl_seconds)
    async with session_scope() as db:
        await nonce_repo.issue(
            db,
            nonce=nonce,
            ttl_seconds=settings.siwe_nonce_ttl_seconds,
            address=address,
            chain_id=payload.chain_id,
        )

    message = siwe.build_message(
        domain=settings.siwe_domain,
        address=to_checksum_address(address),
        uri=settings.siwe_uri,
        chain_id=payload.chain_id,
        nonce=nonce,
        issued_at=now,
        expiration_time=expires_at,
    )
    return NonceResponse(nonce=nonce, expires_in=settings.siwe_nonce_ttl_seconds, message=message)


@router.post("/verify", response_model=VerifyResponse, response_model_by_alias=True)
async def verify(payload: VerifyRequest, request: Request) -> VerifyResponse:
    container = request.app.state.container
    settings = container.settings
    audit = container.audit

    try:
        message = siwe.parse_message(payload.message)
        siwe.validate_site(
            message, expected_domain=settings.siwe_domain, expected_uri=settings.siwe_uri
        )
        siwe.validate_temporal(message)
    except siwe.SiweError as exc:
        await audit.log("siwe_rejected", reason=exc.reason)
        raise AppError(*E503_UNAUTHENTICATED) from exc

    if message.chain_id not in settings.allowed_chain_ids:
        raise AppError(*E102_CHAIN_NOT_ALLOWED)

    # 先验签再核销：验签失败不该烧掉用户的 nonce
    recovered = siwe.recover_address(payload.message, payload.signature)
    if recovered.lower() != message.address:
        await audit.log("siwe_rejected", reason="签名地址与声明地址不一致")
        raise AppError(*E503_UNAUTHENTICATED)

    async with session_scope() as db:
        bound = await nonce_repo.consume(db, message.nonce)
        if bound == "":
            raise AppError(*E503_UNAUTHENTICATED)
        if bound and bound != message.address:
            raise AppError(*E503_UNAUTHENTICATED)

        token, claims = tokens.issue(
            secret=settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
            ttl_seconds=settings.jwt_ttl_seconds,
            address=message.address,
            chain_id=message.chain_id,
        )
        session_id = uuid.uuid4().hex
        await conversation_repo.touch_session(
            db,
            session_id=session_id,
            siwe_user_id=claims.user_id,
            wallet_address=claims.address,
            chain_id=claims.chain_id,
            ttl_seconds=settings.conversation_ttl_seconds,
        )

    await audit.log("siwe_login", siwe_user_id=claims.user_id, chain_id=claims.chain_id)
    return VerifyResponse(
        session_id=session_id,
        token=token,
        siwe_address=to_checksum_address(claims.address),
        siwe_address_short=_short(claims.address),
        expires_in=settings.jwt_ttl_seconds,
    )
