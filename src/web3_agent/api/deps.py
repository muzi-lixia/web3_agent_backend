"""鉴权与限流依赖。

业务接口一律 require_identity；只有限流依赖用 optional_identity ——
未登录请求要先按 IP 计一次数再 401，直接用 require_identity 会让匿名刷接口不计费。
"""

from __future__ import annotations

from fastapi import Depends, Header, Request

from web3_agent.common.exceptions import E503_UNAUTHENTICATED, AppError
from web3_agent.config.settings import get_settings
from web3_agent.domain.auth.tokens import SiweClaims, decode
from web3_agent.persistence.pg.base import session_scope


class Identity:
    def __init__(self, claims: SiweClaims) -> None:
        self.user_id = claims.user_id
        self.address = claims.address
        self.chain_id = claims.chain_id


def _parse(authorization: str | None) -> Identity:
    if not authorization:
        raise AppError(*E503_UNAUTHENTICATED)
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AppError(*E503_UNAUTHENTICATED)
    settings = get_settings()
    return Identity(decode(secret=settings.jwt_secret, algorithm=settings.jwt_algorithm, token=token))


async def optional_identity(
    authorization: str | None = Header(default=None),
) -> Identity | None:
    if not authorization:
        return None
    return _parse(authorization)


async def require_identity(
    authorization: str | None = Header(default=None),
) -> Identity:
    return _parse(authorization)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def _enforce(request: Request, identity: Identity | None, *, auth_endpoint: bool) -> None:
    container = request.app.state.container
    key = container.quota.key_for(
        user_id=identity.user_id if identity else None, client_ip=client_ip(request)
    )
    async with session_scope() as db:
        await container.quota.check_request(
            db, key=key, logged_in=identity is not None, auth_endpoint=auth_endpoint
        )
    request.state.quota_key = key


async def chat_rate_limit(
    request: Request, identity: Identity | None = Depends(optional_identity)
) -> None:
    await _enforce(request, identity, auth_endpoint=False)


async def auth_rate_limit(request: Request) -> None:
    await _enforce(request, None, auth_endpoint=True)
