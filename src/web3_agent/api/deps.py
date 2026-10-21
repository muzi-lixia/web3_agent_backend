"""鉴权依赖：解析 Authorization: Bearer 的 SIWE JWT。

V1a 只读匿名可访问，所以这里是"可选身份"；拿到就带 siwe_address 做归属校验，
拿不到也不拒绝。写接口必须改用 require_identity。
"""

from __future__ import annotations

from fastapi import Header

from web3_agent.common.exceptions import AppError
from web3_agent.config.settings import get_settings

try:
    import jwt
except ImportError:  # pragma: no cover 依赖未装时明确失败，不静默放行
    jwt = None


class Identity:
    def __init__(self, siwe_user_id: str, siwe_address: str) -> None:
        self.siwe_user_id = siwe_user_id
        self.siwe_address = siwe_address


def _decode(token: str) -> Identity:
    if jwt is None:
        raise AppError("E401", "鉴权组件未就绪", 401)
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except Exception as exc:  # noqa: BLE001 一切解码失败都按未授权处理
        raise AppError("E401", "登录状态无效，请重新连接钱包", 401) from exc

    address = payload.get("siwe_address") or payload.get("sub")
    user_id = payload.get("siwe_user_id") or ""
    if not address:
        raise AppError("E401", "登录状态无效，请重新连接钱包", 401)
    return Identity(siwe_user_id=user_id, siwe_address=address)


async def optional_identity(
    authorization: str | None = Header(default=None),
) -> Identity | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AppError("E401", "登录状态无效，请重新连接钱包", 401)
    return _decode(token)


async def require_identity(
    authorization: str | None = Header(default=None),
) -> Identity:
    if not authorization:
        raise AppError("E401", "该操作需要连接钱包", 401)
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AppError("E401", "登录状态无效，请重新连接钱包", 401)
    return _decode(token)
