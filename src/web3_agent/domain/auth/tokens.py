"""SIWE 会话令牌：签发与解析。HS256，无状态。

claims 里必须同时有 siwe_user_id 与 siwe_address：
user_id 用于配额/审计的归属键，address 用于写链路的身份一致性校验（E508）。
两者缺一就不能判定"这个 token 能不能操作这个 workflow"。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt

from web3_agent.common.evm_address import normalize_address
from web3_agent.common.exceptions import E503_UNAUTHENTICATED, AppError

_USER_NAMESPACE = uuid.UUID("f9989dd8-f823-4680-86c4-26f3eeb6729c")


@dataclass(frozen=True)
class SiweClaims:
    user_id: str
    address: str
    chain_id: int
    expires_at: datetime


def user_id_for(address: str) -> str:
    """用户主键由地址确定性推导成 UUID。

    不含地址本身，且同一钱包永远得到同一个 id —— 不另开用户表也能稳定归属。
    命名空间一旦有真实用户就不能改：改了所有人的 id 全变，历史数据全部失联。
    """
    return str(uuid.uuid5(_USER_NAMESPACE, address.lower()))


def issue(
    *,
    secret: str,
    algorithm: str,
    ttl_seconds: int,
    address: str,
    chain_id: int,
    now: datetime | None = None,
) -> tuple[str, SiweClaims]:
    moment = now or datetime.now(timezone.utc)
    expires_at = moment + timedelta(seconds=ttl_seconds)
    normalized = normalize_address(address)
    if normalized is None:
        raise AppError(*E503_UNAUTHENTICATED)
    claims = SiweClaims(
        user_id=user_id_for(normalized),
        address=normalized,
        chain_id=chain_id,
        expires_at=expires_at,
    )
    payload = {
        "sub": claims.user_id,
        "siwe_user_id": claims.user_id,
        "siwe_address": claims.address,
        "chain_id": chain_id,
        "iat": int(moment.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, secret, algorithm=algorithm), claims


def decode(*, secret: str, algorithm: str, token: str) -> SiweClaims:
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AppError(*E503_UNAUTHENTICATED) from exc
    except Exception as exc:  # noqa: BLE001 一切解码失败都按未授权处理
        raise AppError(*E503_UNAUTHENTICATED) from exc

    address = payload.get("siwe_address") or payload.get("sub")
    user_id = payload.get("siwe_user_id") or payload.get("sub") or ""
    if not address or not user_id:
        raise AppError(*E503_UNAUTHENTICATED)
    normalized = normalize_address(str(address))
    # 签发时 id 由地址推导，解码时重新推导一次：密钥泄漏后攻击者能签任意 claims，
    # 但把 id 和地址解耦不了，被冒用的仍然只能是那个地址自己的身份。
    if normalized is None or user_id != user_id_for(normalized):
        raise AppError(*E503_UNAUTHENTICATED)
    return SiweClaims(
        user_id=user_id,
        address=normalized,
        chain_id=int(payload.get("chain_id") or 0),
        expires_at=datetime.fromtimestamp(int(payload.get("exp") or 0), tz=timezone.utc),
    )
