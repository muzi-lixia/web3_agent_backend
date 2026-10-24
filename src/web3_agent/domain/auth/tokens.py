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

# _USER_NAMESPACE：由钱包地址推导稳定用户 UUID 的固定命名空间，不应随部署变化。
_USER_NAMESPACE = uuid.UUID("f9989dd8-f823-4680-86c4-26f3eeb6729c")


# 数据职责：JWT 验证后提取的不可变登录身份，供接口鉴权与数据归属使用。
@dataclass(frozen=True)
class SiweClaims:
    # user_id：由钱包地址稳定推导的用户标识，用于数据归属和配额。
    user_id: str
    # address：验签成功并归一化的钱包地址。
    address: str
    # chain_id：签发此 JWT 时验证的网络编号，聊天请求的网络必须与之匹配。
    chain_id: int
    # expires_at：凭据或租约失效时间；超过后不能继续使用。
    expires_at: datetime


def user_id_for(address: str) -> str:
    """用户主键由地址确定性推导成 UUID。

    不含地址本身，且同一钱包永远得到同一个 id —— 不另开用户表也能稳定归属。
    命名空间一旦有真实用户就不能改：改了所有人的 id 全变，历史数据全部失联。

    Args:
        address: 待校验、登录或查询的钱包地址。
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
    # JWT 字段 sub/siwe_user_id 为稳定用户 ID，siwe_address 为钱包，chain_id 为已验证网络。
    # iat/exp 为 Unix 秒级签发与到期时间，jti 为本令牌随机唯一编号。
    """规范化地址并生成有到期时间的 JWT，同时返回解析后的身份字段。

    Args:
        secret: 用于签发或校验 JWT 的 HMAC 密钥。
        algorithm: 被允许的 JWT 签名算法。
        ttl_seconds: 挑战、凭据或租约的有效秒数。
        address: 待校验、登录或查询的钱包地址。
        chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
        now: 注入的当前时间；为空时读取当前 UTC 时间。

    Returns:
        tuple[str, SiweClaims]: 签名 JWT 与对应的身份声明。
    """
    moment = now or datetime.now(timezone.utc)
    expires_at = moment + timedelta(seconds=ttl_seconds)
    # 将身份地址归一化后再推导用户 ID，避免大小写导致同一钱包分裂归属。
    normalized = normalize_address(address)
    if normalized is None:
        raise AppError(*E503_UNAUTHENTICATED)
    claims = SiweClaims(
        user_id=user_id_for(normalized),
        address=normalized,
        chain_id=chain_id,
        expires_at=expires_at,
    )
    # 构造要签入 JWT 的身份声明和时效字段，后续验证只信任签名通过的声明。
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
    """校验 JWT 签名与时效，核对用户 ID 和地址关系，返回内部身份对象。

    Args:
        secret: 用于签发或校验 JWT 的 HMAC 密钥。
        algorithm: 被允许的 JWT 签名算法。
        token: 待验证的 JWT 字符串。

    Returns:
        SiweClaims: 已验证的用户标识、钱包、网络和到期时间。

    Raises:
        AppError: JWT 无效、已过期，或身份声明与钱包地址不匹配。
    """
    try:
        # 仅按服务端配置的算法验证签名与标准时效，不接受令牌自行指定其他算法。
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AppError(*E503_UNAUTHENTICATED) from exc
    except Exception as exc:  # noqa: BLE001 一切解码失败都按未授权处理
        raise AppError(*E503_UNAUTHENTICATED) from exc

    address = payload.get("siwe_address") or payload.get("sub")
    user_id = payload.get("siwe_user_id") or payload.get("sub") or ""
    if not address or not user_id:
        raise AppError(*E503_UNAUTHENTICATED)
    # 验签通过后仍核对地址与稳定用户 ID 的绑定关系。
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
