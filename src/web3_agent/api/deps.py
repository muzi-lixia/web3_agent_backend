"""鉴权与限流依赖。

业务接口一律 require_identity；只有限流依赖用 optional_identity ——
未登录请求要先按 IP 计一次数再 401，直接用 require_identity 会让匿名刷接口不计费。
"""

from __future__ import annotations

from ipaddress import ip_address

from fastapi import Header, Request

from web3_agent.common.exceptions import E503_UNAUTHENTICATED, AppError
from web3_agent.config.settings import get_settings
from web3_agent.domain.auth.tokens import SiweClaims, decode
from web3_agent.persistence.pg.base import session_scope


# 数据职责：接口层使用的已验证身份，由服务端 JWT claims 构建。
class Identity:
    def __init__(self, claims: SiweClaims) -> None:
        # user_id：由钱包地址稳定推导的用户标识，用于数据归属和配额。
        """从通过校验的 JWT 声明复制用户 ID、钱包和网络，供接口层使用。

        Args:
            claims: 通过 JWT 校验的身份声明。
        """
        self.user_id = claims.user_id
        # address：钱包地址；登录接口用它绑定签名身份，查询接口用它指定账户。
        self.address = claims.address
        # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
        self.chain_id = claims.chain_id


def _parse(authorization: str | None) -> Identity:
    """解析 Authorization Bearer 头并验证 JWT，返回服务端身份。

    Args:
        authorization: 原始 HTTP Authorization 头，格式为 Bearer <JWT>；未提供时为 None。
    """
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
    """在提供鉴权头时校验身份，未提供时返回 None。

    Args:
        authorization: 原始 HTTP Authorization 头，格式为 Bearer <JWT>；未提供时为 None。
    """
    if not authorization:
        return None
    return _parse(authorization)


async def require_identity(
    authorization: str | None = Header(default=None),
) -> Identity:
    """要求有效登录凭据；解析失败时抛未登录业务错误。

    Args:
        authorization: 原始 HTTP Authorization 头，格式为 Bearer <JWT>；未提供时为 None。
    """
    return _parse(authorization)


def client_ip(request: Request) -> str:
    """请求方地址。部署在反代之后按 trusted_proxy_hops 取 X-Forwarded-For。

    取错了不会报错，只会让所有匿名请求共用一个限流桶，所以这里对结果做一次
    IP 校验：头部是客户端可写的，脏值不能进限流键。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    hops = get_settings().trusted_proxy_hops
    if hops > 0:
        forwarded = request.headers.get("x-forwarded-for", "")
        chain = [part.strip() for part in forwarded.split(",") if part.strip()]
        if len(chain) >= hops:
            # 按部署声明的可信代理层数从转发链尾部取地址，不直接信任客户端最左侧值。
            candidate = chain[-hops]
            try:
                return str(ip_address(candidate))
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


async def _enforce(request: Request, identity: Identity | None, *, auth_endpoint: bool) -> None:
    """计算配额主体并检查当前接口额度，将主体键存入请求上下文。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
        auth_endpoint: 是否使用鉴权接口专用的请求额度。
    """
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
    request: Request, authorization: str | None = Header(default=None)
) -> None:
    """入口限流。

    凭据无效时按匿名计数，再交给业务依赖去 401 —— 如果这里直接抛，
    随便带一个坏 token 就能绕过整个计数器：依赖解析失败，计数一行都不会执行。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        authorization: 原始 HTTP Authorization 头，格式为 Bearer <JWT>；未提供时为 None。
    """
    try:
        # 有效凭据按用户归并额度；解析失败只在限流阶段退回匿名主体。
        identity = await optional_identity(authorization)
    except AppError:
        identity = None
    await _enforce(request, identity, auth_endpoint=False)


async def auth_rate_limit(request: Request) -> None:
    """按客户端 IP 应用鉴权接口专用请求额度。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    await _enforce(request, None, auth_endpoint=True)
