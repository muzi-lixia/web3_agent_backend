"""EIP-4361（Sign-In with Ethereum）消息解析与验签。

只做三件事：把消息文本解析成字段、按配置校验 domain/uri/时效、从签名反推地址。
nonce 的签发与核销在 nonce_repo，本模块不碰数据库，方便单测。

解析刻意采用逐行读法而不是一条大正则：statement 允许跨行、可选字段允许缺席，
正则写出来的分组一旦失配，报错会指向一个无关的行号，排查成本高于自己写循环。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from eth_account import Account
from eth_account.messages import encode_defunct

from web3_agent.common.evm_address import normalize_address

# _HEADER_SUFFIX：SIWE 首行声明要求的固定结尾。
_HEADER_SUFFIX = " wants you to sign in with your Ethereum account:"
# _NONCE_RE：SIWE nonce 字符及最小长度校验规则。
_NONCE_RE = re.compile(r"^[A-Za-z0-9]{8,}$")
# _REQUIRED_KEYS：解析 SIWE 声明时必须具备的字段名。
_REQUIRED_KEYS = ("URI", "Version", "Chain ID", "Nonce", "Issued At")
# 地址之后是空行 + 可选 statement + 空行（EIP-4361 ABNF）。
# 判断"这一行是 statement 还是字段"，只能靠已知字段名，不能靠冒号——statement 里也可能有冒号。
# _FIELD_PREFIXES：SIWE 标准字段前缀，用于区分多行声明和键值字段。
_FIELD_PREFIXES = (
    "URI: ",
    "Version: ",
    "Chain ID: ",
    "Nonce: ",
    "Issued At: ",
    "Expiration Time: ",
    "Not Before: ",
    "Request ID: ",
    "Resources:",
)


class SiweError(Exception):
    """SIWE 校验失败。reason 只进审计，不进响应——对外统一是"登录状态无效"。"""

    def __init__(self, reason: str) -> None:
        """初始化 SiweError 所需的依赖和运行状态。

        Args:
            reason: 内部失败原因，用于异常和审计。
        """
        super().__init__(reason)
        self.reason = reason


# 数据职责：解析后的 SIWE 签名声明；不负责 nonce 的数据库消费。
@dataclass(frozen=True)
class SiweMessage:
    # domain：签名声明所属站点的域名，必须匹配服务端配置。
    domain: str
    # address：钱包地址；登录接口用它绑定签名身份，查询接口用它指定账户。
    address: str
    # uri：签名声明的站点 URI，必须匹配服务端配置。
    uri: str
    # version：SIWE 协议版本，当前验签仅接受字符串 1。
    version: str
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int
    # nonce：SIWE 一次性随机挑战值，验签成功后核销以防重放。
    nonce: str
    # issued_at：挑战或签名消息的签发时间。
    issued_at: datetime
    # statement：钱包签名时展示给用户的可选登录声明。
    statement: str | None = None
    # expiration_time：SIWE 签名消息的到期时间，时效校验要求提供。
    expiration_time: datetime | None = None
    # not_before：SIWE 消息最早可接受的时间；为空表示不额外限制。
    not_before: datetime | None = None
    # request_id：SIWE 消息可选的请求标识。
    request_id: str | None = None
    # resources：SIWE 消息附带的资源 URI 列表。
    resources: tuple[str, ...] = field(default_factory=tuple)


def _parse_time(value: str, key: str) -> datetime:
    """解析 SIWE ISO 时间文本；未带时区的值按 UTC 解释。

    Args:
        value: 待校验、转换或缓存的值。
        key: 当前字段名、缓存键或幂等键，具体以当前操作为准。
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SiweError(f"{key} 不是合法时间") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_message(text: str) -> SiweMessage:
    # 返回值：已拆分并校验基础字段的 SiweMessage；完整站点和时效校验由调用方继续执行。
    """逐行解析 SIWE 声明、地址、可选 statement 和标准字段，并校验必填项。

    Args:
        text: 待检查、解析或处理的原始文本。

    Raises:
        SiweError: 消息结构、地址、nonce 或必需字段不合法。
    """
    lines = text.splitlines()
    if len(lines) < 6:
        raise SiweError("消息行数不足")
    if not lines[0].endswith(_HEADER_SUFFIX):
        raise SiweError("首行不是 SIWE 声明")

    domain = lines[0][: -len(_HEADER_SUFFIX)]
    if not domain:
        raise SiweError("domain 为空")

    address = normalize_address(lines[1].strip())
    if address is None:
        raise SiweError("地址不合法")

    index = 2
    while index < len(lines) and lines[index].strip() == "":
        index += 1

    statement: str | None = None
    if index < len(lines) and not lines[index].startswith(_FIELD_PREFIXES):
        buffer: list[str] = []
        while index < len(lines) and lines[index].strip() != "":
            buffer.append(lines[index])
            index += 1
        statement = "\n".join(buffer) or None

    fields: dict[str, str] = {}
    resources: list[str] = []
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        if line.startswith("Resources:"):
            while index < len(lines) and lines[index].startswith("- "):
                resources.append(lines[index][2:].strip())
                index += 1
            continue
        # 只在第一个标准分隔符处分割，保留字段值中可能包含的 URI 冒号。
        key, sep, value = line.partition(": ")
        if not sep:
            raise SiweError(f"无法解析的行：{line[:32]}")
        fields[key] = value.strip()

    # 标准字段读取完成后统一检查必填项，再进行链编号和时间解析。
    missing = [k for k in _REQUIRED_KEYS if k not in fields]
    if missing:
        raise SiweError(f"缺少字段：{','.join(missing)}")

    try:
        chain_id = int(fields["Chain ID"])
    except ValueError as exc:
        raise SiweError("Chain ID 不是整数") from exc

    nonce = fields["Nonce"]
    if not _NONCE_RE.match(nonce):
        raise SiweError("nonce 格式不合法")

    return SiweMessage(
        domain=domain,
        address=address,
        uri=fields["URI"],
        version=fields["Version"],
        chain_id=chain_id,
        nonce=nonce,
        issued_at=_parse_time(fields["Issued At"], "Issued At"),
        statement=statement,
        expiration_time=(
            _parse_time(fields["Expiration Time"], "Expiration Time")
            if "Expiration Time" in fields
            else None
        ),
        not_before=(
            _parse_time(fields["Not Before"], "Not Before") if "Not Before" in fields else None
        ),
        request_id=fields.get("Request ID"),
        resources=tuple(resources),
    )


def validate_temporal(message: SiweMessage, *, now: datetime | None = None) -> None:
    """校验时效。没有 expirationTime 的消息永久有效，直接拒。

    Args:
        message: 按当前函数用途传入的用户文本或已解析 SIWE 消息。
        now: 注入的当前时间；为空时读取当前 UTC 时间。

    Raises:
        SiweError: 消息未设置到期时间、已过期或尚未生效。
    """
    moment = now or datetime.now(timezone.utc)
    # 签名声明必须明确到期，避免同一声明具有无限时间接受窗口。
    if message.expiration_time is None:
        raise SiweError("缺少 Expiration Time")
    if message.expiration_time <= moment:
        raise SiweError("消息已过期")
    if message.not_before is not None and message.not_before > moment:
        raise SiweError("消息尚未生效")


def validate_site(message: SiweMessage, *, expected_domain: str, expected_uri: str) -> None:
    """校验 SIWE 域名、URI 和协议版本，防止跨站登录消息被接受。

    Args:
        message: 按当前函数用途传入的用户文本或已解析 SIWE 消息。
        expected_domain: 服务端要求匹配的站点域名。
        expected_uri: 服务端要求匹配的站点 URI。

    Raises:
        SiweError: 站点域名、URI 或协议版本与要求不一致。
    """
    if message.domain.lower() != expected_domain.lower():
        raise SiweError(f"domain 不匹配：{message.domain}")
    if message.uri.rstrip("/") != expected_uri.rstrip("/"):
        raise SiweError(f"uri 不匹配：{message.uri}")
    if message.version != "1":
        raise SiweError(f"版本不支持：{message.version}")


def recover_address(text: str, signature: str) -> str:
    """从 EIP-191 签名反推地址，返回 EIP-55 写法。签名非法时抛 SiweError。

    Args:
        text: 待检查、解析或处理的原始文本。
        signature: 钱包对 SIWE 完整原文生成的 personal_sign 签名。

    Returns:
        签名恢复出的 EIP-55 钱包地址；非法签名抛 SiweError。

    Raises:
        SiweError: 签名无法解析或恢复地址。
    """
    try:
        recovered = Account.recover_message(encode_defunct(text=text), signature=signature)
    except Exception as exc:  # eth-account 对非法签名抛的异常类型不统一
        raise SiweError(f"签名无法解析：{type(exc).__name__}") from exc
    return recovered


def build_message(
    *,
    domain: str,
    address: str,
    uri: str,
    chain_id: int,
    nonce: str,
    issued_at: datetime,
    expiration_time: datetime,
    statement: str = "Sign in to Web3 Agent",
) -> str:
    """按 EIP-4361 生成消息。前端可用同一个函数对齐格式，测例也用它做往返验证。

    Args:
        domain: 写入 SIWE 消息的站点域名。
        address: 待校验、登录或查询的钱包地址。
        uri: 写入 SIWE 消息的站点 URI。
        chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
        nonce: 一次性登录挑战字符串。
        issued_at: 登录消息签发时间。
        expiration_time: 登录消息到期时间。
        statement: 钱包签名弹窗中展示的说明文本。
    """
    return (
        f"{domain} wants you to sign in with your Ethereum account:\n"
        f"{address}\n"
        f"\n"
        f"{statement}\n"
        f"\n"
        f"URI: {uri}\n"
        f"Version: 1\n"
        f"Chain ID: {chain_id}\n"
        f"Nonce: {nonce}\n"
        f"Issued At: {_iso(issued_at)}\n"
        f"Expiration Time: {_iso(expiration_time)}"
    )


def _iso(moment: datetime) -> str:
    """把时间转换为带毫秒和 Z 后缀的 UTC ISO 文本。

    Args:
        moment: 待格式化的带时区时间。
    """
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
