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

_HEADER_SUFFIX = " wants you to sign in with your Ethereum account:"
_NONCE_RE = re.compile(r"^[A-Za-z0-9]{8,}$")
_REQUIRED_KEYS = ("URI", "Version", "Chain ID", "Nonce", "Issued At")
# 地址之后是空行 + 可选 statement + 空行（EIP-4361 ABNF）。
# 判断"这一行是 statement 还是字段"，只能靠已知字段名，不能靠冒号——statement 里也可能有冒号。
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
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SiweMessage:
    domain: str
    address: str
    uri: str
    version: str
    chain_id: int
    nonce: str
    issued_at: datetime
    statement: str | None = None
    expiration_time: datetime | None = None
    not_before: datetime | None = None
    request_id: str | None = None
    resources: tuple[str, ...] = field(default_factory=tuple)


def _parse_time(value: str, key: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SiweError(f"{key} 不是合法时间") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_message(text: str) -> SiweMessage:
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
        key, sep, value = line.partition(": ")
        if not sep:
            raise SiweError(f"无法解析的行：{line[:32]}")
        fields[key] = value.strip()

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
    """校验时效。没有 expirationTime 的消息永久有效，直接拒。"""
    moment = now or datetime.now(timezone.utc)
    if message.expiration_time is None:
        raise SiweError("缺少 Expiration Time")
    if message.expiration_time <= moment:
        raise SiweError("消息已过期")
    if message.not_before is not None and message.not_before > moment:
        raise SiweError("消息尚未生效")


def validate_site(message: SiweMessage, *, expected_domain: str, expected_uri: str) -> None:
    if message.domain.lower() != expected_domain.lower():
        raise SiweError(f"domain 不匹配：{message.domain}")
    if message.uri.rstrip("/") != expected_uri.rstrip("/"):
        raise SiweError(f"uri 不匹配：{message.uri}")
    if message.version != "1":
        raise SiweError(f"版本不支持：{message.version}")


def recover_address(text: str, signature: str) -> str:
    """从 EIP-191 签名反推地址，返回 EIP-55 写法。签名非法时抛 SiweError。"""
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
    """按 EIP-4361 生成消息。前端可用同一个函数对齐格式，测例也用它做往返验证。"""
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
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
