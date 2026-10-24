"""EVM 地址：格式判断、EIP-55 校验和、小写标准化。

结构化地址一律走这里做拒绝式校验（PRD 8.1），不做截断清洗——
截断过的地址仍然是"看起来合法"的地址，比直接报错危险得多。
"""

from __future__ import annotations

import re

from eth_utils import keccak

# _ADDRESS_RE：0x 前缀与 40 位十六进制地址的基础格式规则。
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
# _ZERO_ADDRESS：全零地址，本系统地址校验不接受此值。
_ZERO_ADDRESS = "0x" + "0" * 40


def is_evm_address(value: str) -> bool:
    """格式合法且不是零地址。零地址永远不是有效的转账/授权目标。

    Args:
        value: 待校验、转换或缓存的值。
    """
    return bool(_ADDRESS_RE.match(value)) and value.lower() != _ZERO_ADDRESS


def to_checksum_address(value: str) -> str:
    """按 EIP-55 生成带校验和的写法。

    Args:
        value: 待校验、转换或缓存的值。
    """
    body = value.lower().removeprefix("0x")
    digest = keccak(body.encode("ascii")).hex()
    return "0x" + "".join(c.upper() if int(digest[i], 16) >= 8 else c for i, c in enumerate(body))


def normalize_address(value: str) -> str | None:
    """校验并返回小写地址，不合法返回 None。

    全小写输入视为未提供校验和，直接放行（前端普遍这么传）；
    混合大小写必须与 EIP-55 一致，否则判定为输错字符。

    Args:
        value: 待校验、转换或缓存的值。
    """
    if not is_evm_address(value):
        return None
    if value == value.lower() or value == value.upper():
        return value.lower()
    if to_checksum_address(value) != value:
        return None
    return value.lower()
