"""私钥 / 助记词模式检测。

接入 LangSmith 之后，用户随手粘进对话的私钥会被上报到第三方，所以这一层
必须在调用模型之前拦截：命中就拒答、写审计、并让该轮 hide_inputs 生效。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PRIVATE_KEY = re.compile(r"(?<![0-9a-fA-F])(0x)?[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_MNEMONIC = re.compile(r"\b(?:[a-z]{3,12}\s+){11,23}[a-z]{3,12}\b")


@dataclass(frozen=True)
class SecretHit:
    kind: str
    excerpt: str


def detect(text: str) -> SecretHit | None:
    match = _PRIVATE_KEY.search(text)
    if match:
        value = match.group(0)
        return SecretHit(kind="private_key", excerpt=f"{value[:6]}…{value[-4:]}")

    match = _MNEMONIC.search(text)
    if match and len(match.group(0).split()) >= 12:
        return SecretHit(kind="mnemonic", excerpt="[12+ words]")

    return None
