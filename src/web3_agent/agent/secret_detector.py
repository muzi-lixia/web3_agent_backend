"""私钥 / 助记词模式检测。

接入 LangSmith 之后，用户随手粘进对话的私钥会被上报到第三方，所以这一层
必须在调用模型之前拦截：命中就拒答、写审计、并让该轮 hide_inputs 生效。

判定按真实凭证形态来，不是"像 hex 就拦"：
- 裸 64 位 hex 无前缀 = 私钥（普通哈希很少这样贴，交易哈希一律带 0x）；
- 带 0x 的 64 位 hex 默认当交易哈希放过，只有同文出现「私钥/private key」才拦；
- 助记词要求整行就是 12/15/18/21/24 个纯小写单词，普通英文句子带标点或大写，不命中。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 无前缀的 64 位 hex。前面不能是 hex 字符或 x，否则会把 0x 后面的部分切成第二段。
# _BARE_KEY：疑似无 0x 前缀私钥的 64 位十六进制模式。
_BARE_KEY = re.compile(r"(?<![0-9a-fA-Fx])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
# 带 0x 的 64 位 hex：交易哈希、区块哈希都长这样，单独出现不能判定为密钥。
# _PREFIXED_KEY：带 0x 的 64 位十六进制模式，需结合私钥语义判定。
_PREFIXED_KEY = re.compile(r"\b0x[0-9a-fA-F]{64}\b")
# _KEY_HINT：用于区别私钥与普通链上哈希的敏感语义提示。
_KEY_HINT = re.compile(
    r"(?i)\bprivate[\s_-]*key\b|\bsecret[\s_-]*key\b|\bpriv[\s_-]*key\b|私钥|秘钥|助记词"
)
# _WORD：疑似助记词单个单词的字符和长度规则。
_WORD = re.compile(r"[a-z]{3,8}")
# BIP39 只生成这四种长度（12/15/18/21/24）；词表里全是 3–8 位小写单词。
# _MNEMONIC_COUNTS：疑似助记词检测接受的标准词数集合。
_MNEMONIC_COUNTS = frozenset({12, 15, 18, 21, 24})


# 数据职责：检测到疑似密钥时返回的风险类型和脱敏片段。
@dataclass(frozen=True)
class SecretHit:
    # kind：命中的敏感信息类型：private_key 私钥或 mnemonic 疑似助记词。
    kind: str
    # excerpt：敏感内容的脱敏摘要，仅供审计识别风险类别。
    excerpt: str


def detect(text: str) -> SecretHit | None:
    """检查输入是否疑似私钥或助记词；命中时返回脱敏的风险记录。

    Args:
        text: 待检查、解析或处理的原始文本。
    """
    match = _BARE_KEY.search(text)
    if match is not None:
        value = match.group(0)
        return SecretHit(kind="private_key", excerpt=f"{value[:6]}…{value[-4:]}")

    if _KEY_HINT.search(text) is not None:
        match = _PREFIXED_KEY.search(text)
        if match is not None:
            value = match.group(0)
            return SecretHit(kind="private_key", excerpt=f"{value[:6]}…{value[-4:]}")

    if _looks_like_mnemonic(text):
        return SecretHit(kind="mnemonic", excerpt="[12+ words]")
    return None


def _looks_like_mnemonic(text: str) -> bool:
    """按行判：某一行整行恰好是 BIP39 允许的词数，且每个词都是纯小写字母。

    只要求"出现 12 个连续小写短词"是不够的 —— 普通英文长句能满足，
    而且句子里带标点、首字母大写，真实助记词不会。

    Args:
        text: 待检查、解析或处理的原始文本。
    """
    for line in text.splitlines():
        words = line.split()
        if len(words) not in _MNEMONIC_COUNTS:
            continue
        if all(_WORD.fullmatch(word) for word in words):
            return True
    return False
