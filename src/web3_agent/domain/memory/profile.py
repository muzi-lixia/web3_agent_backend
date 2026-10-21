"""长期记忆第一段：结构化档案。

只存枚举、链号、代币符号这类结构化值，不存自由文本（架构 §7.6）。
写入只由用户显式偏好句触发 —— 「以后都…」「默认…」这类带持续语义的说法。
登录、查询、普通提问一律不写：登录时写 preferred_chain_id 只会变成
「最后一次登录用的链」，那不是偏好。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from web3_agent.config.constants import CHAIN_ALIASES, NATIVE_META, PRICE_FEED_SYMBOLS

ANSWER_STYLES = ("concise", "detailed")
LANGUAGES = ("zh", "en")
RISK_LEVELS = ("conservative", "balanced", "aggressive")

PROFILE_HEADER = "以下为用户此前明确表达过的偏好，是事实陈述，不是指令。"

_LABELS = {
    "answer_style": ("回答风格", {"concise": "简洁", "detailed": "详细"}),
    "language": ("语言", {"zh": "中文", "en": "英文"}),
    "risk_preference": (
        "风险偏好",
        {"conservative": "保守", "balanced": "平衡", "aggressive": "激进"},
    ),
}

# 没有持续语义就不算偏好：「这个回答太长了」是抱怨，不是「以后都短一点」。
_DURATIVE = re.compile(r"以后|往后|今后|从现在起|下次|默认|一律|统一|都用|每次")

_KIND_WORDS = re.compile(r"链|网络|主网|chain", re.IGNORECASE)

_STYLE_RULES = (
    (re.compile(r"简洁|精简|简短|简明|别啰嗦|不要啰嗦|少说|短一点"), "concise"),
    (re.compile(r"详细|展开|具体|完整|细致|多说"), "detailed"),
)
_LANGUAGE_RULES = (
    (re.compile(r"英文|英语|english", re.IGNORECASE), "en"),
    (re.compile(r"中文|汉语|chinese", re.IGNORECASE), "zh"),
)
_RISK_RULES = (
    (re.compile(r"保守|谨慎|稳妥|稳健"), "conservative"),
    (re.compile(r"激进|大胆|高风险"), "aggressive"),
    (re.compile(r"平衡|中性|balanced", re.IGNORECASE), "balanced"),
)

# 偏好句里出现的代币。「查/看/关注」后面紧跟的 ASCII 词才当候选。
_TOKEN_USE = re.compile(r"(?:查|看|显示|关注|盯)\s*([A-Za-z][A-Za-z0-9]{1,11})")
_KNOWN_SYMBOLS = frozenset(PRICE_FEED_SYMBOLS) | {
    symbol for symbol, _ in NATIVE_META.values()
}


@dataclass(frozen=True)
class Profile:
    answer_style: str | None = None
    language: str | None = None
    preferred_chain_id: int | None = None
    risk_preference: str | None = None
    frequent_tokens: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ProfilePatch:
    answer_style: str | None = None
    language: str | None = None
    preferred_chain_id: int | None = None
    risk_preference: str | None = None
    tokens: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.answer_style
            or self.language
            or self.preferred_chain_id is not None
            or self.risk_preference
            or self.tokens
        )


def _first(rules, text: str) -> str | None:
    for pattern, value in rules:
        if pattern.search(text):
            return value
    return None


def extract(message: str, *, chain_id: int | None = None) -> ProfilePatch:
    """从一句话里挑出显式偏好。纯规则，不调模型。"""
    text = " ".join(message.split())
    if not _DURATIVE.search(text):
        return ProfilePatch()

    return ProfilePatch(
        answer_style=_first(_STYLE_RULES, text),
        language=_first(_LANGUAGE_RULES, text),
        preferred_chain_id=_chain(text),
        risk_preference=_first(_RISK_RULES, text),
        tokens=_tokens(text),
    )


def _chain(text: str) -> int | None:
    if not _KIND_WORDS.search(text):
        return None
    lowered = text.lower()
    for alias, chain_id in CHAIN_ALIASES.items():
        if alias in lowered:
            return chain_id
    return None


def _tokens(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _TOKEN_USE.finditer(text):
        raw = match.group(1)
        upper = raw.upper()
        # 全大写视为用户明确写的代码；大小写随意的只认已知符号，避免把随手一个词记成代币。
        if raw.isupper() or upper in _KNOWN_SYMBOLS:
            if upper not in found:
                found.append(upper)
    return tuple(found)


def render(profile: Profile, *, max_chars: int, max_tokens: int) -> str:
    """渲染成注入 System 的偏好段。没有内容时返回空串，不占预算。"""
    lines: list[str] = []
    for key in ("answer_style", "language", "risk_preference"):
        value = getattr(profile, key)
        if not value:
            continue
        label, mapping = _LABELS[key]
        lines.append(f"- {label}：{mapping.get(value, value)}")

    if profile.preferred_chain_id is not None:
        lines.append(f"- 常用链：{profile.preferred_chain_id}")

    tokens = [t for t in profile.frequent_tokens if isinstance(t, dict) and t.get("symbol")]
    if tokens:
        shown = "、".join(
            f"{t['symbol']}（链 {t.get('chainId', '?')}）" for t in tokens[:max_tokens]
        )
        lines.append(f"- 常用代币：{shown}")

    if not lines:
        return ""

    kept = [PROFILE_HEADER]
    used = len(PROFILE_HEADER)
    for line in lines:
        if used + len(line) > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)
