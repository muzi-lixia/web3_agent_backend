"""SanitizePolicy：只清洗自由文本。

地址、金额、枚举等结构化字段不走这里——格式不对由 Validators 直接拒绝，
不做截断或"修一下"，否则脏值会变成合法值。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior)\s+(instruction|prompt)",
    r"disregard\s+(all\s+)?(previous|prior)",
    r"system\s*prompt",
    r"developer\s*mode",
    r"忽略(之前|以上|前面|先前)的?(所有)?(指令|提示|要求)",
    r"你现在是",
    r"进入.{0,4}模式",
)
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

MAX_FREE_TEXT = 32


@dataclass(frozen=True)
class SanitizeResult:
    text: str
    hijacked: bool


def sanitize_free_text(text: str, max_len: int = MAX_FREE_TEXT) -> SanitizeResult:
    cleaned = _ZERO_WIDTH.sub("", text)
    cleaned = _CONTROL.sub("", cleaned)
    hijacked = bool(_INJECTION.search(cleaned))
    if hijacked:
        cleaned = _INJECTION.sub("[filtered]", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "…"
    return SanitizeResult(text=cleaned, hijacked=hijacked)


def wrap_chain_data(text: str) -> str:
    """链上返回的自由文本进 prompt 前必须包裹，避免被当成指令。"""
    safe = text.replace("</chain_data>", "<\\/chain_data>")
    return f"<chain_data>{safe}</chain_data>"
