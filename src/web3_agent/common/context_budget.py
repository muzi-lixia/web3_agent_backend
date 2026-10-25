"""上下文 token 预算：估算长度并按优先级裁剪注入内容。

Ollama 超限是静默截断——它不会报错，只会把最早的 token 丢掉，被丢的可能是
系统约束或最新指令。这里不引入真实 tokenizer（本地 qwen 没有轻量离线切词器，
引 transformers 太重），改用保守估算：宁可多算、提前裁，也不低估导致超限。

估算规则（偏保守，字符 → token）：
- 中文及全角字符：1 字 ≈ 1 token；
- ASCII 字母数字：4 字符 ≈ 1 token。
"""

from __future__ import annotations

# 英文/数字大致 4 个字符折 1 个 token，中文等宽字符 1 字折 1 个 token。
_ASCII_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """估算一段文本占用的 token 数，偏保守（只会高估，不低估）。

    Args:
        text: 待估算的文本。

    Returns:
        估算 token 数，最少返回 0。
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ch.isascii() and not ch.isspace())
    non_ascii = len(text) - ascii_chars
    ascii_tokens = (ascii_chars + _ASCII_PER_TOKEN - 1) // _ASCII_PER_TOKEN
    return ascii_tokens + non_ascii


def trim_history(
    history: list[tuple[str, str]],
    budget_tokens: int,
) -> list[tuple[str, str]]:
    """按 token 预算从旧到新裁剪历史，保留一条连续的结尾。

    最新一条始终保留：用户下一句多半就在问它，截掉等于把上下文切成两半。

    Args:
        history: 按时间排列的 (角色, 正文) 消息列表。
        budget_tokens: 分配给历史的 token 预算。

    Returns:
        能装进预算的 (角色, 正文) 子集，顺序与传入一致。
    """
    if budget_tokens <= 0:
        return []
    kept: list[tuple[str, str]] = []
    used = 0
    for role, content in reversed(history):
        cost = estimate_tokens(content or "")
        if kept and used + cost > budget_tokens:
            break
        kept.append((role, content))
        used += cost
    kept.reverse()
    return kept
