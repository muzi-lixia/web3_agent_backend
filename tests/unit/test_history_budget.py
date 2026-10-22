"""历史窗口的字符预算。

只按条数截管不住体积：一条 50 笔交易的直出文本就有 7000 字符，
两三轮下来会把 num_ctx 挤满，而 Ollama 超限是静默截断。
"""

from __future__ import annotations

from web3_agent.persistence.pg.repositories.conversation_repo import _fit_budget


def test_over_budget_drops_the_oldest_and_keeps_a_contiguous_tail() -> None:
    messages = [("user", "a" * 100), ("assistant", "b" * 100), ("user", "c" * 100)]

    kept = _fit_budget(messages, max_chars=250)

    assert kept == [("assistant", "b" * 100), ("user", "c" * 100)]


def test_newest_message_survives_even_when_it_alone_exceeds_the_budget() -> None:
    """最新一条永远保留：用户下一句多半就在问它。"""
    messages = [("user", "a" * 50), ("assistant", "b" * 900)]

    kept = _fit_budget(messages, max_chars=100)

    assert kept == [("assistant", "b" * 900)]


def test_zero_budget_means_no_trimming() -> None:
    messages = [("user", "a" * 5000), ("assistant", "b" * 5000)]

    assert _fit_budget(messages, max_chars=0) == messages
