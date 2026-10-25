"""上下文 token 估算与历史裁剪的保守性验证。"""

from __future__ import annotations

from web3_agent.common.context_budget import estimate_tokens, trim_history
from web3_agent.agent.graph.read_graph import initial_state
from web3_agent.config.constants import Intent


def test_estimate_empty_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_chinese_one_char_one_token() -> None:
    # 中文按 1 字 1 token 估算，保守不低估。
    assert estimate_tokens("你好世界") == 4


def test_estimate_ascii_four_chars_one_token() -> None:
    # 英文/数字按 4 字符 1 token，向上取整。
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a") == 1


def test_estimate_mixed() -> None:
    # 混合文本：中文按字、ASCII 按 4 字 1 token 分别累计。
    # 转(1) + 5(1) + BERA(1，4 个 ASCII 字符) = 3
    assert estimate_tokens("转5BERA") == 3


def test_trim_history_keeps_latest_when_over_budget() -> None:
    history = [
        ("user", "第一句很长很长很长很长很长很长很长很长"),
        ("assistant", "回答"),
        ("user", "最新这句必须保留"),
    ]
    # 预算只够放最后一条（最新这句 8 字 = 8 token）。
    kept = trim_history(history, budget_tokens=8)
    assert kept == [("user", "最新这句必须保留")]


def test_trim_history_keeps_all_when_under_budget() -> None:
    history = [("user", "你好"), ("assistant", "好")]
    kept = trim_history(history, budget_tokens=100)
    assert kept == history


def test_trim_history_zero_budget_returns_empty() -> None:
    history = [("user", "你好")]
    assert trim_history(history, budget_tokens=0) == []


def test_trim_history_negative_budget_returns_empty() -> None:
    history = [("user", "你好")]
    assert trim_history(history, budget_tokens=-1) == []


def test_trim_history_preserves_role_and_order() -> None:
    history = [
        ("user", "aa"),
        ("assistant", "bb"),
        ("user", "cc"),
        ("assistant", "dd"),
    ]
    # 每条 1 token，预算 3 条 → 保留末尾 3 条。
    kept = trim_history(history, budget_tokens=3)
    assert kept == [("assistant", "bb"), ("user", "cc"), ("assistant", "dd")]


def test_initial_state_trims_history_when_budget_tight() -> None:
    history = [
        ("user", "第一句历史很长很长很长很长"),
        ("assistant", "回答内容"),
        ("user", "最新的这句话"),
    ]
    # 极小预算：只够 system，历史一条都不留，用户输入仍保留。
    state = initial_state(
        history, "查余额", "0x1234", intent=Intent.QUERY_ASSET, context_budget_tokens=120
    )
    contents = [m.content for m in state["messages"]]
    assert "最新的这句话" not in contents
    assert contents[-1] == "查余额"
    assert isinstance(state["messages"][0].content, str)


def test_initial_state_keeps_history_when_budget_ample() -> None:
    history = [("user", f"第{i}条历史消息内容" + "x" * 50) for i in range(20)]
    state = initial_state(
        history, "查余额", "0x1234", intent=Intent.QUERY_ASSET, context_budget_tokens=6553
    )
    kept = [m for m in state["messages"] if "历史消息" in str(m.content)]
    assert len(kept) == 20
