"""记忆注入形态：按时间倒序取软记忆，并且单独成块、不混进系统规则。"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import SystemMessage

from web3_agent.agent.agent_loop import _constraints_of, _soft_memories
from web3_agent.agent.graph.read_graph import initial_state
from web3_agent.config.constants import Intent

WALLET = "0x" + "1" * 40


def _row(text: str, kind: str, subject: str = "s") -> SimpleNamespace:
    return SimpleNamespace(text=text, memory_type=kind, subject=subject, confidence=0.9)


def test_soft_memories_are_capped_and_skip_constraints() -> None:
    rows = [
        _row("最近提到的背景", "profile", "recent"),
        _row("不允许转账", "constraint", "rule"),
        _row("稍早的背景", "preference", "earlier"),
        _row("更早的背景", "profile", "oldest"),
    ]

    soft = _soft_memories(rows, cap=2)

    # 输入已是 created_at 倒序，取前两条非 constraint 即可，不需要再排一次。
    assert [item.text for item in soft] == ["最近提到的背景", "稍早的背景"]


def test_constraints_keep_their_own_cap() -> None:
    rows = [_row(f"约束{i}", "constraint", f"r{i}") for i in range(7)]

    assert len(_constraints_of(rows, cap=5)) == 5


def test_memories_go_into_a_labelled_block_not_the_system_rules() -> None:
    state = initial_state(
        [],
        "查一下余额",
        WALLET,
        memories="以下是用户此前提到过的背景。\n- 用户在做跨链套利",
        intent=Intent.QUERY_ASSET,
    )
    systems = [m.content for m in state["messages"] if isinstance(m, SystemMessage)]

    # 规则那条不含记忆原文：背景靠结构分隔，不靠模型从长文本里自己分辨。
    assert "跨链套利" not in systems[0]
    assert any("<user_background>" in s and "跨链套利" in s for s in systems)


def test_no_memory_means_no_extra_message() -> None:
    state = initial_state([], "查一下余额", WALLET, intent=Intent.QUERY_ASSET)
    systems = [m.content for m in state["messages"] if isinstance(m, SystemMessage)]

    assert len(systems) == 1
    assert "<user_background>" not in systems[0]
