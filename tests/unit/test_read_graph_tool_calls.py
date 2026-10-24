"""只读图：工具轨迹必须累计，不能只留最后一批。

证据校验依赖这里 —— 只留最后一批时，先成功再失败的一轮会被误判成「无据作答」。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import CallSource

WALLET = "0x63a9f8df422e1983d42565234557d46d18dbece9"


class _Model:
    """按预定顺序吐出三轮回复：先调 A，再调 B，最后收尾。"""

    def __init__(self) -> None:
        self._rounds = [
            AIMessage(content="", tool_calls=[{"name": "tool_a", "args": {}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "tool_b", "args": {}, "id": "c2"}]),
            AIMessage(content="查完了"),
        ]

    async def ainvoke(self, messages):
        return self._rounds.pop(0)


class _Runner:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def run(self, name: str, args: dict, *, source: CallSource):
        self.seen.append(name)
        ok = name == "tool_a"
        return f'{{"ok": {str(ok).lower()}}}', ToolCallRecord(
            name=name, ok=ok, error=None if ok else "E303"
        )


def test_records_from_every_round_are_kept() -> None:
    runner = _Runner()
    graph = build_read_graph(_Model(), runner=runner, max_steps=5)

    state = asyncio.run(graph.ainvoke(initial_state([], "查一下", WALLET)))

    assert state["messages"][-1].content == "查完了"
    assert [r["name"] for r in state["tool_calls"]] == ["tool_a", "tool_b"]
    assert [r["ok"] for r in state["tool_calls"]] == [True, False]


def test_every_call_in_one_message_runs() -> None:
    """一条模型消息带多个调用时逐个执行；次数约束由意图白名单与步数上限负责。"""
    calls = [
        {"name": name, "args": {}, "id": f"c{index}"}
        for index, name in enumerate(("tool_a", "tool_b", "tool_c"))
    ]

    class _Burst:
        def __init__(self) -> None:
            self._rounds = [AIMessage(content="", tool_calls=calls), AIMessage(content="查完了")]

        async def ainvoke(self, messages):
            return self._rounds.pop(0)

    runner = _Runner()
    graph = build_read_graph(_Burst(), runner=runner, max_steps=5)
    state = asyncio.run(graph.ainvoke(initial_state([], "查一下", WALLET)))

    assert runner.seen == ["tool_a", "tool_b", "tool_c"]
    assert [(r["name"], r["ok"]) for r in state["tool_calls"]] == [
        ("tool_a", True),
        ("tool_b", False),
        ("tool_c", False),
    ]
    assert state["messages"][-1].content == "查完了"
