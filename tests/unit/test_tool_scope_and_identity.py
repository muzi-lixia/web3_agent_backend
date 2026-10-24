"""执行侧的两道闸：本轮只认意图对应的工具，身份地址不接受模型传入的值。

这两条都不能靠提示词或某条正常路由兑现 —— 模型看到的工具列表是全局的，
它随时可以点名别的工具、并自己填一个地址。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from web3_agent.agent.graph.read_graph import (
    allowed_tools,
    build_read_graph,
    evidence_tools,
    initial_state,
)
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import CallSource, Intent

WALLET = "0x63a9f8df422e1983d42565234557d46d18dbece9"
OTHER = "0x1111111111111111111111111111111111111111"


class _Model:
    """先吐一次工具调用，再给一句收尾话。"""

    def __init__(self, name: str, args: dict) -> None:
        self._rounds = [
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}]),
            AIMessage(content="查完了"),
        ]

    async def ainvoke(self, messages):
        return self._rounds.pop(0) if len(self._rounds) > 1 else self._rounds[-1]


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def run(self, name: str, args: dict, *, source: CallSource):
        self.calls.append((name, args))
        return '{"status": "empty"}', ToolCallRecord(name=name, ok=True)


def _run(model: _Model, runner: _Runner, **state):
    graph = build_read_graph(model, runner=runner, max_steps=5)
    return asyncio.run(graph.ainvoke(initial_state([], "查一下", **state)))


def test_allowed_tools_follow_the_intent() -> None:
    assert evidence_tools(Intent.QUERY_ASSET) == frozenset({"get_asset_balance"})
    assert evidence_tools(Intent.QUERY_TX) == frozenset({"get_transaction_records"})
    assert evidence_tools(Intent.CHITCHAT) == frozenset()
    # 知识检索与数据工具平行：查资产的一轮里也允许调它。
    assert allowed_tools(Intent.QUERY_ASSET) == frozenset(
        {"get_asset_balance", "rag_general_knowledge"}
    )
    assert allowed_tools(Intent.QUERY_KNOWLEDGE) == frozenset({"rag_general_knowledge"})
    # 没有意图的图不是从对话入口进来的，不做范围限制。
    assert allowed_tools(None) is None


def test_knowledge_retrieval_runs_alongside_the_data_tool() -> None:
    """一句话里既要数据又要解释：检索是与数据工具平行的来源，不按越界拒掉。"""
    runner = _Runner()
    state = _run(
        _Model("rag_general_knowledge", {"query_str": "什么是质押"}),
        runner,
        wallet_address=WALLET,
        intent=Intent.QUERY_ASSET,
    )

    assert [name for name, _ in runner.calls] == ["rag_general_knowledge"]
    assert state["tool_calls"][0]["ok"] is True


def test_tool_outside_this_intent_is_refused_before_it_runs() -> None:
    """复现里的那条路径：查资产的一轮里点名交易记录，地址还是别人的。"""
    runner = _Runner()
    state = _run(
        _Model("get_transaction_records", {"address": OTHER}),
        runner,
        wallet_address=WALLET,
        intent=Intent.QUERY_ASSET,
    )

    assert runner.calls == []
    assert [r["ok"] for r in state["tool_calls"]] == [False]


def test_transaction_address_always_comes_from_the_session() -> None:
    runner = _Runner()
    state = _run(
        _Model("get_transaction_records", {"address": OTHER}),
        runner,
        wallet_address=WALLET,
    )

    assert runner.calls == [("get_transaction_records", {"address": WALLET})]
    assert state["tool_calls"][0]["ok"] is True


def test_identity_bound_tool_never_runs_without_a_session_wallet() -> None:
    runner = _Runner()
    state = _run(
        _Model("get_transaction_records", {"address": OTHER}),
        runner,
        wallet_address="",
    )

    assert runner.calls == []
    assert state["tool_calls"][0]["error"] == "E503"


def test_balance_still_accepts_an_explicit_third_party_address() -> None:
    """余额是公开链上数据，说了别人才查别人；交易记录才做身份绑定。"""
    runner = _Runner()
    _run(
        _Model("get_asset_balance", {"address": OTHER}),
        runner,
        wallet_address=WALLET,
        intent=Intent.QUERY_ASSET,
    )

    assert runner.calls == [("get_asset_balance", {"address": OTHER})]
