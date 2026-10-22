"""编排层：无据作答必须被拦掉。

图与 DB 依赖全部换成桩，只保留真实的 JsonGuard、IntentRouter 和证据校验，
验证的是「意图需要数据、却没有任何成功工具调用」时那一段接线有没有生效。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage

from web3_agent.agent.agent_loop import AgentLoop
from web3_agent.agent.json_guard import JsonGuard
from web3_agent.common.types import LlmOutput, ToolCallRecord
from web3_agent.config.constants import MSG_NO_DATA, Intent
from web3_agent.config.settings import Settings

FABRICATED = "您的交易记录如下：\n1. 交易哈希: 0x1234567890abcdef，金额 10.00 BERA"


class _Llm:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent

    async def extract_intent(self, history, user_message, **kwargs) -> LlmOutput:
        return LlmOutput(intent=self.intent, confidence=0.95, slots={})


class _Audit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def log(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))


class _Conversation:
    async def load(self, db, *, siwe_user_id: str) -> list[tuple[str, str]]:
        return []

    async def save(self, db, **kwargs: Any) -> None:
        pass


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0]


class _Graph:
    """按给定内容返回一次回答，不执行任何工具。"""

    def __init__(self, answer: str, records: list[ToolCallRecord] | None = None) -> None:
        self.answer = answer
        self.records = records or []

    async def ainvoke(self, state, config=None):
        return {
            "messages": [AIMessage(content=self.answer)],
            "steps": 1,
            "tool_calls": [r.model_dump() for r in self.records],
        }


def _loop(graph: _Graph, intent: Intent) -> AgentLoop:
    loop = AgentLoop(
        Settings(),
        llm=_Llm(intent),
        guard=JsonGuard(),
        conversation=_Conversation(),
        audit=_Audit(),
        model_with_tools=None,
        runner=None,
        embedder=_Embedder(),
    )
    loop._graph = graph  # 只替换图，其余编排逻辑保持真实
    return loop


async def _empty(*args: Any, **kwargs: Any) -> str:
    return ""


def _without_db(loop: AgentLoop) -> AgentLoop:
    loop._load_summary = _empty
    loop._sync_profile = _empty
    loop._load_memories = _empty
    return loop


def test_fabricated_answer_without_any_tool_call_is_replaced() -> None:
    loop = _without_db(_loop(_Graph(FABRICATED), Intent.QUERY_TX))

    outcome = asyncio.run(
        loop.handle(
            None,
            siwe_user_id="u-1",
            wallet_address="0x63a9f8df422e1983d42565234557d46d18dbece9",
            chain_id=80094,
            user_message="查一下我的交易记录",
        )
    )

    assert outcome.content == MSG_NO_DATA
    assert outcome.degraded is True
    assert "answer_without_evidence" in [event for event, _ in loop._audit.events]


def test_answer_is_kept_when_a_tool_call_succeeded() -> None:
    records = [ToolCallRecord(name="get_transaction_records", ok=True)]
    loop = _without_db(_loop(_Graph(FABRICATED, records), Intent.QUERY_TX))

    outcome = asyncio.run(
        loop.handle(
            None,
            siwe_user_id="u-1",
            wallet_address="0x63a9f8df422e1983d42565234557d46d18dbece9",
            chain_id=80094,
            user_message="查一下我的交易记录",
        )
    )

    assert outcome.content == FABRICATED
    assert outcome.degraded is False
    assert "answer_without_evidence" not in [event for event, _ in loop._audit.events]


def test_all_tool_calls_failed_is_treated_as_no_evidence() -> None:
    records = [ToolCallRecord(name="get_transaction_records", ok=False, error="E303")]
    loop = _without_db(_loop(_Graph(FABRICATED, records), Intent.QUERY_TX))

    outcome = asyncio.run(
        loop.handle(
            None,
            siwe_user_id="u-1",
            wallet_address="0x63a9f8df422e1983d42565234557d46d18dbece9",
            chain_id=80094,
            user_message="查一下我的交易记录",
        )
    )

    assert outcome.content == MSG_NO_DATA
    assert outcome.degraded is True


def test_chitchat_without_tools_is_not_touched() -> None:
    loop = _without_db(_loop(_Graph("你好，我能帮你查链上资产与交易。"), Intent.CHITCHAT))

    outcome = asyncio.run(
        loop.handle(
            None,
            siwe_user_id="u-1",
            wallet_address="0x63a9f8df422e1983d42565234557d46d18dbece9",
            chain_id=80094,
            user_message="你好",
        )
    )

    assert outcome.content.startswith("你好")
    assert outcome.degraded is False


class _RetryGraph:
    """第一次不调工具，第二次调 —— 验证重跑那一步确实发生了。"""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, state, config=None):
        self.calls += 1
        if self.calls == 1:
            return {"messages": [AIMessage(content=FABRICATED)], "steps": 1, "tool_calls": []}
        record = ToolCallRecord(name="get_transaction_records", ok=True)
        return {
            "messages": [AIMessage(content="查到了两笔交易")],
            "steps": 2,
            "tool_calls": [record.model_dump()],
        }


def test_missing_tool_call_triggers_one_retry() -> None:
    graph = _RetryGraph()
    loop = _without_db(_loop(graph, Intent.QUERY_TX))

    outcome = asyncio.run(
        loop.handle(
            None,
            siwe_user_id="u-1",
            wallet_address="0x63a9f8df422e1983d42565234557d46d18dbece9",
            chain_id=80094,
            user_message="查一下我的交易记录",
        )
    )

    assert graph.calls == 2
    assert outcome.content == "查到了两笔交易"
    assert outcome.degraded is False
    assert "tool_call_retried" in [event for event, _ in loop._audit.events]
