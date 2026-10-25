"""多步计划图的执行顺序与挂起语义。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from web3_agent.agent import transfer_task
from web3_agent.agent.agent_loop import AgentLoop
from web3_agent.agent.graph import plan_graph
from web3_agent.agent.multi_hop import amount_is_quoted, needs_plan
from web3_agent.common.types import Hop, ToolCallRecord
from web3_agent.config.constants import MSG_WRITE_TASK_FAILED, MSG_WRITE_NOT_ENABLED
from web3_agent.config.settings import Settings
from web3_agent.domain.workflow import service as workflow_service

_WALLET = "0x2222222222222222222222222222222222222222"
_TARGET = "0x1111111111111111111111111111111111111111"
_MESSAGE = (
    f"查一下我有多少bera，如果超过10BERA，就转给{_TARGET}这个地址5BERA"
)

_BALANCE = {
    "balances": [
        {"chainId": 80094, "symbol": "BERA", "decimals": 18, "raw": "12400000000000000000", "human": "12.4"}
    ]
}

_READY = {
    "status": "ready",
    "asset": {"symbol": "BERA", "human": "5"},
    "recipient": _TARGET,
    "from": _WALLET,
    "tx": {"to": _TARGET, "value": "5000000000000000000", "data": "0x"},
    "chainId": 80094,
}


def _balance(raw: str, human: str, symbol: str = "BERA", chain: int = 80094) -> dict:
    return {
        "balances": [
            {"chainId": chain, "symbol": symbol, "decimals": 18, "raw": raw, "human": human}
        ]
    }


class _Runner:
    """按顺序回放工具结果，并记录每次调用的参数。"""

    def __init__(self, payloads: list[dict], ok: bool = True) -> None:
        self._payloads = list(payloads)
        self._ok = ok
        self.calls: list[tuple[str, dict]] = []

    async def run(self, name: str, args: dict, *, source):
        self.calls.append((name, args))
        payload = self._payloads.pop(0) if self._payloads else {}
        return (
            json.dumps(payload, ensure_ascii=False),
            ToolCallRecord(name=name, args=args, ok=self._ok, error=None if self._ok else "E303"),
        )


@pytest.fixture
def task_store(monkeypatch):
    """替掉建任务，避免这些用例依赖数据库。"""
    store: dict = {"created": []}

    async def _create(**kwargs):
        store["created"].append(kwargs)
        return "task-1"

    monkeypatch.setattr(workflow_service, "create_transfer", _create)
    return store


def _hops(
    *,
    amount: str = "5",
    op: str = "gte",
    right: str = "10",
    symbol: str = "BERA",
    tool: str = "post_transfer",
) -> list[Hop]:
    return [
        Hop(kind="read", tool="get_asset_balance", args={"symbol": "BERA"}),
        Hop(kind="cond", symbol=symbol, op=op, right=right),
        Hop(
            kind="write",
            tool=tool,
            args={"toAddress": _TARGET, "tokenSymbol": "BERA", "humanAmount": amount},
        ),
    ]


def _initial(hops: list[Hop], thread_id: str = "t1", message: str = _MESSAGE):
    return plan_graph.initial(
        hops=hops,
        user_message=message,
        siwe_user_id="u1",
        wallet_address=_WALLET,
        chain_id=80094,
        thread_id=thread_id,
        ttl_seconds=900,
    )


def _run(runner, hops: list[Hop], thread_id: str = "t1", message: str = _MESSAGE):
    graph = plan_graph.build_plan_graph(runner, InMemorySaver())

    async def _call():
        return graph, await graph.ainvoke(
            _initial(hops, thread_id, message),
            config={"configurable": {"thread_id": thread_id}},
        )

    return asyncio.run(_call())


def test_needs_plan_only_when_write_follows_read_or_condition():
    assert needs_plan(_hops()) is True
    assert needs_plan([Hop(kind="read", tool="get_asset_balance")]) is False
    assert needs_plan(
        [Hop(kind="write", tool="post_transfer", args={"humanAmount": "5"})]
    ) is False


def test_condition_met_suspends_with_task_and_evidence(task_store):
    runner = _Runner([_BALANCE, _READY])
    graph, state = _run(runner, _hops())

    assert "__interrupt__" in state
    assert state["task_id"] == "task-1"
    assert state["verdicts"] == [
        {"symbol": "BERA", "actual": "12.4", "op": "gte", "right": "10"}
    ]
    # 检查点标识随任务落库，确认接口才有东西可以恢复。
    assert task_store["created"][0]["plan_thread_id"] == "t1"
    # 地址参数用会话身份覆盖，模型给的其他地址不采纳。
    assert runner.calls[0][1]["address"] == _WALLET


def test_resume_finishes_the_plan(task_store):
    runner = _Runner([_BALANCE, _READY])
    graph, _ = _run(runner, _hops())

    asyncio.run(
        graph.ainvoke(
            Command(resume=True), config={"configurable": {"thread_id": "t1"}}
        )
    )

    snapshot = asyncio.run(graph.aget_state({"configurable": {"thread_id": "t1"}}))
    assert snapshot.next == ()


def test_condition_not_met_never_creates_a_task(task_store):
    _, state = _run(_Runner([_balance("8200000000000000000", "8.2")]), _hops())

    assert task_store["created"] == []
    assert state["kind"] == "answer"
    assert "8.2" in state["content"]


def test_gte_boundary_is_inclusive_and_gt_is_not(task_store):
    exact = _balance("10000000000000000000", "10")
    _, inclusive = _run(_Runner([exact, _READY]), _hops(), thread_id="a")
    assert inclusive["task_id"] == "task-1"

    task_store["created"].clear()
    _, exclusive = _run(_Runner([exact, _READY]), _hops(op="gt"), thread_id="b")
    assert task_store["created"] == []
    assert exclusive["kind"] == "answer"


def test_unknown_amount_asks_instead_of_guessing(task_store):
    _, state = _run(_Runner([_balance("0", "0")]), _hops(symbol="USDC"))

    assert task_store["created"] == []
    assert state["kind"] == "ask"
    assert "USDC" in state["content"]


def test_amount_taken_from_threshold_is_refused(task_store):
    # 原话里 10 是条件阈值；模型把它当成转账金额时不能放行。
    _, state = _run(_Runner([_BALANCE, _READY]), _hops(amount="10"))

    assert task_store["created"] == []
    assert state["kind"] == "ask"


def test_spoken_amount_is_accepted_via_original_fragment(task_store):
    # 「五个」抽不出数字，靠模型给的原话片段确认金额确实出自本人之口。
    message = f"余额够 10 个 BERA 就转五个给 {_TARGET}"
    hops = [
        Hop(kind="read", tool="get_asset_balance"),
        Hop(kind="cond", symbol="BERA", op="gte", right="10"),
        Hop(
            kind="write",
            tool="post_transfer",
            args={
                "toAddress": _TARGET,
                "tokenSymbol": "BERA",
                "humanAmount": "5",
                "humanAmountText": "五个",
            },
        ),
    ]
    _, state = _run(_Runner([_BALANCE, _READY]), hops, message=message)

    assert state["task_id"] == "task-1"


def test_insufficient_balance_stops_before_creating_task(task_store):
    # 条件成立（0 小于 100）但余额不够转 5：不该出一张签了必然回滚的卡片。
    _, state = _run(
        _Runner([_balance("0", "0")]),
        _hops(op="lt", right="100"),
        message=f"余额不到 100 BERA 就转 5 BERA 给 {_TARGET}",
    )

    assert task_store["created"] == []
    assert state["kind"] == "answer"
    assert "余额不足" in state["content"]


def test_afford_skips_when_the_token_was_not_queried(task_store):
    # 查的是 USDC，转的是 BERA：没有可比余额，交给构造工具，不在这里拦。
    usdc = _balance("0", "0", symbol="USDC")
    hops = [
        Hop(kind="read", tool="get_asset_balance", args={"symbol": "USDC"}),
        Hop(
            kind="write",
            tool="post_transfer",
            args={"toAddress": _TARGET, "tokenSymbol": "BERA", "humanAmount": "5"},
        ),
    ]
    _, state = _run(
        _Runner([usdc, _READY]),
        hops,
        message=f"先看下我有多少 USDC，然后转 5 BERA 给 {_TARGET}",
    )

    assert state["task_id"] == "task-1"


def test_two_writes_are_refused(task_store):
    hops = [
        Hop(kind="read", tool="get_asset_balance"),
        Hop(kind="write", tool="post_transfer", args={"humanAmount": "5"}),
        Hop(kind="write", tool="post_transfer", args={"humanAmount": "5"}),
    ]
    _, state = _run(_Runner([_BALANCE]), hops)

    assert task_store["created"] == []
    assert state["kind"] == "ask"


def test_write_before_read_is_refused(task_store):
    hops = [
        Hop(kind="write", tool="post_transfer", args={"humanAmount": "5"}),
        Hop(kind="read", tool="get_asset_balance"),
    ]
    _, state = _run(_Runner([_BALANCE]), hops)

    assert task_store["created"] == []
    assert state["kind"] == "ask"


def test_two_conditions_are_refused(task_store):
    hops = [
        Hop(kind="read", tool="get_asset_balance"),
        Hop(kind="cond", symbol="BERA", op="gte", right="10"),
        Hop(kind="cond", symbol="BERA", op="gte", right="10"),
        Hop(kind="write", tool="post_transfer", args={"humanAmount": "5"}),
    ]
    _, state = _run(_Runner([_BALANCE]), hops)

    assert task_store["created"] == []
    assert state["kind"] == "ask"


def test_threshold_carried_over_from_last_turn_is_questioned(task_store):
    # 上一轮说过 10，这一轮只说「转 5 BERA」：不能拿旧门槛去执行这一轮的钱。
    _, state = _run(
        _Runner([_BALANCE, _READY]),
        _hops(),
        message=f"转 5 BERA 给 {_TARGET}",
    )

    assert task_store["created"] == []
    assert state["kind"] == "ask"
    assert "10" in state["content"]


def test_amount_duplicated_as_threshold_falls_back_to_single_step(task_store):
    # 「转 5 BERA」原话只有一个数字，模型把它同时写成门槛和金额：按单步执行。
    _, state = _run(
        _Runner([_BALANCE, _READY]),
        _hops(amount="5", right="5"),
        message=f"转 5 BERA 给 {_TARGET}",
    )

    assert state["task_id"] == "task-1"
    assert state["verdicts"] == []


def test_same_value_twice_in_words_keeps_the_condition(task_store):
    # 「超过 5 就转 5」原话里 5 出现两次，门槛与金额各占一个，条件成立。
    _, state = _run(
        _Runner([_BALANCE, _READY]),
        _hops(amount="5", right="5"),
        message=f"超过 5 BERA 就转 5 BERA 给 {_TARGET}",
    )

    assert len(state["verdicts"]) == 1


def test_read_failure_stops_before_judging(task_store):
    _, state = _run(_Runner([{}], ok=False), _hops())

    assert task_store["created"] == []
    assert state["kind"] == "answer"


def test_unexpected_write_tool_reports_not_enabled(task_store):
    _, state = _run(_Runner([_BALANCE, _READY]), _hops(tool="post_approve"))

    assert task_store["created"] == []
    assert state["content"] == MSG_WRITE_NOT_ENABLED


def test_balance_is_read_on_the_session_chain(task_store):
    # 同符号资产在别的链上也有余额，但比较的只能是会话所在的这条链。
    payload = {
        "balances": [
            {"chainId": 1, "symbol": "BERA", "decimals": 18, "raw": "1", "human": "1"},
            {"chainId": 80094, "symbol": "BERA", "decimals": 18, "raw": "12400000000000000000", "human": "12.4"},
        ]
    }
    _, state = _run(_Runner([payload, _READY]), _hops())

    assert state["task_id"] == "task-1"
    assert state["verdicts"] == [
        {"symbol": "BERA", "actual": "12.4", "op": "gte", "right": "10"}
    ]


def test_opening_puts_the_condition_before_the_confirmation():
    text = plan_graph.opening(
        [{"symbol": "BERA", "actual": "12.4", "op": "gte", "right": "10"}], "请确认转账。"
    )

    assert text.startswith("你持有 12.4 BERA")
    assert text.endswith("请确认转账。")


def test_reserved_threshold_is_consumed_once():
    message = f"如果超过10就转10给{_TARGET}"

    assert amount_is_quoted("10", message, reserved=("10",)) is True
    assert amount_is_quoted("10", _MESSAGE, reserved=("10",)) is False


def test_condition_without_read_step_fetches_the_balance(task_store):
    # 「超过 10 就转 5」没说「查一下」：执行器自己补一次读数，不能回「没查到」。
    hops = [
        Hop(kind="cond", symbol="BERA", op="gt", right="10"),
        Hop(
            kind="write",
            tool="post_transfer",
            args={"toAddress": _TARGET, "tokenSymbol": "BERA", "humanAmount": "5"},
        ),
    ]
    runner = _Runner([_BALANCE, _READY])
    _, state = _run(
        runner, hops, message=f"我的 BERA 超过 10 就转 5 BERA 给 {_TARGET}"
    )

    assert state["task_id"] == "task-1"
    assert runner.calls[0][0] == "get_asset_balance"
    assert runner.calls[0][1]["symbol"] == "BERA"


def test_spoken_threshold_is_accepted_via_original_fragment(task_store):
    # 「超过十」抽不出数字，门槛同样靠模型给的原话片段核对。
    hops = [
        Hop(kind="read", tool="get_asset_balance", args={"symbol": "BERA"}),
        Hop(kind="cond", symbol="BERA", op="gt", right="10", rightText="十"),
        Hop(
            kind="write",
            tool="post_transfer",
            args={"toAddress": _TARGET, "tokenSymbol": "BERA", "humanAmount": "5"},
        ),
    ]
    _, state = _run(
        _Runner([_BALANCE, _READY]),
        hops,
        message=f"我的 BERA 超过十就转 5 BERA 给 {_TARGET}",
    )

    assert state["task_id"] == "task-1"


def test_threshold_written_with_unit_is_parsed(task_store):
    # 阈值写成「10 BERA」时取开头的数字，不因为带了单位就判成没说过。
    _, state = _run(_Runner([_BALANCE, _READY]), _hops(right="10 BERA"))

    assert state["task_id"] == "task-1"


def test_task_creation_failure_reports_the_write_error(task_store, monkeypatch):
    async def _boom(payload, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(transfer_task, "create", _boom)
    _, state = _run(_Runner([_BALANCE, _READY]), _hops())

    assert state["kind"] == "answer"
    assert state["content"] == MSG_WRITE_TASK_FAILED


def test_degraded_plan_refuses_two_writes():    # 检查点不可用时退回单步，但也不能把两笔转账悄悄合成一笔。
    loop = AgentLoop(
        Settings(),
        llm=SimpleNamespace(answer_model=None),
        guard=SimpleNamespace(),
        audit=SimpleNamespace(log=AsyncMock()),
        model_with_tools=None,
        runner=None,
    )
    hops = [
        Hop(kind="read", tool="get_asset_balance", args={"symbol": "BERA"}),
        Hop(
            kind="write",
            tool="post_transfer",
            args={"toAddress": _TARGET, "tokenSymbol": "BERA", "humanAmount": "5"},
        ),
        Hop(
            kind="write",
            tool="post_transfer",
            args={"toAddress": _TARGET, "tokenSymbol": "BERA", "humanAmount": "6"},
        ),
    ]
    outcome = asyncio.run(
        loop._plan_outcome(
            hops,
            siwe_user_id="u1",
            wallet_address=_WALLET,
            chain_id=80094,
            user_message=_MESSAGE,
        )
    )

    assert outcome.kind == "ask"


def test_resume_returns_the_evidence_and_drops_the_checkpoint(task_store):
    # 确认审计要记下「在什么余额下批的这笔」，依据只能从挂起时的状态里取。
    saver = InMemorySaver()
    loop = AgentLoop(
        Settings(),
        llm=SimpleNamespace(answer_model=None),
        guard=SimpleNamespace(),
        audit=SimpleNamespace(log=AsyncMock()),
        model_with_tools=None,
        runner=_Runner([_BALANCE, _READY]),
        checkpointer=saver,
    )
    asyncio.run(
        loop._plan_graph.ainvoke(_initial(_hops()), config={"configurable": {"thread_id": "t1"}})
    )

    evidence = asyncio.run(loop.resume_plan("t1"))

    assert evidence == [{"symbol": "BERA", "actual": "12.4", "op": "gte", "right": "10"}]
    # 检查点已经清掉，再点一次确认不会重放同一份依据。
    assert asyncio.run(loop.resume_plan("t1")) is None
