"""交易记录直出：工具结果直接成形，模型不再复述一遍。"""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.agent.render import render_transactions
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import CallSource, Intent

WALLET = "0x63a9f8df422e1983d42565234557d46d18dbece9"
COUNTERPARTY = "0x1111111111111111111111111111111111111111"


def _event(value: str, *, decimals: int = 18, symbol: str | None = None,
           sender: str = WALLET, recipient: str = COUNTERPARTY,
           source: str = "txlist", contract: str | None = None) -> dict:
    return {"source": source, "from": sender, "to": recipient, "value": value,
            "decimals": decimals, "tokenSymbol": symbol, "contractAddress": contract}


def _record(*events: dict, chain_id: int = 1, hash_: str | None = None,
            sender: str = WALLET, recipient: str = COUNTERPARTY) -> dict:
    return {
        "chainId": chain_id,
        "hash": hash_ or "0x" + "a" * 64,
        "blockNumber": 21000000,
        "timeStamp": 1730000000,
        "from": sender,
        "to": recipient,
        "events": list(events),
    }


PAYLOAD = {"address": WALLET, "records": [_record(_event("50000000000000000"))]}


def test_render_keeps_amount_and_full_hash() -> None:
    text = render_transactions(PAYLOAD)

    assert "0.05 ETH" in text
    assert "转出" in text
    assert "0x" + "a" * 64 in text
    assert "最近 1 笔交易" in text


def test_render_marks_incoming_and_token_decimals() -> None:
    payload = {
        "address": WALLET,
        "records": [
            _record(
                _event("1500000", decimals=6, symbol="USDC", source="tokentx",
                       sender="0x2222222222222222222222222222222222222222", recipient=WALLET)
            )
        ],
    }

    text = render_transactions(payload)

    assert "转入 1.5 USDC" in text


def test_one_transaction_shows_every_asset_movement() -> None:
    """swap：一笔交易里有一进一出两条代币事件，都必须出现。"""
    record = _record(
        _event("0", source="txlist"),                                   # 调用 router，无原生转账
        _event("100000000", decimals=6, symbol="USDC", source="tokentx"),
        _event("20000000000000000", symbol="WETH", source="tokentx",
               sender=COUNTERPARTY, recipient=WALLET),
    )

    text = render_transactions({"address": WALLET, "records": [record]})

    assert "转出 100 USDC" in text and "转入 0.02 WETH" in text
    # 零值原生腿是合约调用本身，有非零事件时不该占位置。
    assert "0 ETH" not in text


def test_all_zero_events_read_as_contract_call() -> None:
    text = render_transactions({"address": WALLET, "records": [_record(_event("0"))]})
    assert "无资产变动" in text


def test_many_events_are_capped_with_explicit_count() -> None:
    record = _record(*[_event("1000000", decimals=6, symbol=f"T{i}", source="tokentx")
                       for i in range(10)])

    text = render_transactions({"address": WALLET, "records": [record]})

    assert "另有 2 项" in text


def test_render_empty_records_says_so() -> None:
    assert "没有查到交易记录" in render_transactions({"address": WALLET, "records": []})


class _TxCaller:
    """只回一轮工具调用。若图再回模型一步，第二次 ainvoke 就会抛错。"""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("直出路径不该再回模型")
        return AIMessage(
            content="",
            tool_calls=[{"name": "get_transaction_records", "args": {}, "id": "c1"}],
        )


class _Explainer:
    """工具报错后接手解释。它只该被调用一次。"""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="暂时查不到交易记录")


class _TxRunner:
    async def run(self, name: str, args: dict, *, source: CallSource):
        return json.dumps(PAYLOAD, ensure_ascii=False), ToolCallRecord(name=name, ok=True)


class _FailingRunner:
    async def run(self, name: str, args: dict, *, source: CallSource):
        body = '{"error": "E303", "message": "交易记录数据源未配置"}'
        return body, ToolCallRecord(name=name, ok=False, error="E303")


def _state(intent: Intent):
    return initial_state([], "我最近有哪些交易", WALLET, intent=intent)


def test_query_tx_stops_after_tools() -> None:
    model = _TxCaller()
    graph = build_read_graph(model, runner=_TxRunner(), max_steps=5)

    state = asyncio.run(graph.ainvoke(_state(Intent.QUERY_TX)))

    # 工具和参数都来自意图，这一步不再问模型。
    assert model.calls == 0
    assert state["direct_answer"].startswith("最近 1 笔交易")
    assert state["steps"] == 0


def test_other_intent_goes_back_to_model() -> None:
    model = _TxCaller()
    graph = build_read_graph(model, runner=_TxRunner(), max_steps=5)

    try:
        asyncio.run(graph.ainvoke(_state(Intent.QUERY_ASSET)))
    except AssertionError as exc:
        assert "直出路径不该再回模型" in str(exc)
    else:  # pragma: no cover - 走到这里说明边接错了
        raise AssertionError("非直出意图应当回到模型")


def test_tool_error_falls_back_to_model() -> None:
    model = _Explainer()
    graph = build_read_graph(model, runner=_FailingRunner(), max_steps=5)

    state = asyncio.run(graph.ainvoke(_state(Intent.QUERY_TX)))

    assert model.calls == 1
    assert state["direct_answer"] == ""
    assert state["messages"][-1].content == "暂时查不到交易记录"
