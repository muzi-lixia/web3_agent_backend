"""交易记录直出：连「要不要调工具」都不再问模型。

意图已经给出工具和参数，模型在这一步只做两件本来就有答案的事（选工具、填参数），
所以直接合成这次调用，省掉一整次全上下文往返。
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import CallSource, Intent

WALLET = "0x63a9f8df422e1983d42565234557d46d18dbece9"


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="这一步不该问模型")


class _Runner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, name: str, args: dict, *, source: CallSource):
        self.calls.append(args)
        payload = {
            "address": args["address"],
            "records": [
                {
                    "chainId": 1,
                    "hash": "0x9f2c" * 16,
                    "timeStamp": 1730000000,
                    "from": WALLET,
                    "to": "0x1111111111111111111111111111111111111111",
                    "events": [
                        {
                            "source": "txlist",
                            "from": WALLET,
                            "to": "0x1111111111111111111111111111111111111111",
                            "value": "50000000000000000",
                            "decimals": 18,
                            "tokenSymbol": None,
                            "contractAddress": None,
                        }
                    ],
                }
            ],
        }
        return json.dumps(payload), ToolCallRecord(name=name, ok=True)


def test_query_tx_takes_the_tool_call_from_the_intent() -> None:
    model, runner = _Model(), _Runner()
    graph = build_read_graph(model, runner=runner, max_steps=5)

    state = asyncio.run(
        graph.ainvoke(
            initial_state([], "查我最近 3 笔交易", WALLET, intent=Intent.QUERY_TX)
        )
    )

    assert model.calls == 0
    assert runner.calls == [{"address": WALLET}]
    assert state["direct_answer"].startswith("最近 1 笔交易")


def test_without_a_wallet_the_model_still_decides() -> None:
    """没有会话地址就凑不出调用参数，回退到模型那条路。"""
    model, runner = _Model(), _Runner()
    graph = build_read_graph(model, runner=runner, max_steps=5)

    asyncio.run(graph.ainvoke(initial_state([], "查我最近 3 笔交易", "", intent=Intent.QUERY_TX)))

    assert model.calls == 1
    assert runner.calls == []
