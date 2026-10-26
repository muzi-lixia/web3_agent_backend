"""一次性冒烟检查：不启服务，直接过一遍装配 + 意图抽取 + 只读图。

用法：.venv/bin/python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web3_agent.agent.json_guard import JsonGuard  # noqa: E402
from web3_agent.agent.llm_adapter import LlmAdapter  # noqa: E402
from web3_agent.agent.tool_registry import ToolRegistry  # noqa: E402
from web3_agent.agent.tool_runner import ToolRunner  # noqa: E402
from web3_agent.config.constants import CallSource, ToolGroup  # noqa: E402
from web3_agent.config.settings import Settings  # noqa: E402
from web3_agent.web3.chain_adapter import ChainAdapter  # noqa: E402
from web3_agent.web3.tools.read.get_asset_balance import build_get_asset_balance  # noqa: E402
from web3_agent.web3.tools.read.get_stake_info import build_get_stake_info  # noqa: E402
from web3_agent.web3.tools.read.get_transaction_records import build_get_transaction_records  # noqa: E402


async def main() -> None:
    """构造真实模型和链工具执行人工冒烟流程，可按 --graph 额外运行只读图。"""
    settings = Settings()
    print("model:", settings.model_name)

    adapter = ChainAdapter(settings)
    registry = ToolRegistry()
    registry.register(build_get_asset_balance(adapter), ToolGroup.LLM_VISIBLE, mutating=False)
    registry.register(
        build_get_transaction_records(settings), ToolGroup.LLM_VISIBLE, mutating=False
    )
    registry.register(
        build_get_stake_info(adapter, settings.stake_contracts), ToolGroup.LLM_VISIBLE, mutating=False
    )
    print("llm_visible tools:", [t.name for t in registry.llm_tools()])

    llm = LlmAdapter(settings)
    print("model check:", await llm.check_model())

    guard = JsonGuard()
    for message in ["我的 ETH 余额是多少", "帮我转 10 USDC 给 0x" + "1" * 40, "今天天气怎么样"]:
        result = await guard.run(lambda m=message: llm.extract_intent([], m))
        print(f"{message!r:44s} -> {result.output}")

    runner = ToolRunner(registry)
    print("tool direct call:", await runner.run(
        "get_asset_balance", {"address": "0x" + "1" * 40, "topN": 5}, source=CallSource.LLM
    ))

    if "--graph" in sys.argv:
        from web3_agent.agent.graph.read_graph import build_read_graph, initial_state

        graph = build_read_graph(
            llm.bind_tools(registry.llm_tools()),
            runner=runner,
            max_steps=settings.read_max_react_steps,
        )
        state = await graph.ainvoke(
            initial_state(
                [], "帮我查一下我的 ETH 余额", "0x1111111111111111111111111111111111111111"
            )
        )
        print("graph steps:", state["steps"])
        print("tool calls:", state["tool_calls"])
        print("answer:", state["messages"][-1].content[:300])

    await adapter.aclose()


if __name__ == "__main__":
    asyncio.run(main())
