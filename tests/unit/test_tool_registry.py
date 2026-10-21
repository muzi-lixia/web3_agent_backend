import pytest
from langchain_core.tools import StructuredTool

from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.config.constants import ToolGroup


def _tool(name: str) -> StructuredTool:
    def _run(x: int = 0) -> int:
        return x

    return StructuredTool.from_function(func=_run, name=name, description=f"{name} test tool")


def test_only_read_tools_are_llm_visible() -> None:
    registry = ToolRegistry()
    registry.register(_tool("get_asset_balance"), ToolGroup.LLM_VISIBLE, mutating=False)
    registry.register(_tool("build_stake_transaction"), ToolGroup.STATEMACHINE_ONLY, mutating=True)

    names = [t.name for t in registry.llm_tools()]
    assert names == ["get_asset_balance"]
    assert registry.names(ToolGroup.STATEMACHINE_ONLY) == ["build_stake_transaction"]


def test_mutating_tool_cannot_be_llm_visible() -> None:
    registry = ToolRegistry()
    registry.register(_tool("build_transfer_transaction"), ToolGroup.LLM_VISIBLE, mutating=True)
    with pytest.raises(RuntimeError):
        registry.llm_tools()
