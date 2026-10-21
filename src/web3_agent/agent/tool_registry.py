"""ToolRegistry：工具分组元数据。

写工具（T4–T7）不进 llm_visible，也不进 bind_tools——LLM 看不到它们，
而不是"看得到但被拦住"。这里的断言是这个约束的最后一道保险。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from web3_agent.config.constants import ToolGroup


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    group: ToolGroup
    mutating: bool


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, tool: BaseTool, group: ToolGroup, mutating: bool) -> None:
        self._specs[tool.name] = ToolSpec(tool=tool, group=group, mutating=mutating)

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def llm_tools(self) -> list[BaseTool]:
        tools = []
        for spec in self._specs.values():
            if spec.group is not ToolGroup.LLM_VISIBLE:
                continue
            if spec.mutating:
                raise RuntimeError(f"{spec.tool.name} 是 mutating 工具，不能进 bind_tools")
            tools.append(spec.tool)
        return tools

    def names(self, group: ToolGroup) -> list[str]:
        return [s.tool.name for s in self._specs.values() if s.group is group]
