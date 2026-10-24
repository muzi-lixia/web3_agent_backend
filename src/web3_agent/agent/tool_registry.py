"""ToolRegistry：工具分组元数据。

写工具（T4–T7）不进 llm_visible，也不进 bind_tools——LLM 看不到它们，
而不是"看得到但被拦住"。这里的断言是这个约束的最后一道保险。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from web3_agent.config.constants import ToolGroup


# 数据职责：工具实例的注册元数据，决定可见性与调用来源限制。
@dataclass(frozen=True)
class ToolSpec:
    # tool：封装好的 LangChain 工具实例。
    tool: BaseTool
    # group：工具可见性分组，区分模型可用与仅状态机可用。
    group: ToolGroup
    # mutating：工具是否修改链上状态；为真时不得暴露给模型。
    mutating: bool


# 数据职责：按名称保存工具注册信息，并筛选安全的模型可见列表。
class ToolRegistry:
    def __init__(self) -> None:
        # _specs：工具名到 ToolSpec 的进程内映射。
        """初始化按工具名称索引的注册表。"""
        self._specs: dict[str, ToolSpec] = {}

    def register(self, tool: BaseTool, group: ToolGroup, mutating: bool) -> None:
        """按工具名称注册实例、可见分组和是否修改链上状态。

        Args:
            tool: 需要注册的 LangChain 工具实例。
            group: 工具可见性分组。
            mutating: 该工具是否会修改链上状态。
        """
        self._specs[tool.name] = ToolSpec(tool=tool, group=group, mutating=mutating)

    def get(self, name: str) -> ToolSpec:
        """按名称读取注册工具规格；不存在的名称由字典抛出错误。

        Args:
            name: 注册表中的工具名称。
        """
        return self._specs[name]

    def llm_tools(self) -> list[BaseTool]:
        """筛选模型可见工具，并拒绝把修改状态的工具暴露给模型。"""
        tools = []
        for spec in self._specs.values():
            if spec.group is not ToolGroup.LLM_VISIBLE:
                continue
            if spec.mutating:
                raise RuntimeError(f"{spec.tool.name} 是 mutating 工具，不能进 bind_tools")
            tools.append(spec.tool)
        return tools

    def names(self, group: ToolGroup) -> list[str]:
        """列出指定可见性分组中的工具名。

        Args:
            group: 工具可见性分组。
        """
        return [s.tool.name for s in self._specs.values() if s.group is group]
