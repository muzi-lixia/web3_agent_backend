"""只读 ReAct 图的状态。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ReadState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    tool_calls: Annotated[list[dict], operator.add]
    wallet_address: str
    # 本轮允许直出的工具名。命中且调用成功时结果成形即答案，不再回模型。
    direct_tool: str
    # 直出工具的调用参数。地址与条数在意图阶段就定了，不必再问模型。
    direct_args: dict
    direct_answer: str
    rag_kb_id: str
    rag_query: str
    rag_citations: list[dict]
    rag_status: str
    rag_required: bool
    rag_args: dict
