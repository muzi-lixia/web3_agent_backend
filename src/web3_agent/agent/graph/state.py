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
