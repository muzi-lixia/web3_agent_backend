"""只读 ReAct 图的状态。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from web3_agent.config.constants import Intent


# 数据职责：LangGraph 只读状态：消息由 add_messages 合并，调用轨迹按列表追加，其余字段覆盖更新。
class ReadState(TypedDict):
    # messages：按角色组织的消息序列；后台任务为 (role, content)，图状态为 LangChain 消息。
    messages: Annotated[list[AnyMessage], add_messages]
    # steps：本轮只读图累计的模型调用步数，不等同于下游 RPC 请求数。
    steps: int
    # tool_calls：本轮工具调用轨迹，包含名称、参数及执行结果；未调用时可为空。
    tool_calls: Annotated[list[dict], operator.add]
    # wallet_address：服务端当前登录钱包地址，用于注入工具身份参数。
    wallet_address: str
    # 本轮意图，决定这一轮允许调用哪些工具。空值表示不限（不是从对话入口进的图）。
    # intent：本轮识别出的业务意图；取值受 Intent 枚举约束。
    intent: Intent | None
    # 本轮允许直出的工具名。命中且调用成功时结果成形即答案，不再回模型。
    # direct_tool：本轮可直接执行并渲染答案的工具名；为空时走模型决策。
    direct_tool: str
    # 直出工具的调用参数。地址与条数在意图阶段就定了，不必再问模型。
    # direct_args：后端已确定的直出工具参数，不依赖模型重新生成。
    direct_args: dict
    # direct_answer：工具结果直接渲染的最终文本；有值时无需模型再次复述。
    direct_answer: str
    # rag_kb_id：本轮知识检索锁定的知识库标识。
    rag_kb_id: str
    # rag_query：本轮知识问题的原句，保留用于工具调用和展示。
    rag_query: str
    # 检索阶段消解指代后的问题。召回、重排、回答共用它；rag_query 保留原句供展示与落库。
    # rag_question：检索层消解指代后的独立问题，召回、重排和回答共用。
    rag_question: str
    # rag_citations：本轮检索得到的文档证据，编号后传给回答模型。
    rag_citations: list[dict]
    # rag_status：本轮知识检索状态，如 success、empty、unavailable。
    rag_status: str
    # rag_required：是否为知识问答；为真时由后端安排检索再生成回答。
    rag_required: bool
    # rag_args：后端构造的检索参数，包含问题、知识库及受限历史。
    rag_args: dict
    # rag_answer_context：仅用于理解指代的历史片段，不作为回答事实证据。
    rag_answer_context: str
