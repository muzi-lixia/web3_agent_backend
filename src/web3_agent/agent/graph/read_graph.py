"""只读 ReAct 子图：agent → tools → agent → …，步数上限硬约束。

写路径不在这张图里。LangGraph 只负责只读循环，写操作由状态机接管，
这样"LLM 能不能编排资金动作"这个问题在结构上就不存在。
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from web3_agent.agent.graph.state import ReadState
from web3_agent.agent.prompts import READ_SYSTEM
from web3_agent.rag.answer_prompt import answer_messages
from web3_agent.common.exceptions import AppError, ToolSourceViolation
from web3_agent.common.types import ToolCallRecord
from web3_agent.agent.render import render_transactions
from web3_agent.agent.sanitize import wrap_chain_data
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.config.constants import CallSource, Intent
from web3_agent.web3.chain_adapter import is_evm_address
from web3_agent.common.metering import Resource, reserve

logger = logging.getLogger(__name__)

# 这些工具的地址参数来自会话身份，不允许模型自己编。
# _ADDRESS_ARG：不同链上工具需要注入的钱包参数名称映射。
_ADDRESS_ARG = {
    "get_asset_balance": "address",
    "get_transaction_records": "address",
    "get_stake_info": "userAddress",
}

# 地址必须等于当前登录地址的工具：模型传什么都不采纳，直接覆盖。
# 交易记录只允许查自己的账户，这是业务限制，不能在提示词或某条路由里兑现。
# _SESSION_BOUND：必须绑定当前登录钱包、不允许模型选择其他地址的工具集合。
_SESSION_BOUND = frozenset({"get_transaction_records"})

# _RAG_TOOL：知识问答使用的检索工具稳定名称。
_RAG_TOOL = "rag_general_knowledge"

# 意图 → 本轮要拿到数据的工具。意图识别已经做过一次判断，这个结果要传进图里，
# 否则图里的模型会拿着同一句话从零再判断一遍要不要调工具。
# 一张表两个用途：执行闸看它（加上知识检索），证据校验也看它。
# _INTENT_TOOLS：意图与业务工具及标签的映射，供执行限制和证据检查使用。
_INTENT_TOOLS: dict[Intent, frozenset[str]] = {
    Intent.QUERY_ASSET: frozenset({"get_asset_balance"}),
    Intent.QUERY_TX: frozenset({"get_transaction_records"}),
    Intent.QUERY_STAKE: frozenset({"get_stake_info"}),
    Intent.QUERY_KNOWLEDGE: frozenset({_RAG_TOOL}),
}


def evidence_tools(intent: Intent | None) -> frozenset[str]:
    """该意图必须拿到结果的工具。不在表内（闲聊、越界、写意图）返回空集，不做证据校验。

    知识检索在表外单独放：它回答「这是什么、为什么」，不能证明「我账上有多少钱」。

    Args:
        intent: 已由后端确认的本轮业务意图。
    """
    if intent is None:
        return frozenset()
    return _INTENT_TOOLS.get(intent, frozenset())


def allowed_tools(intent: Intent | None) -> frozenset[str] | None:
    """本轮允许执行的工具集合。None 表示不限。

    模型看到的工具列表是全局的，它随时可以点名别的工具；这一层是执行侧的
    最后一道闸，与提示词和正常路由无关。

    知识检索对所有查询意图都开放：一句话既要数据又要解释时，两个工具都该能调。

    Args:
        intent: 已由后端确认的本轮业务意图。
    """
    if intent is None:
        return None
    tools = evidence_tools(intent)
    if not tools:
        return None
    return tools | {_RAG_TOOL}


# 结果本身就是答案的工具：调用成功即成形返回，模型不再复述一遍。
# 这一类连「要不要调工具」都不必再问模型 —— 工具和参数都已经确定。
# _DIRECT_RENDER：支持由工具结果确定性渲染答案的工具及对应渲染器。
_DIRECT_RENDER = {"get_transaction_records": render_transactions}


def _render_direct(name: str, payload: str) -> str:
    """工具原始输出 → 直出文本。出错或形状不对时返回空串，交回模型解释。

    Args:
        name: 工具名、知识库名称或追踪节点名，具体由当前方法职责决定。
        payload: 工具返回的 JSON 文本。
    """
    renderer = _DIRECT_RENDER.get(name)
    if renderer is None:
        return ""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict) or "error" in data:
        return ""
    return renderer(data)


def build_read_graph(
    model_with_tools: Runnable,
        runner: ToolRunner,
        max_steps: int,
        rag_answer_model: Runnable | None = None
) -> Runnable:
    # 返回值：已编译的 LangGraph Runnable，可通过 ainvoke 执行。
    """构造并编译只读图，连接模型、工具执行和直接回答节点。

    Args:
        model_with_tools: 已绑定只读工具的回答模型。
        runner: 工具统一执行器，负责来源检查和调用轨迹。
        max_steps: 图允许的最大模型调用步数。
        rag_answer_model: 只接收检索证据的知识回答模型。
    """
    async def call_model(state: ReadState) -> dict:
        """根据图状态选择普通回答或知识回答模型，调用前记录模型消耗并递增步数。

        Args:
            state: 当前只读图的状态快照。
        """
        if state.get("rag_required"):
            if rag_answer_model is None:
                raise RuntimeError("RAG 回答模型未配置")
            model = rag_answer_model
            messages = answer_messages(
                state.get("rag_question") or state.get("rag_query", ""),
                state.get("rag_citations", []),
                state.get("rag_answer_context", ""),
            )
        else:
            model = model_with_tools
            messages = state["messages"]
        # 步数上限只挡住"走几轮"，模型调用本身要单独计数：这一轮失败或超时，
        # 已经发生的消耗仍然要能结算。
        reserve(Resource.MODEL)
        response = await model.ainvoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    async def call_tools(state: ReadState) -> dict:
        # 节点返回 messages 与 tool_calls 追加到历史状态；rag_citations、rag_status、rag_question 覆盖更新。
        # 工具可直接渲染成功时补 direct_answer，供路由立即终止而无需模型复述。
        """执行当前模型消息中的工具调用，校验工具范围和身份，累积结果及引用。

        Args:
            state: 当前只读图的状态快照。
        """
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}
        outputs: list[ToolMessage] = []
        records: list[dict] = []
        direct = ""
        citations = list(state.get("rag_citations", []))
        rag_status = state.get("rag_status", "")
        rag_question = state.get("rag_question", "")
        allowed = allowed_tools(state.get("intent"))
        wallet = state.get("wallet_address", "")

        def refuse(call: dict, code: str, message: str) -> None:
            """拒绝一次调用：给模型一条明确的 ToolMessage，同时记成失败，不构成证据。

            Args:
                call: 工具调用描述，或不带参数的可等待模型调用函数。
                code: 已登记的业务错误码。
                message: 按当前函数用途传入的用户文本或已解析 SIWE 消息。
            """
            outputs.append(
                ToolMessage(
                    content=wrap_chain_data(
                        json.dumps({"error": code, "message": message}, ensure_ascii=False)
                    ),
                    tool_call_id=call.get("id", ""),
                    name=call["name"],
                )
            )
            records.append(
                ToolCallRecord(
                    name=call["name"], args=dict(call.get("args") or {}), ok=False, error=code
                ).model_dump()
            )

        for call in last.tool_calls:
            name = call["name"]
            args = dict(call.get("args", {}))

            # 本轮意图只认一个工具：模型点名别的工具就拒绝，已发生的调用无法撤回。
            if allowed is not None and name not in allowed:
                logger.warning("tool rejected intent=%s tool=%s", state.get("intent"), name)
                refuse(call, "E101", f"本轮只允许调用 {'、'.join(sorted(allowed))}，已拒绝 {name}")
                continue

            # 地址参数来自会话身份，不由模型决定。
            arg_name = _ADDRESS_ARG.get(name)
            if arg_name:
                if name in _SESSION_BOUND:
                    if not wallet:
                        refuse(call, "E503", "该查询只能针对当前登录地址，请先连接钱包")
                        continue
                    if str(args.get(arg_name, "")).lower() != wallet.lower():
                        logger.warning("address overridden tool=%s intent=%s", name, state.get("intent"))
                    args[arg_name] = wallet
                elif wallet and not is_evm_address(str(args.get(arg_name, ""))):
                    # 模型编出来的占位地址（如 0xYourAddressHere）不能让用户看到"地址不合法"。
                    args[arg_name] = wallet

            if name == "rag_general_knowledge":
                # 保留原始问题，过滤条件由模型基于明确上下文按需提供。
                args["query_str"] = state.get("rag_query", "")
                args["kb_id"] = state.get("rag_kb_id", "default")
            try:
                content, record = await runner.run(name, args, source=CallSource.LLM)
            except AppError:
                raise
            except ToolSourceViolation:
                # LLM 越界调用写工具：这是内部不变量被打破，不能降级成一条模型
                # 可以继续对话的工具报错。
                logger.error("tool source violation tool=%s", name)
                raise
            except Exception as exc:  # noqa: BLE001 未知工具名也不能让图崩
                # 异常文本走 json.dumps：f-string 拼 JSON，异常里带引号就会生成非法 JSON。
                content = json.dumps({"error": "E101", "message": str(exc)}, ensure_ascii=False)
                record = None
            if name == "rag_general_knowledge":
                try:
                    payload = json.loads(content)
                    rag_status = payload.get("status", "unavailable")
                    # 检索层已经消解了指代，回答直接用这一版：否则召回理解的是另一个问题。
                    rag_question = payload.get("resolved_question") or rag_question
                    for hit in payload.get("hits", []):
                        existing = next((item for item in citations if item["chunk_id"] == hit["chunk_id"]), None)
                        if existing is None:
                            hit["id"] = f"S{len(citations)+1}"
                            citations.append(hit)
                        else:
                            hit["id"] = existing["id"]
                    content = json.dumps(payload, ensure_ascii=False)
                except (ValueError, TypeError, KeyError):
                    rag_status = "unavailable"
            outputs.append(
                ToolMessage(
                    content=wrap_chain_data(content),
                    tool_call_id=call.get("id", ""),
                    name=name,
                )
            )
            if record is not None:
                records.append(record.model_dump())
                if name == state.get("direct_tool") and record.ok:
                    direct = direct or _render_direct(name, content)
        update: dict = {"messages": outputs, "tool_calls": records, "rag_citations": citations,
                        "rag_status": rag_status, "rag_question": rag_question}
        if direct:
            update["direct_answer"] = direct
        return update

    def should_continue(state: ReadState) -> str:
        """根据知识问答状态、模型步数和工具调用决定终止或执行工具。

        Args:
            state: 当前只读图的状态快照。
        """
        if state.get("rag_required"):
            return END
        if state.get("steps", 0) >= max_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_tools(state: ReadState) -> str:
        # 知识检索失败时直接使用固定兜底，不让模型凭空补规则。
        """根据证据和直出结果决定结束图或回到回答模型。

        Args:
            state: 当前只读图的状态快照。
        """
        if state.get("rag_required") and state.get("rag_status") and not state.get("rag_citations"):
            return END
        # 结果已成形就直接收尾，省掉一整步复述。
        return END if state.get("direct_answer") else "agent"

    def plan(state: ReadState) -> dict:
        """直出意图：直接合成这一次工具调用。

        模型在这一步只做两件事 —— 选工具、填参数 —— 两者都已经有确定值。

        Args:
            state: 当前只读图的状态快照。
        """
        if state.get("rag_required"):
            name = "rag_general_knowledge"
            args = state["rag_args"]
        else:
            # 保留原有交易记录直出链路
            name = state["direct_tool"]
            args = state["direct_args"]
        call = {"name": name, "args": args, "id": f"direct-{name}"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    def first_step(state: ReadState) -> str:
        """决定图入口走确定性工具计划还是模型推理节点。

        Args:
            state: 当前只读图的状态快照。
        """
        if state.get("rag_required") or state.get("direct_tool"):
            return "plan"
        return "agent"

    builder = StateGraph(ReadState)
    builder.add_node("plan", plan)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.set_conditional_entry_point(first_step, {"plan": "plan", "agent": "agent"})
    builder.add_edge("plan", "tools")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return builder.compile()


# _ASSISTANT_CONTEXT_CHARS：提供给 RAG 指代消解的单条助手历史字符上限。
_ASSISTANT_CONTEXT_CHARS = 200
# _SOURCE_TAIL：旧助手回答末尾来源清单的匹配规则。
_SOURCE_TAIL = re.compile(r"(来源[:：].*)$", re.S)
# _CITATION_MARK：历史引用编号匹配规则，用于避免旧证据进入当前回答。
_CITATION_MARK = re.compile(r"\[S\d+\]")


def _answer_context(history: list[tuple[str, str]]) -> str:
    """给回答模型一小段此前的对话，只为消解"刚才那个""第二种方案"这类指代。

    助手的话截到 200 字并去掉来源清单与引用编号：上一轮列过什么要能看见，
    但旧来源和旧结论不能跟着进证据区。真正的依据只有本轮检索材料。

    Args:
        history: 按角色组织的近期对话列表。
    """
    lines = []
    for role, content in history[-4:]:
        text = " ".join(content.split())
        if role == "assistant":
            text = _CITATION_MARK.sub("", _SOURCE_TAIL.sub("", text))
            text = text[:_ASSISTANT_CONTEXT_CHARS]
        lines.append(f"{'助手' if role == 'assistant' else '用户'}：{text}")
    return "\n".join(lines)[-2000:]


def initial_state(
    history: list[tuple[str, str]],
    user_message: str,
    wallet_address: str = "",
    summary: str = "",
    profile: str = "",
    memories: str = "",
    intent: Intent | None = None,
    kb_id: str = "default",
) -> ReadState:
    # 返回值：包含消息、执行限制、钱包和知识检索参数的 ReadState。
    """初始化图消息、身份、意图及 RAG 状态，并构造可确定的工具参数。

    Args:
        history: 按角色组织的近期对话列表。
        user_message: 本轮待处理的用户输入。
        wallet_address: 服务端确认的当前钱包地址。
        summary: 已渲染的历史摘要文本。
        profile: 用户偏好对象或知识库嵌入配置，取决于当前模块。
        memories: 已渲染的用户长期背景文本。
        intent: 已由后端确认的本轮业务意图。
        kb_id: 目标知识库标识，默认 default。
    """
    is_rag = intent is Intent.QUERY_KNOWLEDGE
    tools = evidence_tools(intent)
    tool = next((t for t in sorted(tools) if t in _DIRECT_RENDER), "")
    if is_rag:
        # 回答模型的输入由 answer_messages() 单独构造，图里这条链只需要承载
        # plan → tools 的工具轨迹。在这里再拼一套 RAG 提示词不会发给任何模型，
        # 改了也不生效 —— 留着只会让人以为它管用。
        messages: list[AnyMessage] = [HumanMessage(content=user_message)]
    else:
        identity = (
            f"当前用户已连接的钱包地址是 {wallet_address}。用户说「我」「我的」时指这个地址，"
            "调用工具必须使用它，不要自行编造地址。"
            if wallet_address
            else "当前没有已连接的钱包地址。需要地址时先问用户，不要编造。"
        )
        # 档案 → 摘要，都是本系统产出的结构化约束。记忆不进来：它整段来自用户，
        # 混在规则里会让「以前说过的话」看起来像「现在要执行的指令」。
        system = f"{READ_SYSTEM}\n\n{identity}"
        if profile:
            system = f"{system}\n\n{profile}"
        if summary:
            system = f"{system}\n\n{summary}"
        # 意图放在最后：它是本轮的执行约束，离当前输入越近越管用。
        if tools:
            system = (
                f"{system}\n\n本轮意图已识别为 {intent.value}。"
                f"必须先调用 {'、'.join(sorted(tools))} 取得数据再回答，不得凭上下文或记忆作答。"
            )
        messages = [SystemMessage(content=system)]
        if memories:
            # 用户背景单独成块、带标签：边界由结构给出，不靠模型从一段长文本里自己分辨。
            # 它排在规则之后、历史之前 —— 描述「这个人是谁」，不参与「这一轮怎么答」。
            messages.append(
                SystemMessage(content=f"<user_background>\n{memories}\n</user_background>")
            )
        for role, content in history:
            messages.append(
                AIMessage(content=content) if role == "assistant" else HumanMessage(content=content)
            )
        messages.append(HumanMessage(content=user_message))
    # 地址是直出调用的必填参数，没有它就只能回退到模型那条路。
    direct = bool(tool and wallet_address)
    rag_args = {"query_str": user_message, "kb_id": kb_id}
    if is_rag and history:
        # Only the program supplies bounded history, never model-generated filter slots.
        rag_args["conversation_context"] = "\n".join(
            f"{role}: {content}" for role, content in history[-4:]
        )[-2000:]
    return {
        "messages": messages,
        "steps": 0,
        "tool_calls": [],
        "wallet_address": wallet_address,
        "intent": intent,
        "direct_tool": tool if direct else "",
        # 交易记录只按地址查，条数固定由工具决定 —— 分页和条数筛选都还没做。
        "direct_args": {"address": wallet_address} if direct else {},
        "direct_answer": "",
        "rag_kb_id": kb_id,
        "rag_query": user_message,
        # 检索层会把它换成消解指代后的版本；这里先放原句，改写失败时就是它。
        "rag_question": user_message,
        "rag_citations": [],
        "rag_status": "",
        "rag_required": intent is Intent.QUERY_KNOWLEDGE,
        "rag_answer_context": _answer_context(history) if is_rag else "",
        "rag_args": rag_args
    }

if __name__ == "__main__":
    graph = build_read_graph(
        model_with_tools=None,
        runner=None,
        max_steps=5,
    )

    # 打印 Mermaid 图代码
    print(graph.get_graph().draw_mermaid())

    # 如需生成图片，取消下面这行注释；默认需要联网
    # graph.get_graph().draw_mermaid_png(output_file_path="../../../../../read_graph.png")
