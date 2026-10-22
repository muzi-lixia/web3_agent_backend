"""只读 ReAct 子图：agent → tools → agent → …，步数上限硬约束。

写路径不在这张图里。LangGraph 只负责只读循环，写操作由状态机接管，
这样"LLM 能不能编排资金动作"这个问题在结构上就不存在。
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from web3_agent.agent.graph.state import ReadState
from web3_agent.agent.prompts import READ_SYSTEM
from web3_agent.agent.render import render_transactions
from web3_agent.agent.sanitize import wrap_chain_data
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.config.constants import CallSource, Intent
from web3_agent.web3.chain_adapter import is_evm_address

# 这些工具的地址参数来自会话身份，不允许模型自己编。
_ADDRESS_ARG = {
    "get_asset_balance": "address",
    "get_transaction_records": "address",
    "get_stake_info": "userAddress",
}

# 意图 → 必须调用的工具。意图识别已经做过一次判断，这个结果要传进图里，
# 否则图里的模型会拿着同一句话从零再判断一遍要不要调工具。
_INTENT_TOOLS = {
    Intent.QUERY_ASSET: ("get_asset_balance", "资产余额"),
    Intent.QUERY_TX: ("get_transaction_records", "交易记录"),
    Intent.QUERY_STAKE: ("get_stake_info", "质押仓位"),
}

# 结果本身就是答案的工具：调用成功即成形返回，模型不再复述一遍。
# 这一类连「要不要调工具」都不必再问模型 —— 工具和参数都已经确定。
_DIRECT_RENDER = {"get_transaction_records": render_transactions}

# 交易记录工具的条数默认值，与工具签名保持一致。
_DEFAULT_TX_LIMIT = 10


def _slot_limit(slots: dict) -> int:
    """意图里带的条数。缺失或不像数字时返回 0，表示交给工具默认值。"""
    try:
        return max(1, min(int(slots.get("limit")), 50))
    except (TypeError, ValueError):
        return 0


def _render_direct(name: str, payload: str) -> str:
    """工具原始输出 → 直出文本。出错或形状不对时返回空串，交回模型解释。"""
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
    model_with_tools: Runnable, runner: ToolRunner, max_steps: int
) -> Runnable:
    async def call_model(state: ReadState) -> dict:
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    async def call_tools(state: ReadState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}
        outputs: list[ToolMessage] = []
        records: list[dict] = []
        direct = ""
        for call in last.tool_calls:
            args = dict(call.get("args", {}))

            # 模型编出来的占位地址（如 0xYourAddressHere）不能让用户看到"地址不合法"，
            # 用会话里的真实地址回填；有效地址则保留，允许查他人地址。
            arg_name = _ADDRESS_ARG.get(call["name"])
            wallet = state.get("wallet_address", "")
            if arg_name and wallet and not is_evm_address(str(args.get(arg_name, ""))):
                args[arg_name] = wallet

            try:
                content, record = await runner.run(
                    call["name"], args, source=CallSource.LLM
                )
            except Exception as exc:  # noqa: BLE001 未知工具名也不能让图崩
                content = f'{{"error": "E101", "message": "{exc}"}}'
                record = None
            outputs.append(
                ToolMessage(
                    content=wrap_chain_data(content),
                    tool_call_id=call.get("id", ""),
                    name=call["name"],
                )
            )
            if record is not None:
                records.append(record.model_dump())
                if call["name"] == state.get("direct_tool") and record.ok:
                    direct = direct or _render_direct(call["name"], content)
        update: dict = {"messages": outputs, "tool_calls": records}
        if direct:
            update["direct_answer"] = direct
        return update

    def should_continue(state: ReadState) -> str:
        if state.get("steps", 0) >= max_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_tools(state: ReadState) -> str:
        # 结果已成形就直接收尾，省掉一整步复述。
        return END if state.get("direct_answer") else "agent"

    def plan(state: ReadState) -> dict:
        """直出意图：直接合成这一次工具调用。

        模型在这一步只做两件事 —— 选工具、填参数 —— 两者都已经有确定值。
        """
        name = state["direct_tool"]
        call = {"name": name, "args": state["direct_args"], "id": f"direct-{name}"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    def first_step(state: ReadState) -> str:
        return "plan" if state.get("direct_tool") else "agent"

    builder = StateGraph(ReadState)
    builder.add_node("plan", plan)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.set_conditional_entry_point(first_step, {"plan": "plan", "agent": "agent"})
    builder.add_edge("plan", "tools")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return builder.compile()


def initial_state(
    history: list[tuple[str, str]],
    user_message: str,
    wallet_address: str = "",
    summary: str = "",
    profile: str = "",
    memories: str = "",
    intent: Intent | None = None,
    slots: dict | None = None,
) -> ReadState:
    identity = (
        f"当前用户已连接的钱包地址是 {wallet_address}。用户说「我」「我的」时指这个地址，"
        "调用工具必须使用它，不要自行编造地址。"
        if wallet_address
        else "当前没有已连接的钱包地址。需要地址时先问用户，不要编造。"
    )
    system = f"{READ_SYSTEM}\n\n{identity}"
    # 档案 → 背景 → 摘要。前两者是最稳定的长期约束，摘要是一次压缩事实，
    # 都排在原文窗口之前。
    if profile:
        system = f"{system}\n\n{profile}"
    if memories:
        system = f"{system}\n\n{memories}"
    if summary:
        system = f"{system}\n\n{summary}"
    # 意图放在最后：它是本轮的执行约束，离当前输入越近越管用。
    tool = ""
    if intent is not None and intent in _INTENT_TOOLS:
        tool, label = _INTENT_TOOLS[intent]
        system = (
            f"{system}\n\n本轮意图已识别为 {intent.value}（{label}查询）。"
            f"必须先调用 {tool} 取得数据再回答，不得凭上下文或记忆作答。"
        )
    messages = [SystemMessage(content=system)]
    for role, content in history:
        messages.append(
            AIMessage(content=content) if role == "assistant" else HumanMessage(content=content)
        )
    messages.append(HumanMessage(content=user_message))
    # 地址是直出调用的必填参数，没有它就只能回退到模型那条路。
    direct = tool in _DIRECT_RENDER and bool(wallet_address)
    return {
        "messages": messages,
        "steps": 0,
        "tool_calls": [],
        "wallet_address": wallet_address,
        "direct_tool": tool if direct else "",
        "direct_args": (
            {"address": wallet_address, "limit": _slot_limit(slots or {}) or _DEFAULT_TX_LIMIT}
            if direct
            else {}
        ),
        "direct_answer": "",
    }
