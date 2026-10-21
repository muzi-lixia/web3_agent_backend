"""只读 ReAct 子图：agent → tools → agent → …，步数上限硬约束。

写路径不在这张图里。LangGraph 只负责只读循环，写操作由状态机接管，
这样"LLM 能不能编排资金动作"这个问题在结构上就不存在。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from web3_agent.agent.graph.state import ReadState
from web3_agent.agent.prompts import READ_SYSTEM
from web3_agent.agent.sanitize import wrap_chain_data
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.config.constants import CallSource
from web3_agent.web3.chain_adapter import is_evm_address

# 这些工具的地址参数来自会话身份，不允许模型自己编。
_ADDRESS_ARG = {
    "get_asset_balance": "address",
    "get_transaction_records": "address",
    "get_stake_info": "userAddress",
}


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
        return {"messages": outputs, "tool_calls": records}

    def should_continue(state: ReadState) -> str:
        if state.get("steps", 0) >= max_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    builder = StateGraph(ReadState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def initial_state(
    history: list[tuple[str, str]],
    user_message: str,
    wallet_address: str = "",
    summary: str = "",
    profile: str = "",
) -> ReadState:
    identity = (
        f"当前用户已连接的钱包地址是 {wallet_address}。用户说「我」「我的」时指这个地址，"
        "调用工具必须使用它，不要自行编造地址。"
        if wallet_address
        else "当前没有已连接的钱包地址。需要地址时先问用户，不要编造。"
    )
    system = f"{READ_SYSTEM}\n\n{identity}"
    # 偏好 → 摘要。偏好是最稳定的约束，摘要是一次压缩事实，都排在原文窗口之前。
    if profile:
        system = f"{system}\n\n{profile}"
    if summary:
        system = f"{system}\n\n{summary}"
    messages = [SystemMessage(content=system)]
    for role, content in history:
        messages.append(
            AIMessage(content=content) if role == "assistant" else HumanMessage(content=content)
        )
    messages.append(HumanMessage(content=user_message))
    return {"messages": messages, "steps": 0, "tool_calls": [], "wallet_address": wallet_address}
