"""Agent 主循环：Perception → Thought → IntentRouter → Action → 终止。

写路径走到 IntentRouter 后单独出口：转账构造待签名交易并建一个待确认任务，
其余写意图仍提示未接入。本模块不签名、不广播。
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from langsmith import traceable
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.agent import langsmith_tracer, multi_hop, secret_detector, transfer_task
from web3_agent.agent.graph.plan_graph import (
    build_plan_graph,
    initial as plan_initial,
    opening as plan_opening,
)
from web3_agent.agent.graph.read_graph import build_read_graph, evidence_tools, initial_state
from web3_agent.agent.intent_router import route_intent
from web3_agent.agent.json_guard import JsonGuard
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.agent.prompts import CHAT_SYSTEM
from web3_agent.rag.retriever import NO_DOCUMENTS, UNAVAILABLE
from web3_agent.agent.sanitize import sanitize_free_text
from web3_agent.common.exceptions import (
    E507_MODEL_UNAVAILABLE,
    AppError,
    ModelUnavailable,
    error_message,
)
from web3_agent.common.tracing import get_trace_id, set_run_id
from web3_agent.common.types import ChatOutcome, ToolCallRecord
from web3_agent.config.constants import (
    MSG_AMOUNT_MISSING,
    MSG_CHAIN_REQUIRED,
    MSG_CLARIFY,
    MSG_JSON_DEGRADED,
    MSG_NO_DATA,
    MSG_OUT_OF_SCOPE,
    MSG_PLAN_UNAVAILABLE,
    MSG_SECRET_REFUSED,
    MSG_STEP_LIMIT,
    MSG_TRANSFER_CONFIRM,
    MSG_WALLET_REQUIRED,
    MSG_WRITE_NOT_ENABLED,
    MSG_WRITE_TASK_FAILED,
    REQUIRED_SLOTS,
    TOOL_POST_TRANSFER,
    CallSource,
    Intent,
    Route,
)
from web3_agent.config.settings import Settings
from web3_agent.domain.memory import memory, profile, schema
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import (
    conversation_repo,
    memory_repo,
    profile_repo,
)
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger

logger = logging.getLogger(__name__)

# StageFn：阶段上报回调。调用方据此让界面知道这一轮走到哪了，Agent 不关心怎么送达。
StageFn = Callable[[str], Awaitable[None]]


def _turn_inputs(params: dict[str, Any]) -> dict[str, Any]:
    """self 不进 trace：整对象序列化没有意义。

    Args:
        params: 根追踪接收到的方法参数字典。
    """
    return {
        "user_message": params.get("user_message"),
        "siwe_user_id": params.get("siwe_user_id"),
        "wallet_address": params.get("wallet_address"),
        "chain_id": params.get("chain_id"),
    }


def _turn_outputs(outcome: ChatOutcome) -> dict[str, Any]:
    """把内部回答转换为追踪输出，记录内容、步数和降级状态。

    Args:
        outcome: 本轮内部回答及工具轨迹。
    """
    return {
        "kind": outcome.kind,
        "content": outcome.content,
        "steps": outcome.steps,
        "degraded": outcome.degraded,
    }


def _lacks_evidence(intent: Intent, records: list[ToolCallRecord]) -> bool:
    """需要数据的意图，没有一条「该意图对应的」工具成功调用 —— 这时模型在无据作答。

    只认数据工具：问余额却调了知识检索（且成功），同样算没查到余额。

    Args:
        intent: 已由后端确认的本轮业务意图。
        records: 工具执行轨迹列表，供证据及错误检查。
    """
    tools = evidence_tools(intent)
    if not tools:
        return False
    return not any(r.ok and r.name in tools for r in records)


def _failure_notice(records: list[ToolCallRecord]) -> str:
    """全部工具都失败且错误码一致时，用错误码自己的文案。

    「稍后再试」对 E202（该合约暂不支持）这类错误是错的 —— 重试不会变好。
    错误码不一致（多源部分失败）时退回通用兜底。

    Args:
        records: 工具执行轨迹列表，供证据及错误检查。
    """
    codes = {r.error for r in records if not r.ok}
    if records and len(codes) == 1 and all(not r.ok for r in records):
        try:
            return error_message(codes.pop())
        except KeyError:
            return MSG_NO_DATA
    return MSG_NO_DATA


# 重跑图时替换掉模型那句空转回答：它没有数据，留着只会被抄一遍。
# _RETRY_HINT：模型未调用取证工具时的固定重试提示文本。
_RETRY_HINT = (
    "你刚才没有调用任何工具就给出了结果。请立即调用合适的工具获取真实数据后再回答；"
    "拿不到数据就直说拿不到，不要编造任何哈希、金额、时间或状态。"
)


# 工具参数里只留够复盘的分量：问题原文在消息 content 里已经存过一份。
# _PERSIST_QUERY_CHARS：工具轨迹持久化时保留问题文本的最大字符数。
_PERSIST_QUERY_CHARS = 200


def _trim_args(args: dict[str, Any]) -> dict[str, Any]:
    """去掉派生参数，截断与消息重复的长文本。

    Args:
        args: 工具输入参数字典。
    """
    kept = dict(args or {})
    kept.pop("conversation_context", None)
    query = kept.get("query_str")
    if isinstance(query, str) and len(query) > _PERSIST_QUERY_CHARS:
        kept["query_str"] = query[:_PERSIST_QUERY_CHARS]
    return kept


# 数据职责：单轮对话主编排器，负责上下文、路由、工具证据与结果持久化。
class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        llm: LlmAdapter,
        guard: JsonGuard,
        audit: AuditLogger,
        model_with_tools,
        runner,
        checkpointer=None,
    ) -> None:
        """初始化 AgentLoop 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
            llm: 统一模型适配器，负责结构化抽取及回答。
            guard: 结构化输出有限重试保护器。
            audit: 审计日志写入服务。
            model_with_tools: 已绑定只读工具的回答模型。
            runner: 工具统一执行器，负责来源检查和调用轨迹。
            checkpointer: 多步计划的检查点存储；不可用时多步输入降级为追问。
        """
        self._settings = settings
        self._llm = llm
        self._guard = guard
        self._audit = audit
        # _runner：写路径与读路径共用同一个执行入口，来源校验只在一个地方。
        self._runner = runner
        # _graph：初始化时编译的只读 LangGraph，供每轮异步执行。
        self._graph = build_read_graph(
            model_with_tools,
            runner=runner,
            max_steps=settings.read_max_react_steps,
            rag_answer_model=llm.answer_model
        )
        # _plan_graph：多步计划图。检查点不可用时为空，多步输入转追问而不是硬走单步。
        self._plan_graph = (
            build_plan_graph(runner, checkpointer) if checkpointer is not None else None
        )

    async def handle(
        self,
        *,
        siwe_user_id: str,
        wallet_address: str | None,
        chain_id: int | None,
        user_message: str,
        knowledge_base_id: str = "default",
        on_stage: StageFn | None = None,
    ) -> ChatOutcome:
        # 1. 密钥检测必须早于任何模型调用：命中就不外发、不落库。
        """处理单轮输入：先拦截密钥和清洗文本，再建立追踪并执行 Agent 主流程。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
            wallet_address: 服务端确认的当前钱包地址。
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            user_message: 本轮待处理的用户输入。
            knowledge_base_id: 本轮选定的知识库标识。
            on_stage: 阶段回调；为空时不产生进度事件。

        Returns:
            ChatOutcome: 本轮回答、工具轨迹与降级或失效标志。
        """
        hit = secret_detector.detect(user_message)
        if hit is not None:
            await self._audit.log(
                "secret_leak_blocked",
                kind=hit.kind,
                excerpt=hit.excerpt,
                siwe_user_id=siwe_user_id,
            )
            return ChatOutcome(kind="reject", content=MSG_SECRET_REFUSED)

        # 只清洗自由文本，保留结构化钱包和网络参数各自的校验边界。
        sanitized = sanitize_free_text(user_message, max_len=4000)
        if sanitized.hijacked:
            await self._audit.log("prompt_injection_filtered", siwe_user_id=siwe_user_id)

        # 密钥命中时上面已经返回，不会走到这里 —— 所以这一段之后才开 trace。
        extra = langsmith_tracer.root_extra(
            trace_id=get_trace_id(), chain_id=chain_id, siwe_user_id=siwe_user_id
        )
        if extra:
            set_run_id(extra["run_id"])
        return await self._turn(
            siwe_user_id=siwe_user_id,
            user_message=sanitized.text,
            knowledge_base_id=knowledge_base_id,
            wallet_address=wallet_address,
            chain_id=chain_id,
            langsmith_extra=extra,
            on_stage=on_stage,
        )

    @traceable(
        run_type="chain",
        name="chat.turn",
        process_inputs=_turn_inputs,
        process_outputs=_turn_outputs,
    )
    async def _turn(
        self,
        *,
        siwe_user_id: str,
        wallet_address: str | None,
        chain_id: int | None,
        user_message: str,
        knowledge_base_id: str = "default",
        on_stage: StageFn | None = None,
    ) -> ChatOutcome:
        # 2. Perception：短期历史 + 中期摘要 + 长期档案与背景。
        # 这一段全是短查询加一条写入，事务在模型调用之前就关掉 ——
        # 一轮对话要十几秒，连接不能跟着一起挂到回答生成完。
        # 同一个 turn_id 写在 user 和 assistant 两条消息上：并发轮次的 id 会交错，
        # 加载历史时靠它重建「问题在前、回答在后」。
        """执行一轮对话：加载上下文、识别意图、路由、调用工具并持久化回答。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
            wallet_address: 服务端确认的当前钱包地址。
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            user_message: 本轮待处理的用户输入。
            knowledge_base_id: 本轮选定的知识库标识。
            on_stage: 阶段回调；为空时不产生进度事件。
        """
        async def stage(name: str) -> None:
            """上报当前阶段。送达失败只记日志：进度不是回答的一部分。

            Args:
                name: 阶段标识，由调用方解释成展示文案。
            """
            if on_stage is None:
                return
            try:
                await on_stage(name)
            except Exception:  # noqa: BLE001
                logger.warning("stage emit failed stage=%s", name, exc_info=True)

        await stage("intent")
        turn_id = uuid.uuid4().hex
        async with session_scope() as db:
            # 读生命周期版本，并在本事务内锁住它：清空接口要拿排他锁，
            # 两边因此串行 —— 不会出现「读到旧历史、清空、再把答案写回空会话」。
            generation = await conversation_repo.lock_generation(db, siwe_user_id)
            history = await conversation_repo.load_history(
                db,
                siwe_user_id,
                self._settings.conversation_max_messages,
                self._settings.conversation_max_chars,
            )
            summary_text = await self._load_summary(db, siwe_user_id)
            profile_text = await self._sync_profile(db, siwe_user_id, user_message, chain_id)
            memory_rows = await memory_repo.active_items(db, siwe_user_id)
            await conversation_repo.append_message(
                db,
                siwe_user_id=siwe_user_id,
                role="user",
                content=user_message,
                turn_id=turn_id,
            )
        # 意图抽取只吃硬约束：它决定「什么不能做」，软背景对判断意图没有帮助，
        # 留到执行轮再注入。
        memory_items = _constraints_of(memory_rows, self._settings.memory_constraint_max)
        memory_text = self._render_memories(memory_items)

        async def finish(outcome: ChatOutcome) -> ChatOutcome:
            """本轮出口。每一条返回路径都要落库，收成一个口子就不会有人漏写。

            Args:
                outcome: 本轮内部回答及工具轨迹。
            """
            return await self._finish(
                siwe_user_id, outcome, generation=generation, turn_id=turn_id
            )

        # 3. Thought：结构化意图，带熔断
        try:
            # 结构化意图解析只对格式失败有限重试，模型服务故障继续交给外层处理。
            guarded = await self._guard.run(
                lambda: self._llm.extract_intent(
                    history,
                    user_message,
                    summary=summary_text,
                    profile=profile_text,
                    memories=memory_text,
                    config=langsmith_tracer.run_config(
                        "intent.extract",
                        trace_id=get_trace_id(),
                        siwe_user_id=siwe_user_id,
                    ),
                )
            )
        except ModelUnavailable as exc:
            await self._audit.log("model_unavailable", siwe_user_id=siwe_user_id, error=str(exc))
            code, msg, status = E507_MODEL_UNAVAILABLE
            raise AppError(code, msg, status, retryable=True) from exc
        if guarded.degraded or guarded.output is None:
            await self._audit.log(
                "llm_json_degraded", siwe_user_id=siwe_user_id, attempts=guarded.attempts
            )
            return await finish(
                ChatOutcome(kind="answer", content=MSG_JSON_DEGRADED, degraded=True)
            )

        # 4. IntentRouter：唯一路由决策点
        decision = route_intent(guarded.output, user_message=user_message)
        await self._audit.log(
            "intent_routed",
            siwe_user_id=siwe_user_id,
            intent=guarded.output.intent.value,
            confidence=guarded.output.confidence,
            needs_clarification=guarded.output.needs_clarification,
            route=decision.route.value,
            reason=decision.reason,
        )

        if decision.route is Route.CLARIFY:
            return await finish(ChatOutcome(kind="ask", content=MSG_CLARIFY))
        if decision.route is Route.REJECT:
            return await finish(ChatOutcome(kind="reject", content=MSG_OUT_OF_SCOPE))
        # 多步计划先于路由分派：这类输入里既有查询又有写操作，模型给的 intent 可能
        # 只落在其中一半上，按单步路由执行会把另一半静默丢掉。
        if multi_hop.needs_plan(guarded.output.hops):
            await stage("plan")
            return await finish(
                await self._plan_outcome(
                    guarded.output.hops,
                    siwe_user_id=siwe_user_id,
                    wallet_address=wallet_address,
                    chain_id=chain_id,
                    user_message=user_message,
                )
            )
        if decision.route is Route.WRITE:
            await stage("write")
            return await finish(
                await self._write_outcome(
                    guarded.output.intent,
                    guarded.output.slots,
                    siwe_user_id=siwe_user_id,
                    wallet_address=wallet_address,
                    chain_id=chain_id,
                    user_message=user_message,
                )
            )

        if guarded.output.intent is Intent.QUERY_KNOWLEDGE and not self._settings.rag_enabled:
            return await finish(ChatOutcome(kind="answer", content=UNAVAILABLE, degraded=True))
        # 闲聊不进工具图：不带工具就不可能在闲聊里查一遍链，也省掉一次工具往返。
        if guarded.output.intent is Intent.CHITCHAT:
            return await finish(
                await self._chitchat(
                    history,
                    user_message,
                    siwe_user_id,
                    profile_text=profile_text,
                    memory_text=memory_text,
                    summary_text=summary_text,
                )
            )

        # 软记忆留到意图确认之后注入：闲聊轮用不上它，而补充背景只对执行轮有意义。
        # 知识问答那轮同样不需要 —— 它只读本轮检索到的材料。
        if guarded.output.intent is not Intent.QUERY_KNOWLEDGE:
            memory_items = [
                *memory_items,
                *_soft_memories(memory_rows, self._settings.memory_recall_limit),
            ]
        memory_text = self._render_memories(memory_items)

        # 5. Action：只读 ReAct
        await stage("query")
        graph_config = (
            langsmith_tracer.run_config("agent.graph", trace_id=get_trace_id()) or None
        )
        try:
            state = await self._graph.ainvoke(
                initial_state(
                    history,
                    user_message,
                    wallet_address or "",
                    summary=summary_text,
                    profile=profile_text,
                    memories=memory_text,
                    intent=guarded.output.intent,
                    kb_id=knowledge_base_id,
                ),
                config=graph_config,
            )
            # 将图状态恢复为统一轨迹对象，后续只依据成功且相关的工具判断回答有无证据。
            records = [_record(item) for item in state.get("tool_calls", [])]
            retried = False
            # 只在「一条工具都没调」时重跑：调了但全失败说明是数据源问题，再问一遍也是白问。
            if (
                guarded.output.intent is not Intent.QUERY_KNOWLEDGE
                and not records
                and _lacks_evidence(guarded.output.intent, records)
            ):
                # 模型绕过工具直接作答：摘掉那句空转回答，明确要求先取数再答，重跑一次。
                retried = True
                retry_messages = [*state["messages"][:-1], HumanMessage(content=_RETRY_HINT)]
                state = await self._graph.ainvoke(
                    {**state, "messages": retry_messages},
                    config=graph_config,
                )
                # 将图状态恢复为统一轨迹对象，后续只依据成功且相关的工具判断回答有无证据。
                records = [_record(item) for item in state.get("tool_calls", [])]
        except ModelUnavailable as exc:
            await self._audit.log("model_unavailable", siwe_user_id=siwe_user_id, error=str(exc))
            code, msg, status = E507_MODEL_UNAVAILABLE
            raise AppError(code, msg, status, retryable=True) from exc

        answer = state.get("direct_answer") or state["messages"][-1].content
        steps = state.get("steps", 0)
        if steps >= self._settings.read_max_react_steps:
            answer = f"{answer}\n\n{MSG_STEP_LIMIT}"

        no_data = _lacks_evidence(guarded.output.intent, records)
        if state.get("direct_answer"):
            await self._audit.log(
                "answer_direct",
                siwe_user_id=siwe_user_id,
                intent=guarded.output.intent.value,
                tool_calls=len(records),
            )
        if retried and not no_data:
            await self._audit.log(
                "tool_call_retried",
                siwe_user_id=siwe_user_id,
                intent=guarded.output.intent.value,
                steps=steps,
            )
        if no_data:
            # 整段替换而不是追加：模型写的是带哈希和金额的一整段，追加等于把假数据留在里面。
            answer = _failure_notice(records)
            await self._audit.log(
                "answer_without_evidence",
                siwe_user_id=siwe_user_id,
                intent=guarded.output.intent.value,
                steps=steps,
                tool_calls=len(records),
                retried=retried,
            )

        # 检索与数据工具平行：任何一轮都可能顺带查了文档，查到就把引用一起给出去。
        citations = list(state.get("rag_citations", []))
        if guarded.output.intent is Intent.QUERY_KNOWLEDGE:
            if no_data or not citations:
                status = state.get("rag_status", "")
                if not status:
                    answer = "本次未完成知识库检索，暂时无法确认，请稍后重试。"
                elif status == "empty":
                    answer = NO_DOCUMENTS
                else:
                    answer = UNAVAILABLE
                no_data = True
                citations = []
            else:
                answer = str(answer).strip()
        if no_data:
            # 结论已经换成「查不到」，再挂着引用会让人以为那是依据。
            citations = []
        outcome = ChatOutcome(
            kind="answer",
            content=answer if isinstance(answer, str) else str(answer),
            tool_calls=records,
            steps=steps,
            degraded=no_data,
            citations=citations,
        )
        return await finish(outcome)

    async def _plan_outcome(
        self,
        hops,
        *,
        siwe_user_id: str,
        wallet_address: str | None,
        chain_id: int | None,
        user_message: str,
    ) -> ChatOutcome:
        """执行多步计划，走到写操作就挂起等确认。

        Args:
            hops: 模型给出的执行计划。
            siwe_user_id: 服务端确认的登录用户标识，作为任务归属。
            wallet_address: 服务端确认的当前钱包地址。
            chain_id: 交易所在网络编号，由会话注入。
            user_message: 用户本轮原始输入，用于核对金额是否出自原话。
        """
        graph = self._plan_graph
        if graph is None:
            # 没有检查点就存不下挂起位置；带条件的计划丢掉条件等于改指令。
            if multi_hop.has_condition(hops):
                return ChatOutcome(kind="ask", content=MSG_PLAN_UNAVAILABLE)
            # 降级同样要守住执行器的能力边界：多笔写不能只办第一笔。
            shape = multi_hop.plan_shape_error(hops)
            if shape:
                return ChatOutcome(kind="ask", content=shape)
            write = next(hop for hop in hops if hop.kind == "write")
            if write.tool != TOOL_POST_TRANSFER:
                return ChatOutcome(kind="answer", content=MSG_WRITE_NOT_ENABLED)
            # 先查后转没有条件可丢，退回单步：少回答一句，不会动错钱。
            return await self._write_outcome(
                Intent.TRANSFER,
                dict(write.args),
                siwe_user_id=siwe_user_id,
                wallet_address=wallet_address,
                chain_id=chain_id,
                user_message=user_message,
            )
        if not wallet_address:
            return ChatOutcome(kind="ask", content=MSG_WALLET_REQUIRED)
        if chain_id is None:
            return ChatOutcome(kind="ask", content=MSG_CHAIN_REQUIRED)

        thread_id = uuid.uuid4().hex
        try:
            state = await graph.ainvoke(
                plan_initial(
                    hops=hops,
                    user_message=user_message,
                    siwe_user_id=siwe_user_id,
                    wallet_address=wallet_address,
                    chain_id=chain_id,
                    thread_id=thread_id,
                    ttl_seconds=self._settings.workflow_ttl_seconds,
                ),
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception:  # noqa: BLE001 计划执行失败不能把整轮对话打成 500
            logger.exception("plan graph failed thread=%s", thread_id)
            await self._audit.log(
                "plan_failed", siwe_user_id=siwe_user_id, thread_id=thread_id
            )
            return ChatOutcome(kind="answer", content=MSG_NO_DATA, degraded=True)
        calls = [_record(item) for item in state.get("tool_calls", [])]
        if state.get("kind"):
            return ChatOutcome(
                kind=state["kind"], content=state.get("content", ""), tool_calls=calls
            )

        preview = state.get("preview") or {}
        return ChatOutcome(
            kind="intent_confirm",
            content=plan_opening(
                state.get("verdicts", []),
                MSG_TRANSFER_CONFIRM.format(
                    amount=preview.get("humanAmount", ""),
                    symbol=preview.get("tokenSymbol", ""),
                    target=preview.get("target", ""),
                ),
            ),
            intent_preview=preview,
            task_id=state.get("task_id") or None,
            tool_calls=calls,
        )

    async def resume_plan(self, thread_id: str) -> list[dict] | None:
        """让挂起的计划走到终点，并清掉它的检查点。

        调用方是确认接口：它已经推进过状态机，图只负责从挂起处收尾，不再碰状态。
        已经走完的计划不重复恢复 —— 重复点击确认是常态。

        Args:
            thread_id: 计划的检查点标识。

        Returns:
            挂起时算出的判断依据（持有多少、比的是哪个门槛）；已经恢复过或没有计划时为空。
        """
        graph = self._plan_graph
        if graph is None or not thread_id:
            return None
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await graph.aget_state(config)
        except Exception:  # noqa: BLE001 读不到就当作没有挂起
            await self._audit.log("plan_resume_failed", thread_id=thread_id)
            logger.exception("plan resume failed thread=%s", thread_id)
            return None
        if snapshot is None or not snapshot.next:
            return None
        values = snapshot.values or {}
        try:
            # 恢复值只是「挂起已结束」的标记，用户的选择已经落在状态机里。
            await graph.ainvoke(Command(resume=True), config=config)
        except Exception:  # noqa: BLE001 图恢复失败不能影响已经落地的确认结果
            await self._audit.log(
                "plan_resume_failed",
                siwe_user_id=values.get("siwe_user_id"),
                siwe_address=values.get("wallet_address"),
                task_id=values.get("task_id"),
                thread_id=thread_id,
            )
            logger.exception("plan resume failed thread=%s", thread_id)
            return None
        await self._drop_thread(graph, thread_id)
        return values.get("verdicts") or None

    async def drop_plan_thread(self, thread_id: str) -> None:
        """删除一个不会再被恢复的计划检查点。

        用户始终没有确认时，任务由兜底扫描判过期，挂起的图没人再唤醒；
        由扫描方调用这里清掉，否则检查点表只增不减。

        Args:
            thread_id: 计划的检查点标识。
        """
        graph = self._plan_graph
        if graph is None or not thread_id:
            return
        await self._drop_thread(graph, thread_id)

    @staticmethod
    async def _drop_thread(graph, thread_id: str) -> None:
        """删除计划的检查点，避免挂起记录随对话无限累积。

        Args:
            graph: 已编译的计划图。
            thread_id: 要清理的检查点标识。
        """
        saver = getattr(graph, "checkpointer", None)
        if saver is None:
            return
        try:
            await saver.adelete_thread(thread_id)
        except Exception:  # noqa: BLE001 清理失败只留下一条记录，不影响本次确认
            logger.warning("plan checkpoint cleanup failed thread=%s", thread_id)

    async def _write_outcome(
        self,
        intent: Intent,
        slots: dict[str, Any],
        *,
        siwe_user_id: str,
        wallet_address: str | None,
        chain_id: int | None,
        user_message: str,
    ) -> ChatOutcome:
        """校验写操作参数，通过则建任务并回显待确认视图，不通过则追问补参数。

        Args:
            intent: 已通过必填槽位校验的写意图。
            slots: 模型抽取到的业务参数。
            siwe_user_id: 服务端确认的登录用户标识，作为任务归属。
            wallet_address: 服务端确认的当前钱包地址。
            chain_id: 交易所在网络编号，由会话注入。
            user_message: 用户本轮原始输入，用于核对金额是否出自原话。
        """
        # 除转账外的写意图还没有构造能力，保持原有的未接入提示。
        if intent is not Intent.TRANSFER:
            return ChatOutcome(kind="answer", content=MSG_WRITE_NOT_ENABLED)
        if not wallet_address:
            return ChatOutcome(kind="ask", content=MSG_WALLET_REQUIRED)
        if chain_id is None:
            return ChatOutcome(kind="ask", content=MSG_CHAIN_REQUIRED)
        # 多步计划降级到单步时参数来自写步，不一定齐全；缺了就不能往下取下标。
        if any(
            not str(slots.get(key, "")).strip()
            for key in REQUIRED_SLOTS[Intent.TRANSFER]
        ):
            return ChatOutcome(kind="ask", content=MSG_CLARIFY)
        # 模型会把「转点」「一半」这类说法补成一个具体数字，那张卡片上的金额
        # 用户从没说过。核对不上就追问，不拿推测值去构造交易。
        if not _amount_quoted(
            str(slots["humanAmount"]), user_message, str(slots.get("humanAmountText") or "")
        ):
            return ChatOutcome(kind="ask", content=MSG_AMOUNT_MISSING)

        # 参数校验交给构造工具：地址格式、金额精度、代币归属都在那里判定，
        # 能拼出待签名交易才允许进入确认，避免回显一张执行不了的卡片。
        text, record = await self._runner.run(
            TOOL_POST_TRANSFER,
            {
                "from_address": wallet_address,
                "to_address": str(slots["toAddress"]),
                "token_symbol": str(slots["tokenSymbol"]),
                "human_amount": str(slots["humanAmount"]),
                "chain_id": chain_id,
            },
            source=CallSource.STATE_MACHINE,
        )
        payload = json.loads(text)
        if payload.get("status") != "ready":
            return ChatOutcome(
                kind="ask",
                content=str(payload.get("message") or error_message("E304")),
                tool_calls=[record],
            )

        try:
            task_id, preview = await transfer_task.create(
                payload,
                siwe_user_id=siwe_user_id,
                wallet_address=wallet_address,
                chain_id=chain_id,
                ttl_seconds=self._settings.workflow_ttl_seconds,
            )
        except Exception:  # noqa: BLE001 建不出任务就不能给卡片：前端没有可确认的对象
            logger.exception("workflow create failed intent=%s", intent.value)
            return ChatOutcome(
                kind="answer", content=MSG_WRITE_TASK_FAILED, tool_calls=[record]
            )
        return ChatOutcome(
            kind="intent_confirm",
            content=MSG_TRANSFER_CONFIRM.format(
                amount=preview["humanAmount"],
                symbol=preview["tokenSymbol"],
                target=preview["target"],
            ),
            intent_preview=preview,
            task_id=task_id,
            tool_calls=[record],
        )

    async def _chitchat(
        self,
        history: list[tuple[str, str]],
        user_message: str,
        siwe_user_id: str,
        profile_text: str = "",
        memory_text: str = "",
        summary_text: str = "",
    ) -> ChatOutcome:
        """不带工具的一次性回答。

        摘要和长期约束要一起给：用户可能问的是已经滚出近期窗口的事，
        只给 recent history，模型就只能说不知道 —— 而意图识别那边是看得到摘要的。

        Args:
            history: 按角色组织的近期对话列表。
            user_message: 本轮待处理的用户输入。
            siwe_user_id: 数据所属用户的稳定标识。
            profile_text: 已渲染的用户结构化偏好文本。
            memory_text: 已渲染的用户长期背景文本。
            summary_text: 已渲染的历史摘要文本。
        """
        # 先建立聊天能力边界，再追加当前可用的偏好、记忆和历史。
        messages = [SystemMessage(content=CHAT_SYSTEM)]
        if profile_text:
            messages.append(SystemMessage(content=profile_text))
        if summary_text:
            messages.append(SystemMessage(content=summary_text))
        if memory_text:
            # 与只读链路同一形态：用户背景单独成块、带标签，不混进规则。
            messages.append(
                SystemMessage(content=f"<user_background>\n{memory_text}\n</user_background>")
            )
        for role, content in history:
            messages.append(
                AIMessage(content=content) if role == "assistant" else HumanMessage(content=content)
            )
        messages.append(HumanMessage(content=user_message))
        try:
            reply = await self._llm.answer(
                messages,
                config=langsmith_tracer.run_config(
                    "chat.chitchat",
                    trace_id=get_trace_id(),
                    siwe_user_id=siwe_user_id,
                ),
            )
        except ModelUnavailable as exc:
            await self._audit.log("model_unavailable", siwe_user_id=siwe_user_id, error=str(exc))
            code, msg, status = E507_MODEL_UNAVAILABLE
            raise AppError(code, msg, status, retryable=True) from exc
        return ChatOutcome(kind="answer", content=reply.content)

    async def _load_summary(self, db: AsyncSession, siwe_user_id: str) -> str:
        """摘要注入文本。摘要行由后台任务维护，加载时只读最新一条。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            siwe_user_id: 数据所属用户的稳定标识。
        """
        # 摘要行自身 ID 是压缩边界；下一批只读取这个边界之后的旧消息。
        row = await conversation_repo.latest_summary(db, siwe_user_id)
        if row is None or not row.content:
            return ""
        return schema.render(
            schema.loads(row.content),
            max_chars=self._settings.summary_render_chars,
            updated_at=row.created_at,
        )

    async def _sync_profile(
        self, db: AsyncSession, siwe_user_id: str, message: str, chain_id: int | None
    ) -> str:
        """吸收本轮的显式偏好，返回注入文本。

        先合并再渲染：用户说「以后都简洁点」的这一轮，回答就该是简洁的。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            siwe_user_id: 数据所属用户的稳定标识。
            message: 按当前函数用途传入的用户文本或已解析 SIWE 消息。
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
        """
        patch = profile.extract(message, chain_id=chain_id)
        current = await profile_repo.apply_patch(
            db, siwe_user_id=siwe_user_id, patch=patch, chain_id=chain_id
        )
        if not patch.is_empty:
            await self._audit.log(
                "profile_updated",
                siwe_user_id=siwe_user_id,
                answer_style=patch.answer_style,
                language=patch.language,
                preferred_chain_id=patch.preferred_chain_id,
                risk_preference=patch.risk_preference,
                tokens=list(patch.tokens),
            )
        return profile.render(
            current,
            max_chars=self._settings.profile_render_chars,
            max_tokens=self._settings.profile_token_max_items,
        )

    def _render_memories(self, items: list[memory.MemoryItem]) -> str:
        """按字符预算渲染本轮选中的长期记忆。

        Args:
            items: 本批记忆或字段条目。
        """
        return memory.render(items, max_chars=self._settings.memory_render_chars)

    async def _finish(
        self,
        siwe_user_id: str,
        outcome: ChatOutcome,
        *,
        generation: int,
        turn_id: str,
    ) -> ChatOutcome:
        # 工具轨迹随消息落库：历史记录要能复原「这条回答查了什么」。
        # 没有调工具时留 NULL，不要写空数组 —— 绝大多数轮次都是这种。
        # 收一遍参数，落库和响应都用这个形态，不各留一份全文。
        """裁剪工具轨迹，校验会话版本后保存助手回答；过期轮次返回丢弃标志。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
            outcome: 本轮内部回答及工具轨迹。
            generation: 本轮读取历史时的会话版本，写回前必须与数据库当前版本一致。
            turn_id: 同一轮用户问题与助手回答共用的标识。

        Returns:
            ChatOutcome: 已持久化结果；会话失效时返回正文为空且 discarded 为真的结果。
        """
        for call in outcome.tool_calls:
            call.args = _trim_args(call.args)
        # 保存工具轨迹快照，刷新历史后仍能看到本轮使用的数据来源。
        calls = [call.model_dump() for call in outcome.tool_calls]
        # Keep citation snapshots with the existing JSON tool metadata, without DDL.
        # This preserves references after the frontend reloads conversation history.
        for call in calls:
            if call["name"] == "rag_general_knowledge" and call["ok"] and outcome.citations:
                call["citations"] = outcome.citations
                break
        async with session_scope() as db:
            # 版本校验与写入必须同事务：这一轮读了旧历史来生成答案，
            # 期间用户可能已经清空对话。此时答案是「基于已删除内容」的，
            # 写回去等于把刚删的会话复活成一条无头回答。
            if not await conversation_repo.assert_generation(db, siwe_user_id, generation):
                await self._audit.log(
                    "turn_discarded_stale",
                    siwe_user_id=siwe_user_id,
                    turn_id=turn_id,
                    kind=outcome.kind,
                )
                # 不落库只解决了一半：这段正文来自用户刚删掉的历史，交回去等于
                # 清了又冒出来。把它清空并打上标记，由调用方决定怎么收尾。
                return outcome.model_copy(update={
                    "content": "",
                    "discarded": True,
                    "tool_calls": [],
                    "citations": [],
                    "intent_preview": None,
                    "task_id": None,
                    "steps": 0,
                    "degraded": False,
                })
            await conversation_repo.append_message(
                db,
                siwe_user_id=siwe_user_id,
                role="assistant",
                content=outcome.content,
                tool_calls=calls or None,
                turn_id=turn_id,
                # 确认卡片的两个凭据一起落库，刷新后前端才能按 taskId 接着确认。
                task_id=outcome.task_id,
                intent_preview=outcome.intent_preview,
            )
        return outcome


def _amount_quoted(human_amount: str, message: str, quote: str = "") -> bool:
    """判断待确认的金额是否出自用户原话。

    Args:
        human_amount: 模型抽取的转账数量。
        message: 用户本轮原始输入。
        quote: 模型给出的、原话里表示该金额的片段。
    """
    return multi_hop.amount_is_quoted(human_amount, message, quote=quote)


def _record(item: dict) -> ToolCallRecord:
    """将图中的工具调用字典恢复为 ToolCallRecord 领域对象。

    Args:
        item: 图状态中的单条工具调用字典。
    """
    return ToolCallRecord(**item)


def _constraints_of(rows: list, cap: int) -> list[memory.MemoryItem]:
    """硬约束全量注入：它决定「什么不能做」，不该被条数筛掉。

    Args:
        rows: 已读取的数据库记录列表。
        cap: 本批处理或选取的最大条数。
    """
    items = [_as_item(r) for r in rows if r.memory_type == "constraint"]
    return items[:cap]


def _soft_memories(rows: list, cap: int) -> list[memory.MemoryItem]:
    """软记忆按写入时间倒序取前几条。

    rows 已经是 created_at 倒序 —— 最近提到的就是当下最相关的，够用。
    不做相似度筛选：active 集被 subject 唯一索引压到「主题数」量级，
    为这点候选集付一次 query embedding 不划算。

    Args:
        rows: 已读取的数据库记录列表。
        cap: 本批处理或选取的最大条数。
    """
    return [_as_item(r) for r in rows if r.memory_type != "constraint"][:cap]


def _as_item(row) -> memory.MemoryItem:
    """将数据库记忆记录转换为注入提示词使用的 MemoryItem。

    Args:
        row: 单条 ORM 数据记录。
    """
    return memory.MemoryItem(
        text=row.text,
        memory_type=row.memory_type,
        subject=row.subject,
        confidence=row.confidence,
    )
