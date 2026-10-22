"""Agent 主循环：Perception → Thought → IntentRouter → Action → 终止。

写路径在这条链路上只走到 IntentRouter：route=write 就停，交给后续状态机（M2）。
V1a 不生成任何 TxIntent。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.agent import langsmith_tracer, secret_detector
from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.agent.intent_router import route_intent
from web3_agent.agent.json_guard import JsonGuard
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.agent.prompts import CHAT_SYSTEM
from web3_agent.agent.sanitize import sanitize_free_text
from web3_agent.common.exceptions import E507_MODEL_UNAVAILABLE, AppError, ModelUnavailable
from web3_agent.common.tracing import get_trace_id, set_run_id
from web3_agent.common.types import ChatOutcome, ToolCallRecord
from web3_agent.config.constants import (
    MSG_CLARIFY,
    MSG_JSON_DEGRADED,
    MSG_NO_DATA,
    MSG_OUT_OF_SCOPE,
    MSG_SECRET_REFUSED,
    MSG_STEP_LIMIT,
    MSG_WRITE_NOT_ENABLED,
    Intent,
    Route,
)
from web3_agent.config.settings import Settings
from web3_agent.domain.memory import memory, profile, schema
from web3_agent.domain.memory.embedder import Embedder
from web3_agent.persistence.conversation_context import ConversationContext
from web3_agent.persistence.pg.repositories import memory_repo, profile_repo, summary_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger


def _turn_inputs(params: dict[str, Any]) -> dict[str, Any]:
    """self 与 db 不进 trace：一个是整对象，一个是连接，序列化没有意义。"""
    return {
        "user_message": params.get("user_message"),
        "siwe_user_id": params.get("siwe_user_id"),
        "wallet_address": params.get("wallet_address"),
        "chain_id": params.get("chain_id"),
    }


def _turn_outputs(outcome: ChatOutcome) -> dict[str, Any]:
    return {
        "kind": outcome.kind,
        "content": outcome.content,
        "steps": outcome.steps,
        "degraded": outcome.degraded,
    }


# 这三个意图的答案必须是链上数据，不是模型自己写出来的话。
# chitchat 也走 read 路由，但它不需要数据，所以不在其中。
_DATA_INTENTS = frozenset({Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE})


def _lacks_evidence(intent: Intent, records: list[ToolCallRecord]) -> bool:
    """需要数据的意图，一条成功的工具调用都没有 —— 这时模型在无据作答。"""
    return intent in _DATA_INTENTS and not any(r.ok for r in records)


# 重跑图时替换掉模型那句空转回答：它没有数据，留着只会被抄一遍。
_RETRY_HINT = (
    "你刚才没有调用任何工具就给出了结果。请立即调用合适的工具获取真实数据后再回答；"
    "拿不到数据就直说拿不到，不要编造任何哈希、金额、时间或状态。"
)


class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        llm: LlmAdapter,
        guard: JsonGuard,
        conversation: ConversationContext,
        audit: AuditLogger,
        model_with_tools,
        runner,
        embedder: Embedder,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._guard = guard
        self._conversation = conversation
        self._audit = audit
        self._embedder = embedder
        self._graph = build_read_graph(
            model_with_tools, runner=runner, max_steps=settings.read_max_react_steps
        )

    async def handle(
        self,
        db: AsyncSession,
        *,
        siwe_user_id: str,
        wallet_address: str | None,
        chain_id: int | None,
        user_message: str,
    ) -> ChatOutcome:
        # 1. 密钥检测必须早于任何模型调用：命中就不外发、不落库。
        hit = secret_detector.detect(user_message)
        if hit is not None:
            await self._audit.log(
                "secret_leak_blocked",
                kind=hit.kind,
                excerpt=hit.excerpt,
                siwe_user_id=siwe_user_id,
            )
            return ChatOutcome(kind="reject", content=MSG_SECRET_REFUSED)

        sanitized = sanitize_free_text(user_message)
        if sanitized.hijacked:
            await self._audit.log("prompt_injection_filtered", siwe_user_id=siwe_user_id)

        # 密钥命中时上面已经返回，不会走到这里 —— 所以这一段之后才开 trace。
        extra = langsmith_tracer.root_extra(
            trace_id=get_trace_id(), chain_id=chain_id, siwe_user_id=siwe_user_id
        )
        if extra:
            set_run_id(extra["run_id"])
        return await self._turn(
            db,
            siwe_user_id=siwe_user_id,
            user_message=sanitized.text,
            wallet_address=wallet_address,
            chain_id=chain_id,
            langsmith_extra=extra,
        )

    @traceable(
        run_type="chain",
        name="chat.turn",
        process_inputs=_turn_inputs,
        process_outputs=_turn_outputs,
    )
    async def _turn(
        self,
        db: AsyncSession,
        *,
        siwe_user_id: str,
        wallet_address: str | None,
        chain_id: int | None,
        user_message: str,
    ) -> ChatOutcome:
        # 2. Perception：短期历史 + 中期摘要 + 长期档案与背景
        history = await self._conversation.load(
            db, siwe_user_id=siwe_user_id
        )
        summary_text = await self._load_summary(db, siwe_user_id)
        profile_text = await self._sync_profile(db, siwe_user_id, user_message, chain_id)
        # constraint 直接注入；软记忆要向量召回（一次 embed 2s 以上），
        # 留到意图确认之后 —— 闲聊轮用不上它。
        memory_rows = await memory_repo.active_items(db, siwe_user_id)
        memory_items = _constraints_of(memory_rows, self._settings.memory_constraint_max)
        memory_text = self._render_memories(memory_items)
        await self._conversation.save(
            db, siwe_user_id=siwe_user_id, role="user", content=user_message
        )

        # 3. Thought：结构化意图，带熔断
        try:
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
            return await self._finish(
                db, siwe_user_id, ChatOutcome(kind="answer", content=MSG_JSON_DEGRADED, degraded=True)
            )

        # 4. IntentRouter：唯一路由决策点
        decision = route_intent(
            guarded.output, confidence_threshold=self._settings.confidence_threshold
        )
        await self._audit.log(
            "intent_routed",
            siwe_user_id=siwe_user_id,
            intent=guarded.output.intent.value,
            confidence=guarded.output.confidence,
            route=decision.route.value,
            reason=decision.reason,
        )

        if decision.route is Route.CLARIFY:
            return await self._finish(
                db,
                siwe_user_id,
                ChatOutcome(
                    kind="ask",
                    content=MSG_CLARIFY,
                    degraded=False,
                ),
            )
        if decision.route is Route.REJECT:
            return await self._finish(
                db, siwe_user_id, ChatOutcome(kind="reject", content=MSG_OUT_OF_SCOPE)
            )
        if decision.route is Route.WRITE:
            return await self._finish(
                db, siwe_user_id, ChatOutcome(kind="answer", content=MSG_WRITE_NOT_ENABLED)
            )

        # 闲聊不进工具图：不带工具就不可能在闲聊里查一遍链，也省掉一次工具往返。
        if guarded.output.intent is Intent.CHITCHAT:
            return await self._finish(
                db,
                siwe_user_id,
                await self._chitchat(history, user_message, siwe_user_id, profile_text),
            )

        # 要查数据的轮次才值得为软记忆做一次向量召回。
        memory_items = [
            *memory_items,
            *await self._recall(db, siwe_user_id, user_message, memory_rows),
        ]
        memory_text = self._render_memories(memory_items)

        # 5. Action：只读 ReAct
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
                    slots=guarded.output.slots,
                ),
                config=graph_config,
            )
            records = [_record(item) for item in state.get("tool_calls", [])]
            retried = False
            # 只在「一条工具都没调」时重跑：调了但全失败说明是数据源问题，再问一遍也是白问。
            if not records and _lacks_evidence(guarded.output.intent, records):
                # 模型绕过工具直接作答：摘掉那句空转回答，明确要求先取数再答，重跑一次。
                retried = True
                retry_messages = [*state["messages"][:-1], HumanMessage(content=_RETRY_HINT)]
                state = await self._graph.ainvoke(
                    {**state, "messages": retry_messages},
                    config=graph_config,
                )
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
            answer = MSG_NO_DATA
            await self._audit.log(
                "answer_without_evidence",
                siwe_user_id=siwe_user_id,
                intent=guarded.output.intent.value,
                steps=steps,
                tool_calls=len(records),
                retried=retried,
            )

        outcome = ChatOutcome(
            kind="answer",
            content=answer if isinstance(answer, str) else str(answer),
            tool_calls=records,
            steps=steps,
            degraded=no_data,
        )
        return await self._finish(db, siwe_user_id, outcome)

    async def _chitchat(
        self,
        history: list[tuple[str, str]],
        user_message: str,
        siwe_user_id: str,
        profile_text: str = "",
    ) -> ChatOutcome:
        """不带工具的一次性回答。历史原样给，用户问的就是历史本身。"""
        messages = [SystemMessage(content=CHAT_SYSTEM)]
        if profile_text:
            messages.append(SystemMessage(content=profile_text))
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
        """摘要注入文本。status=failed 时也注入上一次成功的那份 —— 它仍然有效。"""
        row = await summary_repo.get(db, siwe_user_id)
        if row is None or not row.content:
            return ""
        return schema.render(
            schema.loads(row.content),
            max_chars=self._settings.summary_render_chars,
            updated_at=row.updated_at,
        )

    async def _sync_profile(
        self, db: AsyncSession, siwe_user_id: str, message: str, chain_id: int | None
    ) -> str:
        """吸收本轮的显式偏好，返回注入文本。

        先合并再渲染：用户说「以后都简洁点」的这一轮，回答就该是简洁的。
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
        return memory.render(items, max_chars=self._settings.memory_render_chars)

    async def _recall(
        self, db: AsyncSession, siwe_user_id: str, query: str, rows: list
    ) -> list[memory.MemoryItem]:
        """软记忆的向量召回。没有软记忆时连 embed 都不做。

        向量化是新增的失败点，坏了只降级到 constraint —— 记忆是增强项，
        不能因为它把整轮对话拖垮。
        """
        if not query.strip() or not any(r.memory_type != "constraint" for r in rows):
            return []
        try:
            vector = await self._embedder.embed(query)
            recalled = await memory_repo.recall(
                db,
                siwe_user_id,
                embedding=vector,
                limit=self._settings.memory_recall_limit,
                max_distance=self._settings.memory_recall_max_distance,
            )
        except ModelUnavailable as exc:
            await self._audit.log(
                "memory_recall_degraded", siwe_user_id=siwe_user_id, error=str(exc)[:200]
            )
            return []
        return [_as_item(r) for r in recalled]

    async def _finish(
        self, db: AsyncSession, siwe_user_id: str, outcome: ChatOutcome
    ) -> ChatOutcome:
        # 工具轨迹随消息落库：历史记录要能复原「这条回答查了什么」。
        # 没有调工具时留 NULL，不要写空数组 —— 绝大多数轮次都是这种。
        calls = [call.model_dump() for call in outcome.tool_calls]
        await self._conversation.save(
            db,
            siwe_user_id=siwe_user_id,
            role="assistant",
            content=outcome.content,
            tool_calls=calls or None,
        )
        return outcome


def _record(item: dict) -> ToolCallRecord:
    return ToolCallRecord(**item)


def _constraints_of(rows: list, cap: int) -> list[memory.MemoryItem]:
    """硬约束全量注入，不受向量召回影响。"""
    items = [_as_item(r) for r in rows if r.memory_type == "constraint"]
    return items[:cap]


def _as_item(row) -> memory.MemoryItem:
    return memory.MemoryItem(
        text=row.text,
        memory_type=row.memory_type,
        subject=row.subject,
        confidence=row.confidence,
    )
