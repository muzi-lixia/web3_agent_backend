"""Agent 主循环：Perception → Thought → IntentRouter → Action → 终止。

写路径在这条链路上只走到 IntentRouter：route=write 就停，交给后续状态机（M2）。
V1a 不生成任何 TxIntent。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.agent import secret_detector
from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.agent.intent_router import route_intent
from web3_agent.agent.json_guard import JsonGuard
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.agent.sanitize import sanitize_free_text
from web3_agent.common.exceptions import E507_MODEL_UNAVAILABLE, AppError, ModelUnavailable
from web3_agent.common.types import ChatOutcome
from web3_agent.config.constants import (
    MSG_CLARIFY,
    MSG_JSON_DEGRADED,
    MSG_OUT_OF_SCOPE,
    MSG_SECRET_REFUSED,
    MSG_STEP_LIMIT,
    MSG_WRITE_NOT_ENABLED,
    Route,
)
from web3_agent.config.settings import Settings
from web3_agent.domain.memory import memory, profile, schema
from web3_agent.domain.memory.embedder import Embedder
from web3_agent.persistence.conversation_context import ConversationContext
from web3_agent.persistence.pg.repositories import memory_repo, profile_repo, summary_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger


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

        # 2. Perception：短期历史 + 中期摘要 + 长期档案与背景
        history = await self._conversation.load(
            db, siwe_user_id=siwe_user_id
        )
        summary_text = await self._load_summary(db, siwe_user_id)
        profile_text = await self._sync_profile(db, siwe_user_id, sanitized.text, chain_id)
        memory_text = await self._load_memories(db, siwe_user_id, sanitized.text)
        await self._conversation.save(
            db, siwe_user_id=siwe_user_id, role="user", content=sanitized.text
        )

        # 3. Thought：结构化意图，带熔断
        try:
            guarded = await self._guard.run(
                lambda: self._llm.extract_intent(
                    history,
                    sanitized.text,
                    summary=summary_text,
                    profile=profile_text,
                    memories=memory_text,
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

        # 5. Action：只读 ReAct
        try:
            state = await self._graph.ainvoke(
                initial_state(
                    history,
                    sanitized.text,
                    wallet_address or "",
                    summary=summary_text,
                    profile=profile_text,
                    memories=memory_text,
                )
            )
        except ModelUnavailable as exc:
            await self._audit.log("model_unavailable", siwe_user_id=siwe_user_id, error=str(exc))
            code, msg, status = E507_MODEL_UNAVAILABLE
            raise AppError(code, msg, status, retryable=True) from exc
        answer = state["messages"][-1].content
        steps = state.get("steps", 0)
        if steps >= self._settings.read_max_react_steps:
            answer = f"{answer}\n\n{MSG_STEP_LIMIT}"

        outcome = ChatOutcome(
            kind="answer",
            content=answer if isinstance(answer, str) else str(answer),
            tool_calls=[_record(item) for item in state.get("tool_calls", [])],
            steps=steps,
        )
        return await self._finish(db, siwe_user_id, outcome)

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

    async def _load_memories(self, db: AsyncSession, siwe_user_id: str, query: str) -> str:
        """长期背景：constraint 全量注入，preference / profile 走向量召回。

        向量化是新增的失败点，坏了只降级到 constraint —— 记忆是增强项，
        不能因为它把整轮对话拖垮。
        """
        rows = await memory_repo.active_items(db, siwe_user_id)
        picked = [_as_item(r) for r in rows if r.memory_type == "constraint"]
        picked = picked[: self._settings.memory_constraint_max]

        if query.strip():
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
                recalled = []
            picked.extend(_as_item(r) for r in recalled)

        return memory.render(picked, max_chars=self._settings.memory_render_chars)

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


def _record(item: dict):
    from web3_agent.common.types import ToolCallRecord

    return ToolCallRecord(**item)


def _as_item(row) -> memory.MemoryItem:
    return memory.MemoryItem(
        text=row.text,
        memory_type=row.memory_type,
        subject=row.subject,
        confidence=row.confidence,
    )
