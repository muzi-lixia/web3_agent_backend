"""长期记忆第二段：自由文本抽取（V2）。

与摘要同构的两段式：事务取数 → 事务外调模型与 embedding → 事务写回。
水位线保证同一段消息只抽一次；积压时分批吃完，水位线单调推进。

和摘要的关键差别：**抽不出东西是正常结果**，不是失败 —— 大多数对话没有值得
长期记住的背景，这种情况水位线照常推进，否则每轮都会重抽同一段。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from web3_agent.agent import langsmith_tracer
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.common.tracing import get_trace_id, set_run_id
from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import Settings
from web3_agent.domain.memory import memory
from web3_agent.domain.memory.embedder import Embedder
from web3_agent.domain.memory.summarizer import INTERNAL_QUOTA_KEY
from web3_agent.domain.quotas import QuotaController
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import conversation_repo, memory_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger
from web3_agent.web3.metering import take_rpc_calls

logger = logging.getLogger(__name__)


@dataclass
class _Job:
    until_id: int
    messages: list[tuple[str, str]]
    existing_texts: list[str]


class MemoryExtractor:
    def __init__(
        self,
        settings: Settings,
        llm: LlmAdapter,
        embedder: Embedder,
        audit: AuditLogger,
        quota: QuotaController,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._embedder = embedder
        self._audit = audit
        self._quota = quota

    def should_run(self, *, pending: int, fail_count: int) -> bool:
        if pending < self._settings.memory_trigger_messages:
            return False
        return fail_count < self._settings.memory_fail_max

    async def run(self, *, siwe_user_id: str, job: _Job | None = None) -> None:
        take_rpc_calls()
        try:
            job = job or await self.collect(siwe_user_id)
            if job is None:
                return
            await self._extract(siwe_user_id, job)
        except Exception:  # noqa: BLE001 后台任务不能把异常丢给事件循环
            logger.exception("memory extract crashed: user=%s", siwe_user_id)

    async def collect(self, siwe_user_id: str) -> _Job | None:
        """探水位线并取出待抽取区间。返回 None 表示这轮没活干。"""
        async with session_scope() as db:
            state = await memory_repo.get_state(db, siwe_user_id)
            since = state.last_extracted_message_id if state else 0
            fail_count = state.fail_count if state else 0

            pending, rows = await conversation_repo.pending_for_memory(
                db,
                siwe_user_id,
                since_id=since,
                cap=self._settings.memory_batch_cap,
            )
            if not rows or not self.should_run(pending=pending, fail_count=fail_count):
                return None

            existing = await memory_repo.active_items(db, siwe_user_id)
            return _Job(
                until_id=rows[-1][0],
                messages=[(role, content) for _, role, content in rows],
                existing_texts=[row.text for row in existing],
            )

    async def _extract(self, siwe_user_id: str, job: _Job) -> None:
        run = langsmith_tracer.run_config(
            "memory.extract",
            trace_id=get_trace_id(),
            until_id=job.until_id,
            siwe_user_id=siwe_user_id,
        )
        if run:
            set_run_id(run["run_id"])
        try:
            raw = await self._llm.extract_memories(
                job.messages, job.existing_texts, config=run or None
            )
        except Exception as exc:  # noqa: BLE001
            await self._charge()
            await self._on_failure(siwe_user_id, exc, job)
            return

        items = _dedupe(raw.items, existing=job.existing_texts)
        await self._charge()

        if items:
            try:
                embeddings = await self._embedder.embed_many([item.text for item in items])
            except Exception as exc:  # noqa: BLE001 向量化失败时消息留存完整，下一轮重抽
                await self._on_failure(siwe_user_id, exc, job)
                return
            async with session_scope() as db:
                await memory_repo.supersede(db, siwe_user_id, [item.subject for item in items])
                await memory_repo.insert_many(db, siwe_user_id, items, embeddings)
                await memory_repo.advance_watermark(db, siwe_user_id, job.until_id)
            await self._audit.log(
                "memory_written",
                siwe_user_id=siwe_user_id,
                until_id=job.until_id,
                subjects=[item.subject for item in items],
            )
            return

        # 没抽到东西也要推进：否则这段对话会被反复重抽
        async with session_scope() as db:
            await memory_repo.advance_watermark(db, siwe_user_id, job.until_id)

    async def _on_failure(self, siwe_user_id: str, exc: Exception, job: _Job) -> None:
        async with session_scope() as db:
            await memory_repo.mark_failed(db, siwe_user_id)
        logger.warning("memory extract failed: user=%s until=%s err=%s", siwe_user_id, job.until_id, exc)
        await self._audit.log("memory_failed", siwe_user_id=siwe_user_id, error=str(exc)[:200])

    async def _charge(self) -> None:
        async with session_scope() as db:
            await self._quota.charge(
                db, key=INTERNAL_QUOTA_KEY, metric=QuotaMetric.STEPS, amount=1
            )


def _dedupe(drafts, *, existing: list[str]) -> list[memory.MemoryItem]:
    """白名单过滤 + 同 subject 只留一条 + 与已有一字不差的丢掉。"""
    seen: dict[str, memory.MemoryItem] = {}
    for draft in drafts:
        item = memory.sanitize(draft.model_dump() if hasattr(draft, "model_dump") else draft)
        if item is None or item.text in existing:
            continue
        current = seen.get(item.subject)
        if current is None or item.confidence > current.confidence:
            seen[item.subject] = item
    return list(seen.values())
