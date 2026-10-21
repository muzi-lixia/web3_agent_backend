"""中期记忆：把挤出窗口的消息压成一条结构化摘要。

两段式，中间不持锁也不持事务：
1. 事务一：读水位线、算出本次要压的区间、取出原文，然后提交；
2. 事务外：调模型（慢，不能占着连接和行锁）；
3. 事务二：按旧水位线写回，对不上就作废。

写回走 UPDATE，不走 upsert —— 清空接口刚删掉摘要行时，这里必须写不进去，
否则被清掉的内容会被后台任务复活。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from web3_agent.agent import langsmith_tracer
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.common.tracing import get_trace_id, set_run_id
from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import Settings
from web3_agent.domain.memory import schema
from web3_agent.domain.quotas import QuotaController
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import conversation_repo, summary_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger
from web3_agent.web3.metering import take_rpc_calls

logger = logging.getLogger(__name__)

# 摘要与记忆抽取的开销记在独立主体上：配额是防滥用，不该让用户为后台压缩撞 429。
INTERNAL_QUOTA_KEY = "internal|memory"


@dataclass
class _Job:
    previous: str
    expected_watermark: int | None
    until_id: int
    messages: list[tuple[str, str]]


class Summarizer:
    def __init__(
        self,
        settings: Settings,
        llm: LlmAdapter,
        audit: AuditLogger,
        quota: QuotaController,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._audit = audit
        self._quota = quota

    def should_run(self, *, pending: int, status: str | None, fail_count: int) -> bool:
        """纯判定，不碰库：挤出窗口的条数够多，且没触发失败熔断。"""
        if pending < self._settings.summary_trigger_min:
            return False
        if status in (None, "ok"):
            return True
        return fail_count < self._settings.summary_fail_max

    async def run(self, *, siwe_user_id: str, own_address: str) -> None:
        """后台任务入口。任何异常都不外抛——它已经不挂在任何请求上了。"""
        take_rpc_calls()  # create_task 会继承主请求的计数，不重置会串回去
        try:
            job = await self._collect(siwe_user_id)
            if job is None:
                return
            await self._summarize(siwe_user_id, own_address, job)
        except Exception:  # noqa: BLE001 后台任务不能把异常丢给事件循环
            logger.exception("summarize task crashed: user=%s", siwe_user_id)

    async def _collect(self, siwe_user_id: str) -> _Job | None:
        async with session_scope() as db:
            row = await summary_repo.get(db, siwe_user_id)
            since = row.last_summarized_message_id if row else 0
            status = row.status if row else None
            fail_count = row.fail_count if row else 0
            previous = row.content if row and row.content else ""

            pending, until_id = await conversation_repo.summarizable_range(
                db,
                siwe_user_id,
                window=self._settings.conversation_max_messages,
                since_id=since,
                cap=self._settings.summary_batch_cap,
            )
            if until_id is None or not self.should_run(
                pending=pending, status=status, fail_count=fail_count
            ):
                return None

            messages = await conversation_repo.load_range(
                db, siwe_user_id, after_id=since, until_id=until_id
            )
            return _Job(
                previous=previous,
                expected_watermark=since if row else None,
                until_id=until_id,
                messages=messages,
            )

    async def _summarize(self, siwe_user_id: str, own_address: str, job: _Job) -> None:
        # 后台任务是独立的 run，但 trace_id 跟着触发它的那轮请求走。
        run = langsmith_tracer.run_config(
            "summary.run", trace_id=get_trace_id(), until_id=job.until_id
        )
        if run:
            set_run_id(run["run_id"])
        try:
            raw = await self._llm.summarize(job.previous, job.messages, config=run or None)
        except Exception as exc:  # noqa: BLE001 模型侧任何失败都只降级、不重试
            await self._charge()
            await self._on_failure(siwe_user_id, exc, job)
            return

        await self._charge()
        fields, dropped = schema.sanitize(
            raw,
            own_address=own_address,
            max_chars=self._settings.summary_field_max_chars,
            max_items=self._settings.summary_field_max_items,
        )
        if dropped:
            await self._audit.log(
                "summary_items_dropped", siwe_user_id=siwe_user_id, dropped=dropped
            )
        if not any(getattr(fields, key) for key in schema.SUMMARY_KEYS):
            # 全被拒等于模型没产出可用内容，推进水位线会永久丢掉这段历史
            await self._on_failure(siwe_user_id, RuntimeError("摘要全部被白名单拒绝"), job)
            return

        async with session_scope() as db:
            written = await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=siwe_user_id,
                content=schema.dumps(fields),
                last_summarized_message_id=job.until_id,
                expected_watermark=job.expected_watermark,
            )
        await self._audit.log(
            "summary_written",
            siwe_user_id=siwe_user_id,
            until_id=job.until_id,
            written=written,
        )

    async def _on_failure(self, siwe_user_id: str, exc: Exception, job: _Job) -> None:
        async with session_scope() as db:
            await summary_repo.mark_failed(db, siwe_user_id)
        logger.warning("summarize failed: user=%s until=%s err=%s", siwe_user_id, job.until_id, exc)
        await self._audit.log("summary_failed", siwe_user_id=siwe_user_id, error=str(exc)[:200])

    async def _charge(self) -> None:
        async with session_scope() as db:
            await self._quota.charge(
                db, key=INTERNAL_QUOTA_KEY, metric=QuotaMetric.STEPS, amount=1
            )
