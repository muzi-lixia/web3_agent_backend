"""中期记忆：把挤出窗口的消息压成一条结构化摘要。

摘要就是 `conversation_message` 里 `role='summary'` 的一行，任何时刻只有一条。
它的 id 天然是水位线 —— 比它新的消息还没被压缩，不用另存一个字段。

两段式，中间不持锁也不持事务：
1. 事务一：读版本、读当前摘要行、算出本次要压的区间、取出原文，然后提交；
2. 事务外：调模型（慢，不能占着连接和行锁）；
3. 事务二：删旧行、插新行，按「删除的条数与读取时的预期是否一致」判定这份结果还有效。

写回不 UPDATE：清空接口刚删掉摘要行时，UPDATE 影响 0 行自然写不进去；
而删旧插新让新行拿到更大的 id，水位线跟着前进。
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
from web3_agent.persistence.pg.repositories import conversation_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger
from web3_agent.common.metering import start_metering

logger = logging.getLogger(__name__)

# 摘要与记忆抽取的开销记在独立主体上：配额是防滥用，不该让用户为后台压缩撞 429。
INTERNAL_QUOTA_KEY = "internal|memory"


@dataclass
class _Job:
    previous: str
    # 读取时那条摘要行的 id。None 表示还没有摘要。写回时用它判断有没有被抢先。
    expected_id: int | None
    until_id: int
    messages: list[tuple[str, str]]
    # 取数时的会话生命周期版本。写回时若已被清空推进，整份结果作废。
    generation: int


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
        # 同一用户同一时刻只跑一份摘要。两轮请求同时跨过触发线时会各派一个任务，
        # 写回的 CAS 只能让后一份作废，白调的那次模型调用要在入口就挡掉。
        self._running: set[str] = set()

    def should_run(self, *, pending: int) -> bool:
        """纯判定，不碰库：挤出窗口的条数够多才压。"""
        return pending >= self._settings.summary_trigger_min

    async def run(
        self, *, siwe_user_id: str, own_address: str, job: _Job | None = None
    ) -> None:
        """后台任务入口。任何异常都不外抛——它已经不挂在任何请求上了。

        job 由调用方预先探好时直接传进来，省掉重复查一次水位线。
        """
        if siwe_user_id in self._running:
            return
        self._running.add(siwe_user_id)
        start_metering(scope="background")  # create_task 会继承主请求的计费表，不换一张会串回去
        try:
            job = job or await self.collect(siwe_user_id)
            if job is None:
                return
            await self._summarize(siwe_user_id, own_address, job)
        except Exception:  # noqa: BLE001 后台任务不能把异常丢给事件循环
            logger.exception("summarize task crashed: user=%s", siwe_user_id)
        finally:
            self._running.discard(siwe_user_id)

    async def collect(self, siwe_user_id: str) -> _Job | None:
        """探水位线并取出待压缩区间。返回 None 表示这轮没活干。"""
        async with session_scope() as db:
            # 版本必须先于任何用户数据读取。READ COMMITTED 下每条语句取新快照，
            # 先读消息、后取版本会把「清空前读到的旧消息」贴上清空后的新版本，
            # 写回校验就成了摆设。提到最前面之后：早于它提交的清空已经生效，
            # 读到的就是干净数据；晚于它提交的清空会推进版本，写回直接失败。
            generation = await conversation_repo.current_generation(db, siwe_user_id)
            row = await conversation_repo.latest_summary(db, siwe_user_id)
            since = row.id if row else 0
            previous = row.content if row else ""

            pending, until_id = await conversation_repo.summarizable_range(
                db,
                siwe_user_id,
                window=self._settings.conversation_max_messages,
                since_id=since,
                cap=self._settings.summary_batch_cap,
            )
            if until_id is None or not self.should_run(pending=pending):
                return None

            messages = await conversation_repo.load_range(
                db, siwe_user_id, after_id=since, until_id=until_id
            )
            return _Job(
                previous=previous,
                expected_id=row.id if row else None,
                until_id=until_id,
                messages=messages,
                generation=generation,
            )

    async def _summarize(self, siwe_user_id: str, own_address: str, job: _Job) -> None:
        # 后台任务是独立的 run，但 trace_id 跟着触发它的那轮请求走。
        run = langsmith_tracer.run_config(
            "summary.run",
            trace_id=get_trace_id(),
            until_id=job.until_id,
            siwe_user_id=siwe_user_id,
        )
        if run:
            set_run_id(run["run_id"])
        try:
            raw = await self._llm.summarize(job.previous, job.messages, config=run or None)
        except Exception as exc:  # noqa: BLE001 模型侧任何失败都只降级、不重试
            # 水位线不动，下一轮会重压同一段 —— 失败不需要额外记账。
            await self._charge()
            await self._audit.log("summary_failed", siwe_user_id=siwe_user_id, error=str(exc)[:200])
            logger.warning("summarize failed: user=%s until=%s err=%s", siwe_user_id, job.until_id, exc)
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
            await self._audit.log(
                "summary_failed",
                siwe_user_id=siwe_user_id,
                error="摘要全部被白名单拒绝",
            )
            return

        async with session_scope() as db:
            written = await conversation_repo.write_summary(
                db,
                siwe_user_id,
                content=schema.dumps(fields),
                expected_id=job.expected_id,
                expected_generation=job.generation,
            )
        if not written:
            # 用户在我们调模型的这几秒里清空了对话，或另一个任务先写完了：
            # 这份摘要基于已失效的区间，一个字都不能留。正常竞态，不算失败。
            await self._audit.log(
                "summary_discarded_stale", siwe_user_id=siwe_user_id, until_id=job.until_id
            )
            return
        await self._audit.log(
            "summary_written",
            siwe_user_id=siwe_user_id,
            until_id=job.until_id,
        )

    async def _charge(self) -> None:
        async with session_scope() as db:
            await self._quota.charge(
                db, key=INTERNAL_QUOTA_KEY, metric=QuotaMetric.STEPS, amount=1
            )
