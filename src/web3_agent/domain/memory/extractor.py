"""长期记忆第二段：自由文本抽取（V2）。

与摘要同构的两段式：事务取数 → 事务外调模型 → 事务写回。

并发由两件东西共同保证：
- 租约（`user_memory_state.claim_token`）：同一用户同一时刻只允许一个任务在跑，
  两个后台任务不会对着同一段消息各抽一遍；
- 写回时比对读取时的水位线与版本：租约过期被接手、或用户中途清空了对话，
  迟到的任务看到条件不成立就整份丢弃，不回退水位线，也不用自己的旧结论
  把新结论顶成 superseded。

和摘要的关键差别：**抽不出东西是正常结果**，不是失败 —— 大多数对话没有值得
长期记住的背景，这种情况水位线照常推进，否则每轮都会重抽同一段。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from web3_agent.agent import langsmith_tracer
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.common.tracing import get_trace_id, set_run_id
from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import Settings
from web3_agent.domain.memory import memory
from web3_agent.domain.memory.summarizer import INTERNAL_QUOTA_KEY
from web3_agent.domain.quotas import QuotaController
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import conversation_repo, memory_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger
from web3_agent.common.metering import start_metering

logger = logging.getLogger(__name__)


# 数据职责：后台记忆抽取任务快照，携带版本、水位线和租约供安全写回。
@dataclass
class _Job:
    # until_id：本批处理区间的最大消息 ID。
    until_id: int
    # messages：按角色组织的消息序列；后台任务为 (role, content)，图状态为 LangChain 消息。
    messages: list[tuple[str, str]]
    # existing_texts：取数时已有的有效长期记忆正文，供模型避免重复抽取。
    existing_texts: list[str]
    # 取数时的会话生命周期版本。写回时若已被清空推进，记忆与水印都不写。
    # generation：用户会话生命周期版本；清空时递增，旧任务写回前必须比对。
    generation: int
    # 取数时的水位线。写回时要求它没变 —— 变了说明这段已经被别人抽过。
    # watermark：取数时的记忆抽取水位线，写回时必须保持不变。
    watermark: int
    # 本次抽取的租约。写回和失败计数都要求它还在自己手里。
    # claim_token：本次后台任务的唯一租约标识；写回时校验，防止旧执行者覆盖新任务。
    claim_token: str


# 数据职责：后台长期记忆抽取服务，借助租约、水位线和会话版本保护写回。
class MemoryExtractor:
    def __init__(
        self,
        settings: Settings,
        llm: LlmAdapter,
        audit: AuditLogger,
        quota: QuotaController,
    ) -> None:
        """初始化 MemoryExtractor 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
            llm: 统一模型适配器，负责结构化抽取及回答。
            audit: 审计日志写入服务。
            quota: 请求限流和资源记账服务。
        """
        self._settings = settings
        self._llm = llm
        self._audit = audit
        self._quota = quota

    def should_run(self, *, pending: int, fail_count: int) -> bool:
        """按待处理消息数及当前对象的失败限制判断是否应触发后台任务。

        Args:
            pending: 尚未处理的消息条数。
            fail_count: 当前累计的抽取失败次数。
        """
        if pending < self._settings.memory_trigger_messages:
            return False
        return fail_count < self._settings.memory_fail_max

    async def run(self, *, siwe_user_id: str, job: _Job | None = None) -> None:
        # 后台处理建立独立账本，避免继承并继续累加触发它的聊天请求用量。
        """启动独立计量的长期记忆后台抽取，异常记录日志而不影响聊天响应。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
            job: 本次已领取的后台任务快照。
        """
        start_metering(scope="background")  # 后台任务用自己的计费表，不串进任何一轮对话
        try:
            job = job or await self.collect(siwe_user_id)
            if job is None:
                return
            await self._extract(siwe_user_id, job)
        except Exception:  # noqa: BLE001 后台任务不能把异常丢给事件循环
            logger.exception("memory extract crashed: user=%s", siwe_user_id)

    async def collect(self, siwe_user_id: str) -> _Job | None:
        """探水位线并取出待抽取区间。返回 None 表示这轮没活干，或已有任务在跑。

        先抢租约再读水位线，顺序不能反：反过来的话，两个任务会读到同一个水位线，
        抢到租约的后一个拿着过期的 `since` 去抽一段已经被抽过的消息，
        写回时才发现水位线变了 —— 模型调用已经白花了。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
        """
        token = uuid.uuid4().hex
        async with session_scope() as db:
            if not await memory_repo.claim(
                db, siwe_user_id, token, self._settings.memory_claim_ttl_seconds
            ):
                return None
            # 版本先于数据读取，理由同 summarizer.collect：读消息之后才取版本，
            # 中间提交的清空会把旧消息标成新版本，写回校验拦不住。
            generation = await conversation_repo.current_generation(db, siwe_user_id)
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
                # 没活干必须把租约还回去：留着会让这个用户的抽取等到过期才恢复。
                await memory_repo.release_claim(db, siwe_user_id, token)
                return None

            existing = await memory_repo.active_items(db, siwe_user_id)
            return _Job(
                until_id=rows[-1][0],
                messages=[(role, content) for _, role, content in rows],
                existing_texts=[row.text for row in existing],
                generation=generation,
                watermark=since,
                claim_token=token,
            )

    async def _extract(self, siwe_user_id: str, job: _Job) -> None:
        """调用模型抽取记忆，清洗去重后按租约、版本和水位线校验写回。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
            job: 本次已领取的后台任务快照。
        """
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

        # 先过滤不可信候选并按主题取舍，再进入持久化事务。
        items = _dedupe(raw.items, existing=job.existing_texts)
        await self._charge()

        # 抽到东西和没抽到东西走同一条写回路径：没抽到也要推进水位线，
        # 否则这段对话会被反复重抽。两种情况都受同一组条件约束。
        async with session_scope() as db:
            written = await memory_repo.commit_extraction(
                db,
                siwe_user_id,
                until_id=job.until_id,
                items=items,
                expected_watermark=job.watermark,
                expected_generation=job.generation,
                claim_token=job.claim_token,
            )
        if not written:
            # 这段区间已经被别的任务抽过，或者用户中途清空了对话。
            # 两种情况都是正常竞态：整份结果丢掉，不补写、不报错。
            await self._audit.log(
                "memory_discarded_stale",
                siwe_user_id=siwe_user_id,
                until_id=job.until_id,
                watermark=job.watermark,
            )
            return
        if not items:
            return
        await self._audit.log(
            "memory_written",
            siwe_user_id=siwe_user_id,
            until_id=job.until_id,
            subjects=[item.subject for item in items],
        )

    async def _on_failure(self, siwe_user_id: str, exc: Exception, job: _Job) -> None:
        """在仍持有有效任务资格时记录抽取失败，并写入日志与审计。

        Args:
            siwe_user_id: 数据所属用户的稳定标识。
            exc: 需要转换、记录或判断是否重试的原始异常。
            job: 本次已领取的后台任务快照。
        """
        async with session_scope() as db:
            await memory_repo.mark_failed(
                db, siwe_user_id, claim_token=job.claim_token, expected_generation=job.generation
            )
        logger.warning("memory extract failed: user=%s until=%s err=%s", siwe_user_id, job.until_id, exc)
        await self._audit.log("memory_failed", siwe_user_id=siwe_user_id, error=str(exc)[:200])

    async def _charge(self) -> None:
        """把后台模型调用记入内部主体配额，避免消耗用户交互额度。"""
        async with session_scope() as db:
            await self._quota.charge(
                db, key=INTERNAL_QUOTA_KEY, metric=QuotaMetric.STEPS, amount=1
            )


def _dedupe(drafts, *, existing: list[str]) -> list[memory.MemoryItem]:
    """白名单过滤 + 同 subject 只留一条 + 与已有一字不差的丢掉。

    Args:
        drafts: 待清洗去重的模型记忆候选列表。
        existing: 已持久化的记忆文本或常用代币集合。
    """
    seen: dict[str, memory.MemoryItem] = {}
    for draft in drafts:
        item = memory.sanitize(draft.model_dump() if hasattr(draft, "model_dump") else draft)
        if item is None or item.text in existing:
            continue
        current = seen.get(item.subject)
        if current is None or item.confidence > current.confidence:
            seen[item.subject] = item
    return list(seen.values())
