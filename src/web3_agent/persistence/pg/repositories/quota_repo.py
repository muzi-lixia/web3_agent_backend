"""quota_counter 读写：1 分钟分桶，单语句 upsert 累加，不用外部缓存。

为什么坚持单条 SQL：这是全项目唯一的写热点（每次 LLM 调用、RPC 调用、工具步数都要 +1）。
先 SELECT 再 UPDATE 会在这个点上放大成两倍往返，而且并发下要处理竞态。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.config.constants import QuotaMetric
from web3_agent.persistence.pg.orm_models import QuotaCounter

# BUCKET_SECONDS：固定配额桶长度，单位为秒。
BUCKET_SECONDS = 60


def bucket_start(now: datetime | None = None) -> datetime:
    """把时间向下取整到分钟，作为窗口起点。

    Args:
        now: 注入的当前时间；为空时读取当前 UTC 时间。

    Returns:
        带 UTC 时区的分钟桶起点。
    """
    moment = now or datetime.now(timezone.utc)
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % BUCKET_SECONDS, tz=timezone.utc)


def _subject(user_key: str, metric: QuotaMetric) -> str:
    """将主体键与配额维度编码为唯一计数主体字符串。

    Args:
        user_key: 按用户或 IP 编码的配额主体。
        metric: 本次计数的配额维度。
    """
    return f"{user_key}|{metric.value}"


async def incr(
    db: AsyncSession,
    *,
    user_key: str,
    metric: QuotaMetric,
    amount: int = 1,
    window_start: datetime | None = None,
) -> int:
    """累加并返回本窗口最新值。一条语句完成，返回的 count 就是判定依据。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        user_key: 按用户或 IP 编码的配额主体。
        metric: 本次计数的配额维度。
        amount: 本次累加或预占的资源数量。
        window_start: 要读取或累计的分钟桶起点；为空时使用当前分钟。
    """
    window = window_start or bucket_start()
    table = QuotaCounter.__table__
    # 以主体和窗口为冲突键，单条 SQL 原子累加，避免读后再写的并发丢计数。
    stmt = pg_insert(table).values(
        subject=_subject(user_key, metric), window_start=window, count=amount
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[table.c.subject, table.c.window_start],
        set_={"count": table.c.count + amount},
    ).returning(table.c.count)
    return int((await db.execute(stmt)).scalar_one())


async def read(
    db: AsyncSession,
    *,
    user_key: str,
    metric: QuotaMetric,
    window_start: datetime | None = None,
) -> int:
    """读取当前主体和维度的分钟桶计数，不存在时返回 0。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        user_key: 按用户或 IP 编码的配额主体。
        metric: 本次计数的配额维度。
        window_start: 要读取或累计的分钟桶起点；为空时使用当前分钟。
    """
    window = window_start or bucket_start()
    stmt = select(QuotaCounter.count).where(
        QuotaCounter.subject == _subject(user_key, metric),
        QuotaCounter.window_start == window,
    )
    value = (await db.execute(stmt)).scalar_one_or_none()
    return int(value or 0)


async def read_many(
    db: AsyncSession,
    *,
    user_key: str,
    metrics: Sequence[QuotaMetric],
    window_start: datetime | None = None,
) -> dict[QuotaMetric, int]:
    """一次取多个维度。请求入口要查两条、快照要查四条，逐个查就是白跑的往返。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        user_key: 按用户或 IP 编码的配额主体。
        metrics: 需要批量读取的配额维度集合。
        window_start: 要读取或累计的分钟桶起点；为空时使用当前分钟。

    Returns:
        以 QuotaMetric 为键的计数字典；数据库未记录的维度为 0。
    """
    window = window_start or bucket_start()
    subjects = [_subject(user_key, metric) for metric in metrics]
    stmt = select(QuotaCounter.subject, QuotaCounter.count).where(
        QuotaCounter.subject.in_(subjects),
        QuotaCounter.window_start == window,
    )
    found = {subject: int(count) for subject, count in (await db.execute(stmt)).all()}
    return {metric: found.get(_subject(user_key, metric), 0) for metric in metrics}


async def cleanup(db: AsyncSession, *, older_than_seconds: int = 3600) -> int:
    """回收老窗口。由单实例定时任务调用，PG 没有 TTL。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        older_than_seconds: 保留窗口秒数，更旧的记录可被清理。
    """
    cutoff = bucket_start() - timedelta(seconds=older_than_seconds)
    result = await db.execute(delete(QuotaCounter).where(QuotaCounter.window_start < cutoff))
    return int(result.rowcount or 0)
