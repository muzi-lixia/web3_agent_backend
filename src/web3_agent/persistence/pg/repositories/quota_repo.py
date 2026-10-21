"""quota_counter 读写：1 分钟分桶，单语句 upsert 累加，不用外部缓存。

为什么坚持单条 SQL：这是全项目唯一的写热点（每次 LLM 调用、RPC 调用、工具步数都要 +1）。
先 SELECT 再 UPDATE 会在这个点上放大成两倍往返，而且并发下要处理竞态。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.config.constants import QuotaMetric
from web3_agent.persistence.pg.orm_models import QuotaCounter

BUCKET_SECONDS = 60


def bucket_start(now: datetime | None = None) -> datetime:
    """把时间向下取整到分钟，作为窗口起点。"""
    moment = now or datetime.now(timezone.utc)
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % BUCKET_SECONDS, tz=timezone.utc)


def _subject(user_key: str, metric: QuotaMetric) -> str:
    return f"{user_key}|{metric.value}"


async def incr(
    db: AsyncSession,
    *,
    user_key: str,
    metric: QuotaMetric,
    amount: int = 1,
    window_start: datetime | None = None,
) -> int:
    """累加并返回本窗口最新值。一条语句完成，返回的 count 就是判定依据。"""
    window = window_start or bucket_start()
    table = QuotaCounter.__table__
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
    window = window_start or bucket_start()
    stmt = select(QuotaCounter.count).where(
        QuotaCounter.subject == _subject(user_key, metric),
        QuotaCounter.window_start == window,
    )
    value = (await db.execute(stmt)).scalar_one_or_none()
    return int(value or 0)


async def cleanup(db: AsyncSession, *, older_than_seconds: int = 3600) -> int:
    """回收老窗口。由单实例定时任务调用，PG 没有 TTL。"""
    cutoff = bucket_start() - timedelta(seconds=older_than_seconds)
    result = await db.execute(delete(QuotaCounter).where(QuotaCounter.window_start < cutoff))
    return int(result.rowcount or 0)
