"""quota_counter 仓储：同窗口累加、跨窗口隔离、过期窗口回收。"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import delete

from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import QuotaCounter
from web3_agent.persistence.pg.repositories import quota_repo


def _pg_ready() -> bool:
    async def probe() -> bool:
        try:
            pg.init_engine(get_settings())
            await pg.create_all()
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            await pg.dispose_engine()

    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg_ready(), reason="本机 PostgreSQL 不可用")


def _run(coro):
    async def wrapper():
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            return await coro()
        finally:
            await pg.dispose_engine()

    return asyncio.run(wrapper())


def _key() -> str:
    return f"test:{uuid.uuid4().hex[:12]}"


def _drop(key: str) -> None:
    async def flow() -> None:
        async with pg.session_scope() as db:
            await db.execute(
                delete(QuotaCounter).where(QuotaCounter.subject.like(f"{key}|%"))
            )

    _run(flow)


def test_incr_accumulates_inside_one_bucket() -> None:
    key = _key()

    async def flow() -> tuple[int, int, int, int]:
        async with pg.session_scope() as db:
            first = await quota_repo.incr(db, user_key=key, metric=QuotaMetric.QPS)
            second = await quota_repo.incr(db, user_key=key, metric=QuotaMetric.QPS)
            third = await quota_repo.incr(db, user_key=key, metric=QuotaMetric.QPS)
        async with pg.session_scope() as db:
            read = await quota_repo.read(db, user_key=key, metric=QuotaMetric.QPS)
        return first, second, third, read

    try:
        assert _run(flow) == (1, 2, 3, 3)
    finally:
        _drop(key)


def test_cleanup_removes_only_buckets_past_retention() -> None:
    key = _key()
    bucket = quota_repo.bucket_start()

    async def seed() -> None:
        async with pg.session_scope() as db:
            for offset in (timedelta(hours=2), timedelta(minutes=30), timedelta(0)):
                await quota_repo.incr(
                    db, user_key=key, metric=QuotaMetric.QPS, window_start=bucket - offset
                )

    async def count_buckets() -> int:
        total = 0
        for offset in (timedelta(hours=2), timedelta(minutes=30), timedelta(0)):
            async with pg.session_scope() as db:
                total += await quota_repo.read(
                    db, user_key=key, metric=QuotaMetric.QPS, window_start=bucket - offset
                )
        return total

    try:
        _run(seed)
        assert _run(count_buckets) == 3

        async def sweep() -> None:
            async with pg.session_scope() as db:
                await quota_repo.cleanup(db)

        _run(sweep)
        # 只剩两条：半小时前的与当前桶都还在，两小时前的被清掉
        assert _run(count_buckets) == 2
    finally:
        _drop(key)
