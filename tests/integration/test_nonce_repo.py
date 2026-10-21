"""nonce 仓储：单次核销与过期清理的保留窗口。"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import SiweNonce
from web3_agent.persistence.pg.repositories import nonce_repo


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _seed(rows: list[tuple[str, datetime]]) -> None:
    async with pg.session_scope() as db:
        for nonce, expires_at in rows:
            db.add(SiweNonce(nonce=nonce, expires_at=expires_at))


async def _exists(nonce: str) -> bool:
    async with pg.session_scope() as db:
        return await nonce_repo.peek(db, nonce) is not None


def test_consume_only_once() -> None:
    nonce = uuid.uuid4().hex

    async def flow() -> tuple[str | None, str]:
        await _seed([(nonce, _now() + timedelta(seconds=300))])
        async with pg.session_scope() as db:
            first = await nonce_repo.consume(db, nonce)
        async with pg.session_scope() as db:
            second = await nonce_repo.consume(db, nonce)
        return first, second

    first, second = _run(flow)
    # issue() 未绑定地址时 address 为空 → 核销成功返回 None，不是空串
    assert first is None
    assert second == ""  # 第二次已 consumed，返回空串表示核销失败


def test_consume_rejects_expired_nonce() -> None:
    nonce = uuid.uuid4().hex

    async def flow() -> str:
        await _seed([(nonce, _now() - timedelta(seconds=1))])
        async with pg.session_scope() as db:
            return await nonce_repo.consume(db, nonce)

    assert _run(flow) == ""


def test_cleanup_keeps_rows_inside_retention_window() -> None:
    fresh = uuid.uuid4().hex  # 已过期 1 小时，仍在 24h 保留窗口内
    long_expired = uuid.uuid4().hex  # 已过期 25 小时，超出窗口
    alive = uuid.uuid4().hex  # 还没过期

    async def flow() -> int:
        await _seed(
            [
                (fresh, _now() - timedelta(hours=1)),
                (long_expired, _now() - timedelta(hours=25)),
                (alive, _now() + timedelta(seconds=300)),
            ]
        )
        async with pg.session_scope() as db:
            return await nonce_repo.cleanup(db)

    try:
        removed = _run(flow)
        assert removed == 1
        assert _run(lambda: _exists(long_expired)) is False
        assert _run(lambda: _exists(fresh)) is True
        assert _run(lambda: _exists(alive)) is True
    finally:
        async def drop() -> None:
            async with pg.session_scope() as db:
                for nonce in (fresh, long_expired, alive):
                    row = await nonce_repo.peek(db, nonce)
                    if row is not None:
                        await db.delete(row)

        _run(drop)
