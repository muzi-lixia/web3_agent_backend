"""workflow_task 的兜底查询：只取该扫的行。"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete

from web3_agent.config.constants import WorkflowState
from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import WorkflowTask
from web3_agent.persistence.pg.repositories import workflow_repo

# _USER：任务归属用户。
_USER = "user-1"
# _ADDRESS：任务归属钱包。
_ADDRESS = "0x" + "1" * 40
# _HASH：已广播的交易哈希。
_HASH = "0x" + "a" * 64


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


def _seed(
    db,
    prefix: str,
    *,
    state: WorkflowState,
    tx_hash: str | None,
    age_seconds: int,
    expires_in_seconds: int = 300,
) -> None:
    """插入一条任务行，updated_at 由 age_seconds 控制。

    Args:
        db: 调用方提供的异步数据库会话。
        prefix: 任务标识前缀，用于清理时圈定本次数据。
        state: 任务当前状态。
        tx_hash: 广播哈希；未广播为空。
        age_seconds: 距当前时刻的秒数，用来构造静默期内外。
        expires_in_seconds: 相对当前时刻的有效期，负数表示已经过期。
    """
    now = datetime.now(timezone.utc)
    db.add(
        WorkflowTask(
            task_id=f"{prefix}-{uuid.uuid4().hex[:12]}",
            siwe_user_id=_USER,
            siwe_address=_ADDRESS,
            action="transfer",
            chain_id=80094,
            state=state.value,
            expire_type="NONE",
            params={},
            tx_intent=None,
            tx_hash=tx_hash,
            expires_at=now + timedelta(seconds=expires_in_seconds),
            updated_at=now - timedelta(seconds=age_seconds),
        )
    )


def test_list_pending_skips_unbroadcast_finished_and_fresh_rows() -> None:
    prefix = f"sw-{uuid.uuid4().hex[:8]}"

    async def flow() -> list[str]:
        now = datetime.now(timezone.utc)
        async with pg.session_scope() as db:
            _seed(db, prefix, state=WorkflowState.SUBMITTED, tx_hash=_HASH, age_seconds=600)
            _seed(db, prefix, state=WorkflowState.PREVIEW, tx_hash=_HASH, age_seconds=600)
            # 没有哈希：交易还没广播，查回执没有意义。
            _seed(db, prefix, state=WorkflowState.SUBMITTED, tx_hash=None, age_seconds=600)
            # 终态：不再推进。
            _seed(db, prefix, state=WorkflowState.SETTLED, tx_hash=_HASH, age_seconds=600)
            # 静默期内：刚被推进过，不重复扫。
            _seed(db, prefix, state=WorkflowState.SUBMITTED, tx_hash=_HASH, age_seconds=5)
        async with pg.session_scope() as db:
            rows = await workflow_repo.list_pending(
                db,
                states=(WorkflowState.PREVIEW.value, WorkflowState.SUBMITTED.value),
                stale_before=now - timedelta(seconds=60),
                # 本次只关心自己插的行，其余残留不参与断言，给足额度避免被挤出窗口。
                limit=200,
            )
        return [(row.state, row.tx_hash) for row in rows if row.task_id.startswith(prefix)]

    async def cleanup() -> None:
        async with pg.session_scope() as db:
            await db.execute(delete(WorkflowTask).where(WorkflowTask.task_id.like(f"{prefix}-%")))

    try:
        rows = _run(flow)
    finally:
        _run(cleanup)
    assert sorted(rows) == sorted(
        [
            (WorkflowState.SUBMITTED.value, _HASH),
            (WorkflowState.PREVIEW.value, _HASH),
        ]
    )


def test_list_pending_includes_expired_unbroadcast_rows() -> None:
    """等确认却没人点、且已过期的行也要扫到，否则会一直挂在 INTENT_CONFIRM。"""
    prefix = f"sw-{uuid.uuid4().hex[:8]}"

    async def flow() -> list[tuple[str, bool]]:
        now = datetime.now(timezone.utc)
        async with pg.session_scope() as db:
            _seed(
                db,
                prefix,
                state=WorkflowState.INTENT_CONFIRM,
                tx_hash=None,
                age_seconds=600,
                expires_in_seconds=-60,
            )
            # 还没过期：没有哈希，也没有要核对的东西，不扫。
            _seed(
                db,
                prefix,
                state=WorkflowState.INTENT_CONFIRM,
                tx_hash=None,
                age_seconds=600,
            )
        async with pg.session_scope() as db:
            rows = await workflow_repo.list_pending(
                db,
                states=(WorkflowState.INTENT_CONFIRM.value,),
                stale_before=now - timedelta(seconds=60),
                limit=200,
            )
        return [
            (row.state, row.tx_hash is not None)
            for row in rows
            if row.task_id.startswith(prefix)
        ]

    async def cleanup() -> None:
        async with pg.session_scope() as db:
            await db.execute(delete(WorkflowTask).where(WorkflowTask.task_id.like(f"{prefix}-%")))

    try:
        rows = _run(flow)
    finally:
        _run(cleanup)
    assert rows == [(WorkflowState.INTENT_CONFIRM.value, False)]
