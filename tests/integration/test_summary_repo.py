"""摘要仓储：区间计算、水位线幂等、清空后不复活。走真实 PG，缺库时整组跳过。"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import ConversationMessage
from web3_agent.persistence.pg.repositories import conversation_repo, summary_repo


def _pg_ready() -> bool:
    async def probe() -> bool:
        try:
            pg.init_engine(get_settings())
            await pg.create_all()
            return True
        except Exception:  # noqa: BLE001 探测失败即跳过
            return False
        finally:
            await pg.dispose_engine()

    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(not _pg_ready(), reason="本机 PostgreSQL 不可用")


def _uid() -> str:
    return f"eip155:0x{uuid.uuid4().hex[:40]}"


async def _seed(uid: str, count: int) -> list[int]:
    async with pg.session_scope() as db:
        rows = [
            ConversationMessage(siwe_user_id=uid, role="user", content=f"m{i}")
            for i in range(1, count + 1)
        ]
        db.add_all(rows)
        await db.flush()
        return [r.id for r in rows]


async def _cleanup(uid: str) -> None:
    async with pg.session_scope() as db:
        await conversation_repo.clear_user_messages(db, uid)
        await summary_repo.delete(db, uid)


def _run(coro):
    async def wrapper():
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            return await coro()
        finally:
            await pg.dispose_engine()

    return asyncio.run(wrapper())


def test_range_starts_at_window_edge() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int | None]:
        await _seed(uid, 45)
        async with pg.session_scope() as db:
            return await conversation_repo.summarizable_range(
                db, uid, window=20, since_id=0, cap=30
            )

    try:
        count, until_id = _run(flow)
        assert count == 25  # 最近 20 条占住窗口，挤出 25 条
        assert until_id is not None
    finally:
        _run(lambda: _cleanup(uid))


def test_range_respects_watermark_and_cap() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int | None]:
        ids = await _seed(uid, 45)
        async with pg.session_scope() as db:
            # 水位线推到第 5 条之后，待压 20 条；单批上限 10 条
            return await conversation_repo.summarizable_range(
                db, uid, window=20, since_id=ids[4], cap=10
            )

    try:
        count, until_id = _run(flow)
        assert count == 20
        assert until_id is not None
    finally:
        _run(lambda: _cleanup(uid))


def test_range_empty_when_all_inside_window() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int | None]:
        await _seed(uid, 12)
        async with pg.session_scope() as db:
            return await conversation_repo.summarizable_range(
                db, uid, window=20, since_id=0, cap=30
            )

    try:
        assert _run(flow) == (0, None)
    finally:
        _run(lambda: _cleanup(uid))


def test_watermark_rejects_stale_write() -> None:
    uid = _uid()

    async def flow() -> tuple[bool, bool, bool]:
        async with pg.session_scope() as db:
            created = await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=uid,
                content='{"core_needs": ["旧"]}',
                last_summarized_message_id=5,
                expected_watermark=None,
            )
        async with pg.session_scope() as db:
            stale = await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=uid,
                content='{"core_needs": ["过期结果"]}',
                last_summarized_message_id=9,
                expected_watermark=3,
            )
        async with pg.session_scope() as db:
            fresh = await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=uid,
                content='{"core_needs": ["新"]}',
                last_summarized_message_id=9,
                expected_watermark=5,
            )
        return created, stale, fresh

    try:
        created, stale, fresh = _run(flow)
        assert created is True
        assert stale is False
        assert fresh is True
    finally:
        _run(lambda: _cleanup(uid))


def test_stale_write_after_clear_does_not_resurrect() -> None:
    uid = _uid()

    async def flow() -> tuple[bool, bool, str]:
        async with pg.session_scope() as db:
            await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=uid,
                content='{"core_needs": ["要查余额"]}',
                last_summarized_message_id=5,
                expected_watermark=None,
            )
        # 清空：消息与摘要一起删
        async with pg.session_scope() as db:
            await conversation_repo.clear_user_messages(db, uid)
            await summary_repo.delete(db, uid)
        # 后台任务拿着旧水位线写回
        async with pg.session_scope() as db:
            written = await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=uid,
                content='{"core_needs": ["清空前的对话"]}',
                last_summarized_message_id=5,
                expected_watermark=5,
            )
        async with pg.session_scope() as db:
            row = await summary_repo.get(db, uid)
        return written, row is None, (row.content if row else "")

    try:
        written, gone, content = _run(flow)
        assert written is False
        assert gone is True
        assert content == ""
    finally:
        _run(lambda: _cleanup(uid))


def test_mark_failed_counts_and_success_resets() -> None:
    uid = _uid()

    async def flow() -> tuple[int, str, int]:
        async with pg.session_scope() as db:
            await summary_repo.mark_failed(db, uid)
        async with pg.session_scope() as db:
            await summary_repo.mark_failed(db, uid)
        async with pg.session_scope() as db:
            row = await summary_repo.get(db, uid)
            failed_count, status = row.fail_count, row.status
        async with pg.session_scope() as db:
            await summary_repo.upsert_with_watermark(
                db,
                siwe_user_id=uid,
                content='{"core_needs": ["x"]}',
                last_summarized_message_id=5,
                expected_watermark=0,
            )
        async with pg.session_scope() as db:
            row = await summary_repo.get(db, uid)
        return failed_count, status, row.fail_count

    try:
        failed_count, status, after_success = _run(flow)
        assert failed_count == 2
        assert status == "failed"
        assert after_success == 0
    finally:
        _run(lambda: _cleanup(uid))
