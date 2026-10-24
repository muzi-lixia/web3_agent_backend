"""摘要存储：区间计算、删旧插新的幂等、清空后不复活。走真实 PG，缺库时整组跳过。

摘要就是 conversation_message 里 role='summary' 的一行，它的 id 同时是水位线。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import ConversationMessage
from web3_agent.persistence.pg.repositories import conversation_repo


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
    return str(uuid.uuid4())


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


def test_summary_row_does_not_take_a_window_slot() -> None:
    """摘要行自己也有 id，但它不是对话内容。

    算进窗口的话，每压一次就会把一条真实消息挤出窗口之外 —— 越压越早，
    最后摘要覆盖的区间和窗口对不上。
    """
    uid = _uid()

    async def flow() -> tuple[int, int | None]:
        await _seed(uid, 45)
        async with pg.session_scope() as db:
            await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["x"]}', expected_id=None, expected_generation=0
            )
        async with pg.session_scope() as db:
            return await conversation_repo.summarizable_range(
                db, uid, window=20, since_id=0, cap=30
            )

    try:
        count, _ = _run(flow)
        # 45 条消息 + 1 条摘要，窗口 20 仍然只数消息：挤出 25 条。
        assert count == 25
    finally:
        _run(lambda: _cleanup(uid))


def test_write_rejects_when_someone_else_wrote_first() -> None:
    uid = _uid()

    async def flow() -> tuple[bool, bool, str]:
        async with pg.session_scope() as db:
            first = await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["先到的"]}',
                expected_id=None, expected_generation=0,
            )
        # 第二个任务也是从「还没有摘要」开始读的，它的区间更旧，必须放弃。
        async with pg.session_scope() as db:
            stale = await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["迟到的"]}',
                expected_id=None, expected_generation=0,
            )
        async with pg.session_scope() as db:
            row = await conversation_repo.latest_summary(db, uid)
        return first, stale, (row.content if row else "")

    try:
        first, stale, content = _run(flow)
        assert first is True
        assert stale is False
        assert "先到的" in content
    finally:
        _run(lambda: _cleanup(uid))


def test_write_replaces_and_advances_watermark() -> None:
    uid = _uid()

    async def flow() -> tuple[bool, int | None, int]:
        async with pg.session_scope() as db:
            await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["第一版"]}',
                expected_id=None, expected_generation=0,
            )
        async with pg.session_scope() as db:
            row = await conversation_repo.latest_summary(db, uid)
            old_id = row.id if row else None
        async with pg.session_scope() as db:
            written = await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["第二版"]}',
                expected_id=old_id, expected_generation=0,
            )
        async with pg.session_scope() as db:
            row = await conversation_repo.latest_summary(db, uid)
        return written, old_id, (row.id if row else -1)

    try:
        written, old_id, new_id = _run(flow)
        assert written is True
        # 删旧插新：新行的 id 更大，水位线跟着前进，而不是原地 UPDATE。
        assert old_id is not None and new_id > old_id
    finally:
        _run(lambda: _cleanup(uid))


def test_stale_write_after_clear_does_not_resurrect() -> None:
    uid = _uid()

    async def flow() -> tuple[bool, str]:
        async with pg.session_scope() as db:
            await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["要查余额"]}',
                expected_id=None, expected_generation=0,
            )
        # 清空：摘要行跟消息一起删
        async with pg.session_scope() as db:
            await conversation_repo.clear_user_messages(db, uid)
        # 后台任务拿着清空前读到的 id 写回
        async with pg.session_scope() as db:
            written = await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["清空前的对话"]}',
                expected_id=1, expected_generation=0,
            )
        async with pg.session_scope() as db:
            row = await conversation_repo.latest_summary(db, uid)
        return written, (row.content if row else "")

    try:
        written, content = _run(flow)
        assert written is False
        assert content == ""
    finally:
        _run(lambda: _cleanup(uid))


def test_write_after_generation_bump_is_rejected() -> None:
    """用户清空时如果原本没有摘要行，删除条数判断不出来 —— 版本号是最后一道闸。"""
    uid = _uid()

    async def flow() -> bool:
        async with pg.session_scope() as db:
            await conversation_repo.bump_generation(db, uid)
        async with pg.session_scope() as db:
            return await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["清空前的对话"]}',
                expected_id=None, expected_generation=0,
            )

    try:
        assert _run(flow) is False
    finally:
        _run(lambda: _cleanup(uid))
