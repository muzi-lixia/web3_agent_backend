"""记忆仓储：active 唯一、supersede 留痕、向量召回、水位线。用构造向量，不起模型。"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from web3_agent.config.settings import get_settings
from web3_agent.domain.memory.memory import MemoryItem
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import ConversationMessage, UserMemory, UserMemoryState
from web3_agent.persistence.pg.repositories import conversation_repo, memory_repo

DIM = 1024


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


def _uid() -> str:
    return str(uuid.uuid4())


def _vec(head: float) -> list[float]:
    return [head] + [0.0] * (DIM - 1)


def _second(index: int) -> list[float]:
    vec = [0.0] * DIM
    vec[index] = 1.0
    return vec


def _item(subject: str, text: str, kind: str = "profile") -> MemoryItem:
    return MemoryItem(text=text, memory_type=kind, subject=subject, confidence=0.9)


def _run(coro):
    async def wrapper():
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            return await coro()
        finally:
            await pg.dispose_engine()

    return asyncio.run(wrapper())


async def _cleanup(uid: str) -> None:
    from sqlalchemy import delete

    async with pg.session_scope() as db:
        await conversation_repo.clear_user_messages(db, uid)
        await db.execute(delete(UserMemory).where(UserMemory.siwe_user_id == uid))
        await db.execute(delete(UserMemoryState).where(UserMemoryState.siwe_user_id == uid))


def test_supersede_leaves_one_active_row() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int]:
        async with pg.session_scope() as db:
            await memory_repo.insert_many(db, uid, [_item("gas", "用户关心 gas 成本")], [_vec(1.0)])
        async with pg.session_scope() as db:
            await memory_repo.supersede(db, uid, ["gas"])
            await memory_repo.insert_many(db, uid, [_item("gas", "用户已经在做套利")], [_vec(1.0)])
        async with pg.session_scope() as db:
            active = await memory_repo.active_items(db, uid)
        async with pg.session_scope() as db:
            total = len(await _all_rows(db, uid))
        return len(active), total

    try:
        active, total = _run(flow)
        assert active == 1
        assert total == 2  # 旧行留痕，不是覆盖
    finally:
        _run(lambda: _cleanup(uid))


async def _all_rows(db, uid: str) -> list[UserMemory]:
    from sqlalchemy import select

    return list(
        (await db.execute(select(UserMemory).where(UserMemory.siwe_user_id == uid)))
        .scalars()
        .all()
    )


def test_partial_unique_index_blocks_second_active() -> None:
    """同 subject 两条 active 必须被库挡住 —— 这是去重的最后一道，不靠代码自觉。"""
    uid = _uid()

    async def flow() -> bool:
        async with pg.session_scope() as db:
            await memory_repo.insert_many(db, uid, [_item("gas", "第一句")], [_vec(1.0)])
        try:
            async with pg.session_scope() as db:
                await memory_repo.insert_many(db, uid, [_item("gas", "第二句")], [_vec(1.0)])
        except IntegrityError:
            return True
        return False

    try:
        assert _run(flow) is True
    finally:
        _run(lambda: _cleanup(uid))


def test_recall_filters_by_distance_and_excludes_constraint() -> None:
    uid = _uid()

    async def flow() -> list[str]:
        async with pg.session_scope() as db:
            await memory_repo.insert_many(
                db,
                uid,
                [
                    _item("near", "方向几乎一样", "preference"),
                    _item("far", "方向差 45 度", "preference"),
                    _item("rule", "不允许转账", "constraint"),
                ],
                [[1.0] + [0.0] * (DIM - 1), [0.7071, 0.7071] + [0.0] * (DIM - 2), [1.0] + [0.0] * (DIM - 1)],
            )
        async with pg.session_scope() as db:
            rows = await memory_repo.recall(
                db, uid, embedding=[1.0] + [0.0] * (DIM - 1), limit=3, max_distance=0.1
            )
        return [r.subject for r in rows]

    try:
        subjects = _run(flow)
        assert subjects == ["near"]  # 45 度那条被距离阈值挡住，constraint 不参与向量召回
    finally:
        _run(lambda: _cleanup(uid))


def test_watermark_advances_and_failure_keeps_it() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int, int]:
        async with pg.session_scope() as db:
            await memory_repo.advance_watermark(db, uid, 42)
        async with pg.session_scope() as db:
            await memory_repo.mark_failed(db, uid)
        async with pg.session_scope() as db:
            state = await memory_repo.get_state(db, uid)
        return state.last_extracted_message_id, state.fail_count, 0

    try:
        watermark, fail_count, _ = _run(flow)
        assert watermark == 42  # 失败不改水位线，下一轮还会重抽这段
        assert fail_count == 1
    finally:
        _run(lambda: _cleanup(uid))


def test_purge_removes_superseded_rows_and_watermark() -> None:
    """合规删除：历史行、active 行、水位线一起清掉，不留「用户聊过多少条」的痕迹。"""
    uid = _uid()

    async def flow() -> tuple[int, int, object]:
        async with pg.session_scope() as db:
            await memory_repo.insert_many(db, uid, [_item("gas", "第一句")], [_vec(1.0)])
        async with pg.session_scope() as db:
            await memory_repo.supersede(db, uid, ["gas"])
            await memory_repo.insert_many(db, uid, [_item("gas", "第二句")], [_vec(1.0)])
            await memory_repo.advance_watermark(db, uid, 7)
        async with pg.session_scope() as db:
            deleted = await memory_repo.purge(db, uid)
        async with pg.session_scope() as db:
            left = len(await _all_rows(db, uid))
            state = await memory_repo.get_state(db, uid)
        return deleted, left, state

    try:
        deleted, left, state = _run(flow)
        assert deleted == 2  # superseded 的旧行也要删
        assert left == 0
        assert state is None
    finally:
        _run(lambda: _cleanup(uid))


def test_pending_for_memory_takes_earliest_batch() -> None:
    uid = _uid()

    async def flow() -> tuple[int, list[str]]:
        async with pg.session_scope() as db:
            rows = [
                ConversationMessage(siwe_user_id=uid, role="user", content=f"m{i}")
                for i in range(1, 26)
            ]
            db.add_all(rows)
            await db.flush()
            first_id = rows[0].id
        async with pg.session_scope() as db:
            total, batch = await conversation_repo.pending_for_memory(
                db, uid, since_id=first_id - 1 + 5, cap=10
            )
        # 从第 6 条开始算，共 20 条待处理，本批取最早的 10 条
        return total, [content for _, _, content in batch]

    try:
        total, batch = _run(flow)
        assert total == 20
        assert batch[0] == "m6"
        assert len(batch) == 10
    finally:
        _run(lambda: _cleanup(uid))
