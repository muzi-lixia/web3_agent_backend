"""记忆仓储：active 唯一、supersede 留痕、按时间倒序取用、水位线。"""

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


def _item(subject: str, text: str, kind: str = "profile") -> MemoryItem:
    return MemoryItem(text=text, memory_type=kind, subject=subject, confidence=0.9)


async def _claim(db, uid: str, ttl: int = 60) -> str:
    token = uuid.uuid4().hex
    assert await memory_repo.claim(db, uid, token, ttl) is True
    return token


async def _commit(
    db, uid: str, *, token: str, until: int, watermark: int = 0, items: list[MemoryItem] = ()
) -> bool:
    return await memory_repo.commit_extraction(
        db,
        uid,
        until_id=until,
        items=list(items),
        expected_watermark=watermark,
        expected_generation=0,
        claim_token=token,
    )


def _run(coro):
    async def wrapper():
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            return await coro()
        finally:
            await pg.dispose_engine()

    return asyncio.run(wrapper())


async def _all_rows(db, uid: str) -> list[UserMemory]:
    from sqlalchemy import select

    return list(
        (await db.execute(select(UserMemory).where(UserMemory.siwe_user_id == uid)))
        .scalars()
        .all()
    )


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
            token = await _claim(db, uid)
            await _commit(db, uid, token=token, until=1, items=[_item("gas", "用户关心 gas 成本")])
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            assert await _commit(
                db, uid, token=token, until=2, watermark=1, items=[_item("gas", "用户已经在做套利")]
            )
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


def test_active_items_returns_newest_first() -> None:
    """注入时按 created_at 倒序取前几条 —— 最近提到的就是当下最相关的。"""
    uid = _uid()

    async def flow() -> list[str]:
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            await _commit(
                db, uid, token=token, until=1,
                items=[_item("earliest", "最早"), _item("middle", "中间")],
            )
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            await _commit(
                db, uid, token=token, until=2, watermark=1,
                items=[_item("latest", "最新")],
            )
        async with pg.session_scope() as db:
            rows = await memory_repo.active_items(db, uid)
        return [r.subject for r in rows]

    try:
        assert _run(flow)[0] == "latest"
    finally:
        _run(lambda: _cleanup(uid))


def test_partial_unique_index_blocks_second_active() -> None:
    """同 subject 两条 active 必须被库挡住 —— 这是去重的最后一道，不靠代码自觉。

    直接走 ORM 插行，绕开 commit_extraction 的 supersede：这里验的是索引本身。
    """
    uid = _uid()

    async def flow() -> bool:
        async with pg.session_scope() as db:
            db.add(
                UserMemory(
                    siwe_user_id=uid, text="第一句", memory_type="profile",
                    subject="gas", confidence=0.9,
                )
            )
            await db.flush()
        try:
            async with pg.session_scope() as db:
                db.add(
                    UserMemory(
                        siwe_user_id=uid, text="第二句", memory_type="profile",
                        subject="gas", confidence=0.9,
                    )
                )
                await db.flush()
        except IntegrityError:
            return True
        return False

    try:
        assert _run(flow) is True
    finally:
        _run(lambda: _cleanup(uid))


def test_watermark_advances_and_failure_keeps_it() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int]:
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            assert await _commit(db, uid, token=token, until=42)
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            await memory_repo.mark_failed(db, uid, claim_token=token, expected_generation=0)
        async with pg.session_scope() as db:
            state = await memory_repo.get_state(db, uid)
        return state.last_extracted_message_id, state.fail_count

    try:
        watermark, fail_count = _run(flow)
        assert watermark == 42  # 失败不改水位线，下一轮还会重抽这段
        assert fail_count == 1
    finally:
        _run(lambda: _cleanup(uid))


def test_purge_removes_superseded_rows_and_watermark() -> None:
    """合规删除：历史行、active 行、水位线一起清掉，不留「用户聊过多少条」的痕迹。"""
    uid = _uid()

    async def flow() -> tuple[int, int, object]:
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            await _commit(db, uid, token=token, until=1, items=[_item("gas", "第一句")])
        async with pg.session_scope() as db:
            token = await _claim(db, uid)
            assert await _commit(
                db, uid, token=token, until=7, watermark=1, items=[_item("gas", "第二句")]
            )
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


def test_pending_for_memory_skips_summary_row() -> None:
    """摘要行不是对话内容，抽取记忆时不能把它当一轮消息喂给模型。"""
    uid = _uid()

    async def flow() -> list[str]:
        async with pg.session_scope() as db:
            await conversation_repo.append_message(db, siwe_user_id=uid, role="user", content="m1")
        async with pg.session_scope() as db:
            await conversation_repo.write_summary(
                db, uid, content='{"core_needs":["摘要"]}', expected_id=None, expected_generation=0
            )
        async with pg.session_scope() as db:
            await conversation_repo.append_message(db, siwe_user_id=uid, role="user", content="m2")
        async with pg.session_scope() as db:
            total, batch = await conversation_repo.pending_for_memory(db, uid, since_id=0, cap=10)
        assert total == 2  # 两条消息，摘要行不算
        return [content for _, _, content in batch]

    try:
        assert _run(flow) == ["m1", "m2"]
    finally:
        _run(lambda: _cleanup(uid))
