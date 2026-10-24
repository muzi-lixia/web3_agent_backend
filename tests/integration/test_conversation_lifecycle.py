"""会话生命周期一致性：清空/删除之后，旧任务不许把内容写回来。

这一组必须用真库。问题本身是两个事务的可见性与锁顺序 ——
假 session 只能验证「函数被调用了」，验不出「写回到底有没有落库」。

两种交错都要覆盖：
- 写事务先拿到共享锁 → 清空阻塞 → 提交后清空把刚写的那行删掉（不复活）；
- 清空先提交 → 写事务的版本判断失败 → 一个字都不写。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.domain.memory.memory import MemoryItem
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import (
    ConversationGeneration,
    UserMemory,
    UserMemoryState,
)
from web3_agent.persistence.pg.repositories import conversation_repo, memory_repo
from sqlalchemy import delete


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


def _run(coro):
    async def wrapper():
        pg.init_engine(get_settings())
        await pg.create_all()
        try:
            return await coro()
        finally:
            await pg.dispose_engine()

    return asyncio.run(wrapper())


def _item(subject: str, text: str) -> MemoryItem:
    return MemoryItem(text=text, memory_type="profile", subject=subject, confidence=0.9)


async def _cleanup(uid: str) -> None:
    async with pg.session_scope() as db:
        await conversation_repo.clear_user_messages(db, uid)
        await db.execute(delete(UserMemory).where(UserMemory.siwe_user_id == uid))
        await db.execute(delete(UserMemoryState).where(UserMemoryState.siwe_user_id == uid))
        await db.execute(
            delete(ConversationGeneration).where(ConversationGeneration.siwe_user_id == uid)
        )


async def _seed_message(uid: str, content: str = "帮我查一下余额") -> None:
    async with pg.session_scope() as db:
        await conversation_repo.append_message(
            db, siwe_user_id=uid, role="user", content=content
        )


def test_generation_is_monotonic() -> None:
    uid = _uid()

    async def flow() -> tuple[int, int, int]:
        async with pg.session_scope() as db:
            first = await conversation_repo.current_generation(db, uid)
        async with pg.session_scope() as db:
            bumped = await conversation_repo.bump_generation(db, uid)
        async with pg.session_scope() as db:
            again = await conversation_repo.bump_generation(db, uid)
        return first, bumped, again

    try:
        first, bumped, again = _run(flow)
        assert first == 0  # 没有版本行时按 0 算
        assert bumped == 1  # 第一次 bump 建行时写 1，不是 0
        assert again == 2
    finally:
        _run(lambda: _cleanup(uid))


def test_write_after_clear_is_rejected_and_row_stays_deleted() -> None:
    """摘要的首次写入（还没有摘要行的那条分支）是最容易复活的一条。"""
    uid = _uid()

    async def flow() -> tuple[bool, object, str]:
        await _seed_message(uid)
        async with pg.session_scope() as db:
            generation = await conversation_repo.current_generation(db, uid)
        # 用户清空：推进版本 + 删消息（摘要行也在同一张表里）
        async with pg.session_scope() as db:
            await conversation_repo.bump_generation(db, uid)
            await conversation_repo.clear_user_messages(db, uid)
        # 后台任务拿着清空前读到的版本写回
        async with pg.session_scope() as db:
            written = await conversation_repo.write_summary(
                db,
                uid,
                content='{"core_needs": ["清空前的对话"]}',
                expected_id=None,
                expected_generation=generation,
            )
        async with pg.session_scope() as db:
            row = await conversation_repo.latest_summary(db, uid)
            messages = await conversation_repo.list_page(db, uid, 10)
        return written, row, "" if not messages else messages[0].content

    try:
        written, row, leftover = _run(flow)
        assert written is False
        assert row is None  # 不是「写进去了但内容为空」，是压根没建行
        assert leftover == ""
    finally:
        _run(lambda: _cleanup(uid))


def test_memory_write_and_watermark_are_rejected_after_clear() -> None:
    """合规删除之后，长期记忆与水位线都不能自己长回来。"""
    uid = _uid()

    async def flow() -> tuple[bool, int, object]:
        # 用户在删除前跑着一次抽取：抢到租约、读好版本
        async with pg.session_scope() as db:
            assert await memory_repo.claim(db, uid, "turn-token", 60) is True
            generation = await conversation_repo.current_generation(db, uid)
        # 删除把记忆、水位线（连同租约行）一起清掉
        async with pg.session_scope() as db:
            await conversation_repo.bump_generation(db, uid)
            await memory_repo.purge(db, uid)
        async with pg.session_scope() as db:
            written = await memory_repo.commit_extraction(
                db,
                uid,
                until_id=42,
                items=[_item("gas", "用户关心 gas 成本")],
                expected_watermark=0,
                expected_generation=generation,
                claim_token="turn-token",
            )
        async with pg.session_scope() as db:
            rows = await memory_repo.active_items(db, uid)
            state = await memory_repo.get_state(db, uid)
        return written, len(rows), state

    try:
        written, active, state = _run(flow)
        assert written is False
        assert active == 0
        assert state is None
    finally:
        _run(lambda: _cleanup(uid))


def test_clear_first_then_a_concurrent_turn_is_discarded() -> None:
    """清空先提交，正在生成的那一轮回来时版本已经变了。"""
    uid = _uid()

    async def flow() -> tuple[bool, int, list]:
        await _seed_message(uid)
        async with pg.session_scope() as db:
            generation = await conversation_repo.lock_generation(db, uid)
        # 模型正在生成的那十几秒里，用户点了清空
        async with pg.session_scope() as db:
            await conversation_repo.bump_generation(db, uid)
            await conversation_repo.clear_user_messages(db, uid)
        async with pg.session_scope() as db:
            fresh = await conversation_repo.assert_generation(db, uid, generation)
            if fresh:
                await conversation_repo.append_message(
                    db, siwe_user_id=uid, role="assistant", content="迟到的回答"
                )
        async with pg.session_scope() as db:
            rows = await conversation_repo.list_page(db, uid, 10)
            current = await conversation_repo.current_generation(db, uid)
        return fresh, current, rows

    try:
        fresh, current, rows = _run(flow)
        assert fresh is False
        assert current == 1
        assert rows == []
    finally:
        _run(lambda: _cleanup(uid))


def test_write_in_flight_is_deleted_by_clear_and_never_resurrected() -> None:
    """写事务先拿到共享锁：清空必须等它提交，然后把那行一起删掉。

    顺序断言是关键 —— 如果清空没有等，就会出现「清完之后写入落库」。
    """
    uid = _uid()
    marks: list[str] = []

    async def flow() -> tuple[list[str], list, int]:
        async def writer() -> None:
            async with pg.session_scope() as db:
                fresh = await conversation_repo.assert_generation(db, uid, 0)
                assert fresh is True
                await conversation_repo.append_message(
                    db, siwe_user_id=uid, role="assistant", content="清空前的回答"
                )
                marks.append("write-locked")
                await asyncio.sleep(0.3)  # 模拟写入尚未提交
            marks.append("write-committed")

        async def clearer() -> None:
            await asyncio.sleep(0.1)  # 让 writer 先拿到锁
            async with pg.session_scope() as db:
                await conversation_repo.bump_generation(db, uid)
                await conversation_repo.clear_user_messages(db, uid)
            marks.append("cleared")

        await asyncio.gather(writer(), clearer())
        async with pg.session_scope() as db:
            rows = await conversation_repo.list_page(db, uid, 10)
            generation = await conversation_repo.current_generation(db, uid)
        return marks, rows, generation

    try:
        marks, rows, generation = _run(flow)
        # 写入先拿锁、后提交；清空被阻塞到提交之后才执行
        assert marks == ["write-locked", "write-committed", "cleared"]
        assert rows == []  # 写入的那行被同一次清空带走
        assert generation == 1
    finally:
        _run(lambda: _cleanup(uid))


def test_interleaved_turns_are_read_back_in_turn_order() -> None:
    """两台设备同时提问：id 顺序是 u1,u2,a1,a2，读出来必须是 u1,a1,u2,a2。

    否则模型会把 a1 当成对 u2 的回答，上下文整体错位。
    """
    uid = _uid()
    turn_a, turn_b = uuid.uuid4().hex, uuid.uuid4().hex

    async def flow() -> list[tuple[str, str]]:
        async with pg.session_scope() as db:
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="user", content="A 的问题", turn_id=turn_a
            )
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="user", content="B 的问题", turn_id=turn_b
            )
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="assistant", content="A 的回答", turn_id=turn_a
            )
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="assistant", content="B 的回答", turn_id=turn_b
            )
        async with pg.session_scope() as db:
            return await conversation_repo.load_history(db, uid, 20, 6000)

    try:
        history = _run(flow)
        assert history == [
            ("user", "A 的问题"),
            ("assistant", "A 的回答"),
            ("user", "B 的问题"),
            ("assistant", "B 的回答"),
        ]
    finally:
        _run(lambda: _cleanup(uid))


def test_legacy_rows_without_turn_id_keep_id_order() -> None:
    """老数据没有 turn_id，不能因为分组把顺序打乱。"""
    uid = _uid()

    async def flow() -> list[tuple[str, str]]:
        await _seed_message(uid, "旧问题一")
        await _seed_message(uid, "旧问题二")
        async with pg.session_scope() as db:
            return await conversation_repo.load_history(db, uid, 20, 6000)

    try:
        assert _run(flow) == [("user", "旧问题一"), ("user", "旧问题二")]
    finally:
        _run(lambda: _cleanup(uid))
