"""记忆抽取的并发一致性：租约互斥、水位线单调、旧任务不许覆盖新结论。

必须用真库。要验的是同一行上的条件更新在并发下谁赢 —— 假 session 只能验
「函数被调用过」，验不出「旧任务到底有没有把新结论标成 superseded」，
也验不出「被丢弃的那次到底有没有留下半条记忆」。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import delete, select

from web3_agent.config.settings import get_settings
from web3_agent.domain.memory.extractor import MemoryExtractor
from web3_agent.domain.memory.memory import MemoryItem
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import (
    ConversationGeneration,
    ConversationMessage,
    UserMemory,
    UserMemoryState,
)
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


async def _commit(
    db, uid: str, *, token: str, until: int, watermark: int, text: str = "结论"
) -> bool:
    return await memory_repo.commit_extraction(
        db,
        uid,
        until_id=until,
        items=[_item("gas", text)],
        expected_watermark=watermark,
        expected_generation=0,
        claim_token=token,
    )


async def _active_texts(db, uid: str) -> list[str]:
    return [row.text for row in await memory_repo.active_items(db, uid)]


def test_claim_is_exclusive_under_concurrency() -> None:
    """两个后台任务同时起来，只有一个能拿到租约 —— 另一个不调模型、不写库。"""
    uid = _uid()

    async def flow() -> list[bool]:
        async def attempt(token: str) -> bool:
            async with pg.session_scope() as db:
                got = await memory_repo.claim(db, uid, token, 60)
                if got:
                    await asyncio.sleep(0.2)  # 模拟任务正在抽取，租约被持有
            return got

        return await asyncio.gather(attempt("token-a"), attempt("token-b"))

    try:
        results = _run(flow)
        assert sorted(results) == [False, True]
    finally:
        _run(lambda: _cleanup(uid))


def test_claim_can_be_taken_over_after_expiry() -> None:
    """进程被杀死时租约不会自己释放，过期之后下一个任务必须能接手。"""
    uid = _uid()

    async def flow() -> tuple[bool, bool, bool]:
        async with pg.session_scope() as db:
            first = await memory_repo.claim(db, uid, "dead-task", 0)  # 立刻过期
        async with pg.session_scope() as db:
            second = await memory_repo.claim(db, uid, "live-task", 60)
        # 过期的那一个再回来想释放，不能把别人的租约清掉
        async with pg.session_scope() as db:
            await memory_repo.release_claim(db, uid, "dead-task")
            state = await memory_repo.get_state(db, uid)
        async with pg.session_scope() as db:
            third = await memory_repo.claim(db, uid, "third", 60)
        return first, second, third

    try:
        first, second, third = _run(flow)
        assert first is True
        assert second is True
        assert third is False  # 仍在 live-task 手里
    finally:
        _run(lambda: _cleanup(uid))


def test_stale_job_cannot_regress_watermark_or_revive_old_conclusion() -> None:
    """慢任务后完成：水位线不能被拉回去，旧结论也不能把新结论顶掉。"""
    uid = _uid()

    async def flow() -> tuple[bool, bool, int, list[str], int]:
        # A 抢到租约后卡住（租约过期），读到的水位线是 0
        async with pg.session_scope() as db:
            assert await memory_repo.claim(db, uid, "slow-a", 0) is True
        # B 接手，抽到第 20 条，写入新结论
        async with pg.session_scope() as db:
            assert await memory_repo.claim(db, uid, "fast-b", 60) is True
            wrote_b = await _commit(
                db, uid, token="fast-b", until=20, watermark=0, text="用户已经在做套利"
            )
        # A 醒过来写回它那一段（1–10）的旧结论
        async with pg.session_scope() as db:
            wrote_a = await _commit(
                db, uid, token="slow-a", until=10, watermark=0, text="用户关心 gas 成本"
            )
        async with pg.session_scope() as db:
            state = await memory_repo.get_state(db, uid)
            texts = await _active_texts(db, uid)
            total = len(
                (
                    await db.execute(select(UserMemory).where(UserMemory.siwe_user_id == uid))
                )
                .scalars()
                .all()
            )
        return wrote_b, wrote_a, state.last_extracted_message_id, texts, total

    try:
        wrote_b, wrote_a, watermark, texts, total = _run(flow)
        assert wrote_b is True
        assert wrote_a is False
        assert watermark == 20  # 没有被回退成 10
        assert texts == ["用户已经在做套利"]  # 旧结论没有把新结论标成 superseded
        assert total == 1  # 被丢弃的那次一条记忆都没留下
    finally:
        _run(lambda: _cleanup(uid))


def test_rejected_commit_writes_nothing_and_releases_its_claim() -> None:
    """条件不匹配就整份丢弃：记忆不写、水位线不动、租约还回去。"""
    uid = _uid()

    async def flow() -> tuple[bool, int, int, str | None]:
        async with pg.session_scope() as db:
            assert await memory_repo.claim(db, uid, "token", 60) is True
        async with pg.session_scope() as db:
            written = await _commit(db, uid, token="token", until=10, watermark=5)
        async with pg.session_scope() as db:
            state = await memory_repo.get_state(db, uid)
            rows = (
                (await db.execute(select(UserMemory).where(UserMemory.siwe_user_id == uid)))
                .scalars()
                .all()
            )
        return written, state.last_extracted_message_id, len(rows), state.claim_token

    try:
        written, watermark, rows, claim = _run(flow)
        assert written is False
        assert watermark == 0
        assert rows == 0
        assert claim is None  # 还回了租约，下一轮不用等过期
    finally:
        _run(lambda: _cleanup(uid))


def test_failure_counter_does_not_recreate_state_after_purge() -> None:
    """失败计数走条件更新：状态行被删掉时不能凭空造一行出来。"""
    uid = _uid()

    async def flow() -> tuple[bool, object]:
        async with pg.session_scope() as db:
            assert await memory_repo.claim(db, uid, "token", 60) is True
            generation = await conversation_repo.current_generation(db, uid)
        async with pg.session_scope() as db:
            await conversation_repo.bump_generation(db, uid)
            await memory_repo.purge(db, uid)
        async with pg.session_scope() as db:
            written = await memory_repo.mark_failed(
                db, uid, claim_token="token", expected_generation=generation
            )
        async with pg.session_scope() as db:
            return written, await memory_repo.get_state(db, uid)

    try:
        written, state = _run(flow)
        assert written is False
        assert state is None
    finally:
        _run(lambda: _cleanup(uid))


def test_collect_hands_out_single_job_under_concurrency() -> None:
    """两台设备同时触发：只有一个后台任务拿到待抽取区间。"""
    uid = _uid()

    async def flow() -> list[bool]:
        async with pg.session_scope() as db:
            db.add_all(
                [
                    ConversationMessage(siwe_user_id=uid, role="user", content=f"m{i}")
                    for i in range(12)
                ]
            )
            await db.flush()
        extractor = MemoryExtractor(get_settings(), None, None, None)  # type: ignore[arg-type]
        jobs = await asyncio.gather(extractor.collect(uid), extractor.collect(uid))
        return [job is not None for job in jobs]

    try:
        assert sorted(_run(flow)) == [False, True]
    finally:
        _run(lambda: _cleanup(uid))
