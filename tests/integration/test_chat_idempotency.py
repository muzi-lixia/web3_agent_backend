"""聊天幂等键：同一 (用户, key) 只能跑一轮，重试取回原响应。

必须用真库：去重靠主键冲突，假 session 验不出并发时只有一方插入成功。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import ChatIdempotency
from web3_agent.persistence.pg.repositories import idempotency_repo
from sqlalchemy import delete, select, update


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


async def _cleanup(user: str) -> None:
    async with pg.session_scope() as db:
        await db.execute(delete(ChatIdempotency).where(ChatIdempotency.siwe_user_id == user))


async def _claim(user, key, request_hash="h1", stale_after=120):
    async with pg.session_scope() as db:
        return await idempotency_repo.claim(
            db, siwe_user_id=user, key=key, request_hash=request_hash,
            stale_after=stale_after,
        )


def test_same_key_replays_stored_response() -> None:
    user, key = _uid(), "key-replay-0001"

    async def flow():
        assert await _claim(user, key) == ("new", None)
        async with pg.session_scope() as db:
            await idempotency_repo.complete(
                db, siwe_user_id=user, key=key, response={"content": "查完了"}
            )
        return await _claim(user, key)

    try:
        assert _run(flow) == ("done", {"content": "查完了"})
    finally:
        _run(lambda: _cleanup(user))


def test_same_key_with_new_content_is_rejected() -> None:
    """拿旧答案顶新问题比报错更糟：用户会以为消息丢了。"""

    async def flow():
        user, key = _uid(), "key-conflict-0001"
        assert await _claim(user, key, "hash-a") == ("new", None)
        return await _claim(user, key, "hash-b"), user

    state, user = _run(flow)
    try:
        assert state == ("conflict", None)
    finally:
        _run(lambda: _cleanup(user))


def test_concurrent_requests_only_one_runs() -> None:
    user, key = _uid(), "key-concurrent-1"

    async def flow():
        async def take():
            return await _claim(user, key)

        return await asyncio.gather(take(), take())

    try:
        # 谁抢到不一定，但绝不能两个都拿到 new —— 那就是两轮对话都执行了。
        states = sorted(state for state, _ in _run(flow))
        assert states == ["new", "running"]
    finally:
        _run(lambda: _cleanup(user))


def test_failed_turn_releases_the_slot() -> None:
    """一轮跑挂了必须放行，否则同一个 key 在一天内都重试不了。"""

    async def flow():
        user, key = _uid(), "key-release-0001"
        assert await _claim(user, key) == ("new", None)
        async with pg.session_scope() as db:
            await idempotency_repo.release(db, siwe_user_id=user, key=key)
        return await _claim(user, key), user

    result, user = _run(flow)
    try:
        assert result == ("new", None)
    finally:
        _run(lambda: _cleanup(user))


def test_release_keeps_completed_response() -> None:
    """已经写回的响应不能被迟到的 release 删掉。"""

    async def flow():
        user, key = _uid(), "key-keep-0001"
        assert await _claim(user, key) == ("new", None)
        async with pg.session_scope() as db:
            await idempotency_repo.complete(
                db, siwe_user_id=user, key=key, response={"content": "done"}
            )
        async with pg.session_scope() as db:
            await idempotency_repo.release(db, siwe_user_id=user, key=key)
        return await _claim(user, key), user

    result, user = _run(flow)
    try:
        assert result == ("done", {"content": "done"})
    finally:
        _run(lambda: _cleanup(user))


def test_expired_keys_are_removed_on_next_claim() -> None:
    user = _uid()

    async def flow():
        stale, fresh = datetime.now(timezone.utc) - timedelta(
            seconds=idempotency_repo.KEEP_SECONDS + 60
        ), datetime.now(timezone.utc)
        async with pg.session_scope() as db:
            db.add(
                ChatIdempotency(
                    siwe_user_id=user,
                    idempotency_key="key-stale-00001",
                    request_hash="h",
                    status="done",
                    response={"content": "old"},
                    created_at=stale,
                )
            )
            db.add(
                ChatIdempotency(
                    siwe_user_id=user,
                    idempotency_key="key-fresh-00001",
                    request_hash="h",
                    status="done",
                    response={"content": "new"},
                    created_at=fresh,
                )
            )
        assert await _claim(user, "key-other-00001") == ("new", None)
        async with pg.session_scope() as db:
            keys = (
                await db.execute(
                    select(ChatIdempotency.idempotency_key).where(
                        ChatIdempotency.siwe_user_id == user
                    )
                )
            ).scalars().all()
        return set(keys)

    try:
        assert _run(flow) == {"key-fresh-00001", "key-other-00001"}
    finally:
        _run(lambda: _cleanup(user))


def test_orphaned_running_slot_is_reclaimed() -> None:
    """进程在写回前被杀掉：占位不能把同一个 key 锁死到第二天。"""

    async def flow():
        user, key = _uid(), "key-orphan-0001"
        assert await _claim(user, key) == ("new", None)
        async with pg.session_scope() as db:
            await db.execute(
                update(ChatIdempotency)
                .where(ChatIdempotency.siwe_user_id == user)
                .values(created_at=datetime.now(timezone.utc) - timedelta(seconds=600))
            )
        return await _claim(user, key), user

    result, user = _run(flow)
    try:
        assert result == ("new", None)
    finally:
        _run(lambda: _cleanup(user))


def test_live_slot_is_not_stolen() -> None:
    """真正在跑的那一轮不能被重试接管，否则两轮同时执行。"""

    async def flow():
        user, key = _uid(), "key-live-0001"
        assert await _claim(user, key) == ("new", None)
        return await _claim(user, key), user

    result, user = _run(flow)
    try:
        assert result == ("running", None)
    finally:
        _run(lambda: _cleanup(user))


def test_purge_invalidates_cached_answers() -> None:
    """清空/删除后拿同一个 key 重试，不能取回删除前那条答案。"""

    async def flow():
        user, key = _uid(), "key-purge-00001"
        assert await _claim(user, key) == ("new", None)
        async with pg.session_scope() as db:
            await idempotency_repo.complete(
                db, siwe_user_id=user, key=key, response={"content": "删除前的答案"}
            )
        assert await _claim(user, key) == ("done", {"content": "删除前的答案"})
        async with pg.session_scope() as db:
            await idempotency_repo.purge_user(db, siwe_user_id=user)
        return await _claim(user, key), user

    result, user = _run(flow)
    try:
        assert result == ("new", None)
    finally:
        _run(lambda: _cleanup(user))


def test_purge_only_touches_its_own_user() -> None:
    """别人的幂等记录不能跟着一起没。"""

    async def flow():
        mine, other = _uid(), _uid()
        assert await _claim(mine, "key-shared-0001") == ("new", None)
        assert await _claim(other, "key-shared-0001") == ("new", None)
        async with pg.session_scope() as db:
            await idempotency_repo.purge_user(db, siwe_user_id=mine)
        return await _claim(other, "key-shared-0001"), mine, other

    result, mine, other = _run(flow)
    try:
        assert result == ("running", None)
    finally:
        _run(lambda: _cleanup(mine))
        _run(lambda: _cleanup(other))
