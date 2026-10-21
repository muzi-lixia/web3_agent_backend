"""档案仓储：补丁合并不覆盖、代币去重按时间排序、清空会话不影响档案。走真实 PG。"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.domain.memory.profile import ProfilePatch
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.repositories import conversation_repo, profile_repo, summary_repo


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

    from web3_agent.persistence.pg.base import session_scope
    from web3_agent.persistence.pg.orm_models import UserProfile

    async with session_scope() as db:
        await db.execute(delete(UserProfile).where(UserProfile.siwe_user_id == uid))


def test_empty_patch_creates_nothing() -> None:
    uid = _uid()

    async def flow() -> tuple[bool, str | None]:
        async with pg.session_scope() as db:
            current = await profile_repo.apply_patch(
                db, siwe_user_id=uid, patch=ProfilePatch(), chain_id=1
            )
        async with pg.session_scope() as db:
            row = await profile_repo.get(db, uid)
        return current.answer_style is None, row.answer_style

    try:
        empty, stored = _run(flow)
        assert empty is True
        assert stored is None
    finally:
        _run(lambda: _cleanup(uid))


def test_patch_merges_without_wiping_other_fields() -> None:
    uid = _uid()

    async def flow() -> tuple[str | None, str | None]:
        async with pg.session_scope() as db:
            await profile_repo.apply_patch(
                db,
                siwe_user_id=uid,
                patch=ProfilePatch(answer_style="concise"),
                chain_id=1,
            )
        async with pg.session_scope() as db:
            await profile_repo.apply_patch(
                db, siwe_user_id=uid, patch=ProfilePatch(language="en"), chain_id=1
            )
        async with pg.session_scope() as db:
            row = await profile_repo.get(db, uid)
        return row.answer_style, row.language

    try:
        style, language = _run(flow)
        assert style == "concise"  # 第二次只说语言，风格不能被清掉
        assert language == "en"
    finally:
        _run(lambda: _cleanup(uid))


def test_token_bump_is_deduped_and_ordered() -> None:
    uid = _uid()

    async def flow() -> list[str]:
        async with pg.session_scope() as db:
            await profile_repo.apply_patch(
                db,
                siwe_user_id=uid,
                patch=ProfilePatch(tokens=("BERA",)),
                chain_id=80094,
            )
        async with pg.session_scope() as db:
            await profile_repo.apply_patch(
                db,
                siwe_user_id=uid,
                patch=ProfilePatch(tokens=("USDT",)),
                chain_id=80094,
            )
        async with pg.session_scope() as db:
            await profile_repo.apply_patch(
                db,
                siwe_user_id=uid,
                patch=ProfilePatch(tokens=("BERA",)),
                chain_id=80094,
            )
        async with pg.session_scope() as db:
            row = await profile_repo.get(db, uid)
        return [t["symbol"] for t in row.frequent_tokens]

    try:
        symbols = _run(flow)
        assert symbols == ["BERA", "USDT"]  # 重复的不追加，最近用的排前面
    finally:
        _run(lambda: _cleanup(uid))


def test_token_without_chain_is_dropped() -> None:
    uid = _uid()

    async def flow() -> tuple:
        async with pg.session_scope() as db:
            row = await profile_repo.apply_patch(
                db,
                siwe_user_id=uid,
                patch=ProfilePatch(tokens=("BERA",)),
                chain_id=None,
            )
        return row.frequent_tokens

    try:
        assert _run(flow) == ()
    finally:
        _run(lambda: _cleanup(uid))


def test_clear_keeps_profile() -> None:
    uid = _uid()

    async def flow() -> tuple[str | None, int]:
        async with pg.session_scope() as db:
            await profile_repo.apply_patch(
                db,
                siwe_user_id=uid,
                patch=ProfilePatch(answer_style="concise"),
                chain_id=1,
            )
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="user", content="hi"
            )
        async with pg.session_scope() as db:
            await conversation_repo.clear_user_messages(db, uid)
            await summary_repo.delete(db, uid)
        async with pg.session_scope() as db:
            row = await profile_repo.get(db, uid)
            left = await conversation_repo.list_page(db, uid, 10)
        return row.answer_style, len(left)

    try:
        style, remaining = _run(flow)
        assert remaining == 0
        assert style == "concise"
    finally:
        _run(lambda: _cleanup(uid))
