"""conversation_message 的工具轨迹落库。不碰模型，只验读写。"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.repositories import conversation_repo

CALLS = [
    {
        "name": "get_asset_balance",
        "args": {"address": "0xabc", "topN": 10},
        "ok": True,
        "error": None,
        "latency_ms": 3820,
    }
]


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


def test_append_message_persists_tool_calls() -> None:
    """工具轨迹跟着 assistant 消息走；没调工具的那轮保持 NULL，不写空数组。"""
    uid = _uid()

    async def flow() -> list:
        async with pg.session_scope() as db:
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="user", content="帮我看看 BERA 余额"
            )
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="assistant", content="余额 0", tool_calls=CALLS
            )
            await conversation_repo.append_message(
                db, siwe_user_id=uid, role="assistant", content="你好"
            )
        async with pg.session_scope() as db:
            return await conversation_repo.list_page(db, uid, 10)

    async def cleanup() -> None:
        async with pg.session_scope() as db:
            await conversation_repo.clear_user_messages(db, uid)

    try:
        rows = _run(flow)
    finally:
        _run(cleanup)
    assert [r.role for r in rows] == ["assistant", "assistant", "user"]  # list_page 倒序
    assert rows[0].tool_calls is None
    assert rows[1].tool_calls == CALLS
    assert rows[2].tool_calls is None


def test_append_message_persists_task_and_preview() -> None:
    """确认卡片的两个凭据跟消息一起落库，刷新后前端才能接着确认。"""
    uid = _uid()
    preview = {
        "action": "transfer",
        "tokenSymbol": "BERA",
        "humanAmount": "0.1",
        "target": "0x" + "2" * 40,
    }

    async def flow() -> list:
        async with pg.session_scope() as db:
            await conversation_repo.append_message(
                db,
                siwe_user_id=uid,
                role="assistant",
                content="请确认转账：0.1 BERA。",
                task_id="task-1",
                intent_preview=preview,
            )
        async with pg.session_scope() as db:
            return await conversation_repo.list_page(db, uid, 10)

    async def cleanup() -> None:
        async with pg.session_scope() as db:
            await conversation_repo.clear_user_messages(db, uid)

    try:
        rows = _run(flow)
    finally:
        _run(cleanup)
    assert rows[0].task_id == "task-1"
    assert rows[0].intent_preview == preview
