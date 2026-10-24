"""结算边界：按已经发生的消耗记账，不看这一轮有没有成功返回。

原来的记账发生在 agent.handle() 返回之后，中途抛错或超时，前面打出去的
模型调用和下游请求就全部白花。这里守住 finally 里那一次结算。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from web3_agent.api.v1 import chat_router
from web3_agent.config.constants import QuotaMetric
from web3_agent.common.metering import Resource, reserve, start_metering


def _container(charge):
    return SimpleNamespace(quota=SimpleNamespace(charge=AsyncMock(side_effect=charge)))


def _session_scope(monkeypatch):
    @asynccontextmanager
    async def scope():
        yield object()

    monkeypatch.setattr(chat_router, "session_scope", scope)


def test_settlement_charges_recorded_usage_once(monkeypatch):
    _session_scope(monkeypatch)
    charges = []

    async def charge(db, *, key, metric, amount):
        charges.append((key, metric, amount))

    start_metering(downstream=4)
    reserve(Resource.MODEL)
    reserve(Resource.MODEL)
    reserve(Resource.RPC)
    reserve(Resource.EXPLORER)
    reserve(Resource.EMBEDDING)

    asyncio.run(chat_router._settle(_container(charge), quota_key="u:1"))
    # 结算过一次就把表清掉，不能重复计费。
    asyncio.run(chat_router._settle(_container(charge), quota_key="u:1"))

    assert charges == [
        ("u:1", QuotaMetric.STEPS, 2),
        # 区块浏览器与向量化都算进下游总量：同样是花钱打出去的外部请求。
        ("u:1", QuotaMetric.RPC_CALLS, 3),
    ]


def test_settlement_failure_does_not_mask_the_turn(monkeypatch):
    _session_scope(monkeypatch)

    async def charge(*args, **kwargs):
        raise RuntimeError("db down")

    start_metering()
    reserve(Resource.MODEL)

    # 结算自己失败时只能记日志：这一轮的答案已经生成，不能因为记账把它变成 500。
    asyncio.run(chat_router._settle(_container(charge), quota_key="u:1"))


def test_no_usage_or_no_key_opens_no_transaction(monkeypatch):
    monkeypatch.setattr(
        chat_router, "session_scope", lambda: (_ for _ in ()).throw(AssertionError("不应开事务"))
    )
    charge = AsyncMock()
    start_metering()
    asyncio.run(chat_router._settle(_container(charge), quota_key="u:1"))
    start_metering()
    reserve(Resource.MODEL)
    asyncio.run(chat_router._settle(_container(charge), quota_key=None))
    charge.assert_not_awaited()
