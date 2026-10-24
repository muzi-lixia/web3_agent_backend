"""本轮预算：跨 gather 子任务累加、调用前扣减、按资源分类。

子任务是复制上下文，所以计费表必须是可变对象共享引用 —— 存成 int 就不回传。
"""

from __future__ import annotations

import asyncio

from web3_agent.common.metering import Resource, reserve, start_metering, take_usage


def test_usage_made_inside_gather_subtasks_is_counted() -> None:
    async def run() -> int:
        start_metering()

        async def one() -> None:
            reserve(Resource.RPC)
            reserve(Resource.RPC)

        await asyncio.gather(one(), one(), one())
        return take_usage()[Resource.RPC]

    assert asyncio.run(run()) == 6


def test_take_resets_the_ledger() -> None:
    async def run():
        start_metering()
        reserve(Resource.MODEL)
        return take_usage(), take_usage()

    first, second = asyncio.run(run())
    assert first[Resource.MODEL] == 1
    assert set(second.values()) == {0}


def test_reserve_refuses_past_the_limit_and_does_not_count() -> None:
    start_metering(downstream=2)
    assert [reserve(Resource.RPC) for _ in range(3)] == [True, True, False]
    # 被拒的调用不能记进去，否则上限会随着尝试次数继续涨。
    assert take_usage()[Resource.RPC] == 2


def test_downstream_budget_covers_rpc_explorer_and_embedding_together() -> None:
    start_metering(downstream=3)
    assert reserve(Resource.RPC) and reserve(Resource.EXPLORER) and reserve(Resource.EMBEDDING)
    assert not reserve(Resource.RPC)
    usage = take_usage()
    # 上限共用，计数分开：结算时要能分清是链上请求还是向量化。
    assert (usage[Resource.RPC], usage[Resource.EXPLORER], usage[Resource.EMBEDDING]) == (1, 1, 1)


def test_model_calls_are_counted_without_a_cap() -> None:
    start_metering()
    assert all(reserve(Resource.MODEL) for _ in range(20))
    assert take_usage()[Resource.MODEL] == 20


def test_reserving_without_a_ledger_is_a_no_op() -> None:
    take_usage()  # 丢掉上一条用例留在主上下文里的表
    assert reserve(Resource.RPC) is True
    assert set(take_usage().values()) == {0}


def test_embedding_is_refused_once_the_downstream_budget_is_gone() -> None:
    """向量化与重排走同一条下游预算，用完就不再发请求。"""
    import pytest

    from web3_agent.common.exceptions import BudgetExhausted
    from web3_agent.config.settings import Settings
    from web3_agent.rag.embedder import Embedder

    start_metering(downstream=1)
    assert reserve(Resource.RPC)
    with pytest.raises(BudgetExhausted):
        asyncio.run(Embedder(Settings(_env_file=None)).embed("hello"))
    # 被拒的调用没有发出，也没有多记一笔。
    assert take_usage()[Resource.RPC] == 1
