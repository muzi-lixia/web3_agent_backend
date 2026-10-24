"""召回分支的预算与回收。

三件事必须分开成立：分支超时只丢自己那一路、父任务被取消要把子任务收回来、
分支异常不能被吞成空结果。另外关键词准备再慢也不能把向量分支的预算吃光。
"""

from __future__ import annotations

import asyncio
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from web3_agent.config.settings import Settings
from web3_agent.rag.models import Citation
from web3_agent.rag.retriever import KnowledgeRetriever, _settle


def _chunk(key: str, text: str) -> Citation:
    return Citation(id="S1", chunk_id=key, title="FAQ", section="", source_url="",
                    version="v1", doc_category="faq", updated_at="2026-01-01",
                    content=text, distance=2)


def test_stage_timeout_keeps_finished_branches() -> None:
    async def run() -> None:
        async def fast():
            return ["kept"]

        async def hanging():
            await asyncio.sleep(10)
            return ["dropped"]

        values, pending = await _settle(
            [asyncio.ensure_future(fast()), asyncio.ensure_future(hanging())],
            monotonic() + 0.05,
        )
        # 慢分支超时只丢自己的结果，已经算出来的那一路照常交出。
        assert values[0] == ["kept"]
        assert values[1] is None
        assert len(pending) == 1

    asyncio.run(run())


def test_parent_cancel_reclaims_running_branches() -> None:
    async def run() -> None:
        marks: list[str] = []

        async def hanging():
            try:
                await asyncio.sleep(10)
                marks.append("finished")
            finally:
                marks.append("cancelled")
            return []

        async def parent():
            branch = asyncio.ensure_future(hanging())
            await _settle([branch], monotonic() + 30)
            return branch

        task = asyncio.ensure_future(parent())
        await asyncio.sleep(0.02)  # 让分支真正进到 sleep 里
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        # 客户端断开后分支不能继续跑完：那会脱离本轮结算继续打 Embedding 和 SQL。
        assert "finished" not in marks
        assert "cancelled" in marks

    asyncio.run(run())


def test_branch_error_is_not_swallowed() -> None:
    async def run() -> None:
        async def boom():
            raise RuntimeError("sql down")

        with pytest.raises(RuntimeError):
            await _settle([asyncio.ensure_future(boom())], monotonic() + 5)

    asyncio.run(run())


def test_slow_keyword_snapshot_does_not_starve_dense_recall() -> None:
    async def run() -> None:
        item = _chunk("dex", "DEX is a decentralized exchange.")

        async def slow_snapshot(**kwargs):
            await asyncio.sleep(5)
            return [item]

        store = SimpleNamespace(
            has_documents=AsyncMock(return_value=True),
            index_state=AsyncMock(return_value=(1, None)),
            keyword_snapshot=slow_snapshot,
            search=AsyncMock(return_value=[item]),
        )
        retriever = KnowledgeRetriever(
            Settings(_env_file=None, rag_enabled=True, rag_hybrid_enabled=True,
                     rag_rewrite_enabled=False, rag_rerank_enabled=False,
                     rag_timeout_seconds=1.0),
            SimpleNamespace(embed=AsyncMock(return_value=[1.0])),
            store,
        )
        result = await retriever.search("DEX")
        # 关键词快照拖到超时也必须让向量跑过：降级路径不能只是写在日志里。
        assert store.search.await_count == 1
        assert result["status"] == "success"
        assert "keyword" in result["retrieval"]["degraded_stages"]

    asyncio.run(run())
