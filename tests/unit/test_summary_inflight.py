"""摘要去重：同一用户同一时刻只跑一份。

两轮请求同时跨过触发线时都会派后台任务，水位线乐观锁只能让后一份写回作废，
模型调用已经花掉了 —— 要在入口就挡住。
"""

from __future__ import annotations

import asyncio

from web3_agent.config.settings import Settings
from web3_agent.domain.memory.summarizer import Summarizer


def _summarizer(seen: list[str], delay: float = 0.05) -> Summarizer:
    summarizer = Summarizer(Settings(_env_file=None), llm=None, audit=None, quota=None)

    async def collect(siwe_user_id: str):
        return object()

    async def summarize(siwe_user_id: str, own_address: str, job) -> None:
        seen.append(siwe_user_id)
        await asyncio.sleep(delay)

    summarizer.collect = collect
    summarizer._summarize = summarize
    return summarizer


def test_same_user_runs_one_summary_at_a_time() -> None:
    seen: list[str] = []
    summarizer = _summarizer(seen)

    async def run() -> None:
        await asyncio.gather(
            summarizer.run(siwe_user_id="u1", own_address="0x1"),
            summarizer.run(siwe_user_id="u1", own_address="0x1"),
        )

    asyncio.run(run())

    assert seen == ["u1"]


def test_other_users_are_not_blocked_and_the_slot_is_released() -> None:
    seen: list[str] = []
    summarizer = _summarizer(seen)

    async def run() -> None:
        await asyncio.gather(
            summarizer.run(siwe_user_id="u1", own_address="0x1"),
            summarizer.run(siwe_user_id="u2", own_address="0x2"),
        )
        # 上一轮结束后同一个用户要能再次触发。
        await summarizer.run(siwe_user_id="u1", own_address="0x1")

    asyncio.run(run())

    assert seen == ["u1", "u2", "u1"]
