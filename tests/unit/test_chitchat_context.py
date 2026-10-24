"""闲聊也要看得到摘要和长期约束。

用户问「之前说的那个计划」，内容多半已经滚出近期窗口。摘要和硬约束只喂给意图识别、
不喂回答模型时，系统能正确路由到闲聊，却只能回一句不知道。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from web3_agent.agent.agent_loop import AgentLoop


def _loop(reply: str = "好的"):
    answer = AsyncMock(return_value=SimpleNamespace(content=reply))
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm = SimpleNamespace(answer=answer)
    loop._audit = SimpleNamespace(log=AsyncMock())
    return loop, answer


def _text(answer: AsyncMock) -> str:
    messages = answer.await_args.args[0]
    return "\n".join(str(message.content) for message in messages)


def test_chitchat_includes_summary_and_constraints() -> None:
    async def run() -> None:
        loop, answer = _loop()
        await loop._chitchat(
            [("user", "帮我看看")],
            "之前说的那个计划是什么",
            "u-1",
            profile_text="用户偏好：回答简短",
            memory_text="长期约束：不要推荐借贷",
            summary_text="历史摘要：用户正在规划质押退出，分三步",
        )
        text = _text(answer)
        assert "用户偏好：回答简短" in text
        assert "不要推荐借贷" in text
        assert "分三步" in text

    asyncio.run(run())


def test_chitchat_stays_minimal_without_extra_context() -> None:
    async def run() -> None:
        loop, answer = _loop()
        await loop._chitchat([("user", "你好")], "在吗", "u-1")
        messages = answer.await_args.args[0]
        # 系统提示 + 一条历史 + 本轮输入，没有内容就不塞空消息
        assert len(messages) == 3

    asyncio.run(run())
