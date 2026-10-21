"""JsonGuard：结构化输出熔断。

第一次解析失败重试一次；仍失败就走纯文本降级——禁止输出地址、金额、动作承诺。
降级文案是固定字符串，不经过模型。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from web3_agent.common.exceptions import ModelUnavailable
from web3_agent.common.types import LlmOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardResult:
    output: LlmOutput | None
    attempts: int
    degraded: bool


class JsonGuard:
    def __init__(self, max_attempts: int = 2) -> None:
        self._max_attempts = max_attempts

    async def run(self, call: Callable[[], Awaitable[LlmOutput]]) -> GuardResult:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return GuardResult(output=await call(), attempts=attempt, degraded=False)
            except ModelUnavailable:
                # 模型不可达不是"解析失败"，降级会把运维故障伪装成用户表达问题。
                raise
            except Exception as exc:  # noqa: BLE001 解析失败与模型抖动都走同一条降级
                last_error = exc
                logger.warning("llm structured output failed (attempt %s): %s", attempt, exc)
        logger.error("llm structured output degraded: %s", last_error)
        return GuardResult(output=None, attempts=self._max_attempts, degraded=True)
