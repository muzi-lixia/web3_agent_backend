"""JsonGuard：结构化输出熔断。

只对「模型给了内容但结构不合法」重试一次；仍失败就走纯文本降级——禁止输出地址、金额、动作承诺。
服务不可达、超时、配置错误不在这里重试：那不是格式问题，重试只会再花一次模型调用。
降级文案是固定字符串，不经过模型。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from web3_agent.common.exceptions import ModelBadOutput, ModelUnavailable
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
            except ModelBadOutput as exc:
                last_error = exc
                logger.warning("llm structured output invalid (attempt %s): %s", attempt, exc)
            except ModelUnavailable:
                # 模型不可达/配置错误不是"解析失败"，降级会把运维故障伪装成用户表达问题。
                raise
        logger.error("llm structured output degraded: %s", last_error)
        return GuardResult(output=None, attempts=self._max_attempts, degraded=True)
