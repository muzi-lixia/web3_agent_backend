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


# 数据职责：结构化意图调用的最终结果，包含实际尝试次数和降级标志。
@dataclass(frozen=True)
class GuardResult:
    # output：通过结构化校验的意图结果；失败降级时为空。
    output: LlmOutput | None
    # attempts：本轮结构化抽取实际尝试次数，包含首次调用。
    attempts: int
    # degraded：是否已耗尽格式重试并放弃结构化意图结果。
    degraded: bool


# 数据职责：只重试模型输出格式错误的有限重试保护器。
class JsonGuard:
    def __init__(self, max_attempts: int = 2) -> None:
        """设置结构化输出允许的最大尝试次数。

        Args:
            max_attempts: 最大执行或重试次数，具体以当前流程为准。
        """
        self._max_attempts = max_attempts

    async def run(self, call: Callable[[], Awaitable[LlmOutput]]) -> GuardResult:
        """在限定次数内尝试结构化意图调用；只重试输出格式错误，耗尽后返回降级结果。

        Args:
            call: 工具调用描述，或不带参数的可等待模型调用函数。
        """
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
