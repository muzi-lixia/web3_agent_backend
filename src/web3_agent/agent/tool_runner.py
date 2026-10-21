"""ToolRunner：工具调用的唯一执行入口。

两件事：来源校验（LLM 不得触发 mutating 工具）、调用记录（耗时 / 成败）。
"""

from __future__ import annotations

import json
import time

from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.common.exceptions import ToolSourceViolation
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import CallSource


class ToolRunner:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def run(
        self, name: str, args: dict, *, source: CallSource
    ) -> tuple[str, ToolCallRecord]:
        spec = self._registry.get(name)
        if spec.mutating and source is not CallSource.STATE_MACHINE:
            raise ToolSourceViolation(f"{name} 只允许状态机调用")

        started = time.monotonic()
        try:
            result = await spec.tool.ainvoke(args)
        except Exception as exc:  # noqa: BLE001 工具异常要变成可回灌的观察结果
            record = ToolCallRecord(
                name=name,
                args=args,
                ok=False,
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            return json.dumps({"error": "E302", "message": str(exc)}, ensure_ascii=False), record

        record = ToolCallRecord(
            name=name,
            args=args,
            ok=True,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        payload = result if isinstance(result, (dict, list)) else {"result": result}
        return json.dumps(payload, ensure_ascii=False, default=str), record
