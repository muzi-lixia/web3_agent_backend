"""ToolRunner：工具调用的唯一执行入口。

两件事：来源校验（LLM 不得触发 mutating 工具）、调用记录（耗时 / 成败）。
"""

from __future__ import annotations

import json
import time
import logging
from pydantic import ValidationError

from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.common.exceptions import ToolSourceViolation, AppError, error_message
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import CallSource


logger = logging.getLogger(__name__)


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
        except AppError:
            # Preserve actionable API failures (KB disabled/configuration errors).
            raise
        except Exception as exc:  # noqa: BLE001 unexpected exceptions never enter model context
            logger.warning("tool execution failed tool=%s", name, exc_info=True)
            code = "E101" if isinstance(exc, ValidationError) else "E304"
            record = ToolCallRecord(
                name=name, args=args, ok=False, error=code,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            return json.dumps({"error": code, "message": error_message(code)}, ensure_ascii=False), record

        payload = result if isinstance(result, (dict, list)) else {"result": result}
        # 只读工具用 {"error": "E303"} 表示查不到（数据源未配置 / 地址不合法 / 无 RPC）。
        # 这种返回没抛异常，但确实没有数据 —— 记成 ok 会让证据校验以为「查到过」。
        failed = isinstance(payload, dict) and bool(payload.get("error"))
        record = ToolCallRecord(
            name=name,
            args=args,
            ok=not failed,
            error=str(payload["error"]) if failed else None,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return json.dumps(payload, ensure_ascii=False, default=str), record
