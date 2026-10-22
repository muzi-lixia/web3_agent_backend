"""工具用 {"error": ...} 表示查不到时，记录必须是 ok=False。

只读工具的错误返回不抛异常（E303 数据源未配置 / E101 地址不合法 / E204 合约不在白名单），
记成 ok=True 会让证据校验以为「查到过」，于是模型编出来的哈希和金额照样能过关。
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.tools import StructuredTool

from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.config.constants import CallSource, ToolGroup


def _runner(payload: dict) -> ToolRunner:
    async def _read(address: str = "") -> dict:
        return payload

    registry = ToolRegistry()
    registry.register(
        StructuredTool.from_function(coroutine=_read, name="fake_read", description="测试用"),
        ToolGroup.LLM_VISIBLE,
        mutating=False,
    )
    return ToolRunner(registry)


def test_error_payload_is_recorded_as_failed() -> None:
    content, record = asyncio.run(
        _runner({"error": "E303", "message": "交易记录数据源未配置"}).run(
            "fake_read", {"address": "0x1"}, source=CallSource.LLM
        )
    )

    assert record.ok is False
    assert record.error == "E303"
    assert json.loads(content)["error"] == "E303"


def test_normal_payload_is_recorded_as_ok() -> None:
    _, record = asyncio.run(
        _runner({"records": []}).run("fake_read", {}, source=CallSource.LLM)
    )

    assert record.ok is True
    assert record.error is None
