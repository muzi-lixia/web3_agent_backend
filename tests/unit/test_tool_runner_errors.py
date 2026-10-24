"""工具用 {"error": ...} 表示查不到时，记录必须是 ok=False。

只读工具的错误返回不抛异常（E303 数据源未配置 / E101 地址不合法 / E202 合约不在白名单），
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


def test_unexpected_exception_uses_neutral_code_and_hides_internal_message():
    async def run():
        async def fail() -> dict:
            raise RuntimeError('private RPC URL with API key')
        registry=ToolRegistry()
        registry.register(StructuredTool.from_function(coroutine=fail,name='get_asset_balance',description='test'),
                          ToolGroup.LLM_VISIBLE,mutating=False)
        content,record=await ToolRunner(registry).run('get_asset_balance',{},source=CallSource.LLM)
        assert json.loads(content)['error']=='E304' and record.error=='E304'
        assert 'private RPC' not in content and '交易记录' not in content
    asyncio.run(run())


def test_actionable_app_error_reaches_api_without_recoding():
    import pytest
    from web3_agent.common.exceptions import AppError,error_tuple
    async def run():
        async def fail() -> dict:
            raise AppError(*error_tuple('E613'))
        registry=ToolRegistry()
        registry.register(StructuredTool.from_function(coroutine=fail,name='rag_general_knowledge',description='test'),
                          ToolGroup.LLM_VISIBLE,mutating=False)
        with pytest.raises(AppError) as error:
            await ToolRunner(registry).run('rag_general_knowledge',{},source=CallSource.LLM)
        assert error.value.code=='E613'
    asyncio.run(run())


def test_graph_does_not_swallow_actionable_tool_error():
    import pytest
    from web3_agent.agent.graph.read_graph import build_read_graph,initial_state
    from web3_agent.config.constants import Intent
    from web3_agent.common.exceptions import AppError,error_tuple
    class Runner:
        async def run(self,*args,**kwargs):
            raise AppError(*error_tuple('E611'))
    async def run():
        graph=build_read_graph(None,Runner(),5,rag_answer_model=None)
        with pytest.raises(AppError) as error:
            await graph.ainvoke(initial_state([],'什么是DEX',intent=Intent.QUERY_KNOWLEDGE))
        assert error.value.code=='E611'
    asyncio.run(run())
