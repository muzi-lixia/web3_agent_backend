"""模型异常只在适配边界归一化一次，调用点不再各写 except。"""
import asyncio

import httpx
import pytest

from web3_agent.agent.json_guard import JsonGuard
from web3_agent.agent.llm_adapter import _GuardedModel
from web3_agent.common.exceptions import ModelBadOutput, ModelConfigError, ModelUnavailable


class _Boom:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def ainvoke(self, messages, config=None, **kwargs):
        raise self._exc

    def invoke(self, messages, config=None, **kwargs):
        raise self._exc


def _guarded(exc: Exception, stage: str = "回答") -> _GuardedModel:
    return _GuardedModel(_Boom(exc), stage)


def test_transport_failure_becomes_model_unavailable():
    with pytest.raises(ModelUnavailable):
        asyncio.run(_guarded(httpx.ConnectError("down")).ainvoke([]))


def test_timeout_becomes_model_unavailable():
    with pytest.raises(ModelUnavailable):
        asyncio.run(_guarded(TimeoutError("slow")).ainvoke([]))


def test_invalid_payload_becomes_bad_output():
    """模型返回了内容但字段不对：这一类才值得走格式修复。"""
    with pytest.raises(ModelBadOutput):
        asyncio.run(_guarded(ValueError("bad json"), "意图抽取").ainvoke([]))


def test_config_error_is_reported_separately_from_transport():
    error = type("ResponseError", (Exception,), {})("model not found")
    error.status_code = 404
    with pytest.raises(ModelConfigError):
        asyncio.run(_guarded(error).ainvoke([]))


def test_unknown_exception_never_leaks_as_generic_error():
    with pytest.raises(ModelUnavailable):
        asyncio.run(_guarded(RuntimeError("weird")).ainvoke([]))


def test_guard_retries_bad_output_then_degrades():
    calls = []

    async def call():
        calls.append(1)
        raise ModelBadOutput("bad")

    result = asyncio.run(JsonGuard(max_attempts=2).run(call))
    assert result.degraded and len(calls) == 2


def test_guard_does_not_retry_a_transport_failure():
    calls = []

    async def call():
        calls.append(1)
        raise ModelUnavailable("down")

    with pytest.raises(ModelUnavailable):
        asyncio.run(JsonGuard(max_attempts=2).run(call))
    assert len(calls) == 1


def test_config_error_is_unavailable_subclass_so_callers_keep_working():
    assert issubclass(ModelConfigError, ModelUnavailable)
