"""模型异常只在适配边界归一化一次，调用点不再各写 except。"""
import asyncio

import httpx
import openai
import pytest

from web3_agent.agent.json_guard import JsonGuard
from web3_agent.common.exceptions import (
    ModelBadOutput,
    ModelConfigError,
    ModelUnavailable,
)
from web3_agent.config.settings import Settings
from web3_agent.llm.client import GuardedModel, _build_chat_model, normalize_model_error
from web3_agent.rag.embedder import Embedder, _build_embeddings


class _Boom:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def ainvoke(self, messages, config=None, **kwargs):
        raise self._exc

    def invoke(self, messages, config=None, **kwargs):
        raise self._exc


def _guarded(exc: Exception, stage: str = "回答") -> GuardedModel:
    return GuardedModel(_Boom(exc), stage)


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


def test_sdk_transport_error_is_classified_as_transport():
    exc = openai.APIConnectionError(request=httpx.Request("POST", "http://model"))
    normalized = normalize_model_error(exc, "回答")
    assert isinstance(normalized, ModelUnavailable)
    assert not isinstance(normalized, ModelConfigError)


def test_online_embedding_sdk_error_is_normalized():
    class _Boom:
        async def aembed_documents(self, texts):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://model"))

    embedder = Embedder(Settings(_env_file=None))
    embedder._model = _Boom()
    with pytest.raises(ModelUnavailable):
        asyncio.run(embedder.embed("hello"))


def test_model_clients_do_not_retry_implicitly():
    settings = Settings(_env_file=None, model_provider="openai_compat", model_api_key="k",
                        embed_provider="openai_compat", embed_api_key="k",
                        model_base_url="http://model", embed_base_url="http://model")
    assert _build_chat_model(settings).max_retries == 0
    assert _build_embeddings(settings).max_retries == 0
