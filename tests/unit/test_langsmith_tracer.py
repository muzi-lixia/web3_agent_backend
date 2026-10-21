"""LangSmith 开关：关闭时业务侧拿不到任何配置，等于这段代码不存在。"""

from __future__ import annotations

from web3_agent.agent import langsmith_tracer
from web3_agent.config.settings import Settings

_UNREACHABLE = "http://127.0.0.1:1"


def _settings(**overrides) -> Settings:
    base = {
        "langsmith_tracing": False,
        "langsmith_api_key": "",
        "langsmith_project": "test",
        "langsmith_endpoint": _UNREACHABLE,
    }
    base.update(overrides)
    return Settings(**base)


def test_disabled_when_switch_is_off() -> None:
    assert langsmith_tracer.configure(_settings(langsmith_api_key="k")) is False
    assert langsmith_tracer.enabled() is False
    assert langsmith_tracer.run_config("chat.turn", trace_id="t-1") == {}


def test_disabled_when_key_is_missing() -> None:
    assert langsmith_tracer.configure(_settings(langsmith_tracing=True)) is False
    assert langsmith_tracer.run_config("chat.turn", trace_id="t-1") == {}


def test_enabled_builds_run_config_with_trace_id() -> None:
    try:
        assert (
            langsmith_tracer.configure(_settings(langsmith_tracing=True, langsmith_api_key="k"))
            is True
        )
        config = langsmith_tracer.run_config("chat.turn", trace_id="t-1", chain_id=1, wallet=None)
        assert config["run_name"] == "chat.turn"
        assert config["metadata"] == {"trace_id": "t-1", "chain_id": 1}
        assert config["run_id"]
    finally:
        langsmith_tracer.configure(_settings())


def test_heartbeat_is_noop_when_disabled() -> None:
    langsmith_tracer.configure(_settings())
    assert langsmith_tracer.heartbeat() is False
