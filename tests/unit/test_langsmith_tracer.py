"""LangSmith 开关：关闭时业务侧拿不到任何配置，等于这段代码不存在。"""

from __future__ import annotations

import os

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


def test_root_extra_carries_run_id_and_filters_none() -> None:
    try:
        langsmith_tracer.configure(_settings(langsmith_tracing=True, langsmith_api_key="k"))
        extra = langsmith_tracer.root_extra(trace_id="t-1", chain_id=1, siwe_user_id=None)
        assert extra["run_id"]
        assert extra["metadata"] == {"trace_id": "t-1", "chain_id": 1}
        assert "run_name" not in extra  # 名字由 @traceable 决定，不能混进 extra
    finally:
        langsmith_tracer.configure(_settings())
    assert langsmith_tracer.root_extra(trace_id="t-1") == {}


def test_heartbeat_probes_api_without_creating_a_run(monkeypatch) -> None:
    """心跳只探出口与凭证，不写 trace：否则每次启动、每小时都往项目里塞一条空 run。"""

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def list_projects(self, **_: object) -> list:
            self.calls += 1
            return []

    fake = FakeClient()
    monkeypatch.setattr(langsmith_tracer, "_probe", lambda: fake)
    try:
        langsmith_tracer.configure(_settings(langsmith_tracing=True, langsmith_api_key="k"))
        assert langsmith_tracer.heartbeat() is True
        assert fake.calls == 1
    finally:
        langsmith_tracer.configure(_settings())


def test_heartbeat_reports_false_when_probe_fails(monkeypatch) -> None:
    class BrokenClient:
        def list_projects(self, **_: object) -> list:
            raise RuntimeError("401")

    monkeypatch.setattr(langsmith_tracer, "_probe", lambda: BrokenClient())
    try:
        langsmith_tracer.configure(_settings(langsmith_tracing=True, langsmith_api_key="k"))
        assert langsmith_tracer.heartbeat() is False
    finally:
        langsmith_tracer.configure(_settings())


def test_project_and_switch_land_in_env() -> None:
    """configure() 的 project 是 ContextVar，只在调用它的上下文可见。

    LangChain 在请求上下文里读不到，就退回环境变量；环境变量也没有，
    run 会全部落到 default 项目 —— 项目面板看起来像一条都没上报。
    """
    try:
        langsmith_tracer.configure(_settings(langsmith_tracing=True, langsmith_api_key="k"))
        assert os.environ["LANGSMITH_PROJECT"] == "test"
        assert os.environ["LANGSMITH_TRACING"] == "true"
    finally:
        langsmith_tracer.configure(_settings())
    assert os.environ["LANGSMITH_TRACING"] == "false"
