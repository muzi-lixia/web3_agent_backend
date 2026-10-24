"""日志与 trace_id 的绑定：日志里没有 trace_id，审计表里的那条记录就无从对应。"""

from __future__ import annotations

import logging

from web3_agent.common.logging_setup import TraceIdFilter, setup_logging
from web3_agent.common.tracing import new_trace_id, set_run_id, set_trace_id


def _record() -> logging.LogRecord:
    return logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)


def test_filter_takes_trace_id_from_context() -> None:
    trace_id = new_trace_id()
    set_trace_id(trace_id)
    record = _record()
    assert TraceIdFilter().filter(record) is True
    assert record.trace_id == trace_id


def test_filter_falls_back_when_outside_request() -> None:
    set_trace_id("")
    record = _record()
    TraceIdFilter().filter(record)
    assert record.trace_id == "-"


def test_setup_logging_emits_trace_id(capsys) -> None:
    setup_logging("INFO")
    set_trace_id("04b2f9c1")
    logging.getLogger("web3_agent.demo").info("charged")
    assert "[04b2f9c1]" in capsys.readouterr().err


def test_run_id_is_independent_from_trace_id() -> None:
    set_trace_id("t-1")
    set_run_id("r-1")
    from web3_agent.common.tracing import get_run_id, get_trace_id

    assert get_trace_id() == "t-1"
    assert get_run_id() == "r-1"
    set_run_id("")


def test_incoming_trace_id_is_sanitized_before_it_reaches_the_logs() -> None:
    """透传的 id 会进日志和响应头，超长或带控制字符的一律换成新生成的。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from web3_agent.middlewares.trace_middleware import TraceMiddleware

    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/ping")
    def ping() -> dict:
        return {}

    with TestClient(app) as client:
        kept = client.get("/ping", headers={"X-Trace-Id": "abc-123_XY"}).headers["X-Trace-Id"]
        replaced = client.get("/ping", headers={"X-Trace-Id": "x" * 200}).headers["X-Trace-Id"]
        injected = client.get("/ping", headers={"X-Trace-Id": "bad value\nrm -rf"}).headers["X-Trace-Id"]

    assert kept == "abc-123_XY"
    assert replaced != "x" * 200 and len(replaced) == 32
    assert injected != "bad value\nrm -rf" and len(injected) == 32
