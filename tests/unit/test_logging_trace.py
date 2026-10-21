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
