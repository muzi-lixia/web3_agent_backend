"""trace_id 生成与上下文传播。一条请求 = 一个 trace_id，跨层级不靠传参。"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_run_id: ContextVar[str] = ContextVar("langsmith_run_id", default="")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> str:
    return _trace_id.get()


def set_run_id(run_id: str) -> None:
    """本轮对应的 LangSmith run id。审计靠它反查当次完整轨迹。"""
    _run_id.set(run_id)


def get_run_id() -> str:
    return _run_id.get()
