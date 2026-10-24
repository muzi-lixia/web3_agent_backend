"""trace_id 生成与上下文传播。一条请求 = 一个 trace_id，跨层级不靠传参。"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

# _trace_id：请求追踪 ID 的上下文变量，隔离不同并发请求。
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
# _run_id：当前 LangSmith 执行 ID 的上下文变量。
_run_id: ContextVar[str] = ContextVar("langsmith_run_id", default="")


def new_trace_id() -> str:
    """生成新的 UUID 十六进制请求追踪标识。"""
    return uuid.uuid4().hex


def set_trace_id(trace_id: str) -> None:
    """把请求追踪标识存入当前上下文，供日志和审计读取。

    Args:
        trace_id: 贯穿当前请求的追踪标识。
    """
    _trace_id.set(trace_id)


def get_trace_id() -> str:
    """读取当前上下文的请求追踪标识。"""
    return _trace_id.get()


def set_run_id(run_id: str) -> None:
    """本轮对应的 LangSmith run id。审计靠它反查当次完整轨迹。

    Args:
        run_id: LangSmith 执行记录标识。
    """
    _run_id.set(run_id)


def get_run_id() -> str:
    """读取当前上下文的 LangSmith 执行记录 ID。"""
    return _run_id.get()
