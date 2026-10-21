"""RPC 调用计量。

配额要按用户算 RPC 消耗，但 ChainAdapter 是进程级单例、工具在闭包里持有它，
没法把"这次请求调了几次 RPC"顺着参数传上来。用 ContextVar 记在当前请求上下文里，
请求结束时取走并清零——同一协程内累加，跨请求不会串。
"""

from __future__ import annotations

from contextvars import ContextVar

_rpc_calls: ContextVar[int] = ContextVar("rpc_calls", default=0)


def record_rpc_call() -> None:
    _rpc_calls.set(_rpc_calls.get() + 1)


def take_rpc_calls() -> int:
    """取走本次请求的累计次数并清零。"""
    value = _rpc_calls.get()
    _rpc_calls.set(0)
    return value
