"""本轮资源计量与预算边界。

配额要按用户算消耗，但 ChainAdapter 是进程级单例、工具在闭包里持有它，
没法把"这一轮用了多少"顺着参数传上来。用 ContextVar 记在当前请求上下文里，
本轮结束时取走并清零。

计数不能存成 int：工具的 RPC 都发生在 asyncio.gather 的子任务里，子任务是复制上下文，
在子任务里 set() 不会回传父上下文。这里存一张可变的计费表 —— 复制的是引用，
子任务原地累加，父任务读到的是同一个对象。本轮开始时必须先在父上下文里
start_metering()，否则每个子任务各自新建一张表，父任务一个都看不到。

扣减发生在调用之前：一条模型消息可以带多个工具调用，一个工具又能并发打出一串
下游请求，先调用后记账时并发分支会同时通过检查。扣减是同步的，asyncio 单线程下
不存在竞态，也不需要锁；返回 False 就表示这一类预算已经用完，本次调用不要发出。
"""

from __future__ import annotations

from contextvars import ContextVar
from enum import Enum


class Resource(str, Enum):
    """资源分类。计数分开记，上限可以合并：下游几类共用一条预算。"""

    MODEL = "model"
    RPC = "rpc"
    EXPLORER = "explorer"
    EMBEDDING = "embedding"


# 向量化与重排同样是按次消耗的自建服务，和链上请求合并成一条下游预算。
_DOWNSTREAM = (Resource.RPC, Resource.EXPLORER, Resource.EMBEDDING)


class _Ledger:
    __slots__ = ("scope", "downstream", "used")

    def __init__(self, scope: str, downstream: int | None) -> None:
        self.scope = scope
        self.downstream = downstream
        self.used = {resource: 0 for resource in Resource}


_ledger: ContextVar[_Ledger | None] = ContextVar("turn_ledger", default=None)


def start_metering(*, downstream: int | None = None, scope: str = "turn") -> None:
    """在本轮上下文里开一张计费表。必须早于任何并发派发。"""
    _ledger.set(_Ledger(scope, downstream))


def reserve(resource: Resource, amount: int = 1) -> bool:
    """调用前扣减。返回 False 表示这一类预算已经用完，本次调用不应发出。

    模型调用照样计数，限量由各自的硬约束负责（ReAct 步数）。
    """
    ledger = _ledger.get()
    if ledger is None:
        return True
    used = ledger.used
    if resource in _DOWNSTREAM and ledger.downstream is not None:
        if sum(used[r] for r in _DOWNSTREAM) + amount > ledger.downstream:
            return False
    used[resource] += amount
    return True


def take_usage() -> dict[Resource, int]:
    """取走本轮用量并清空。异常、超时、取消路径也要走到这里。"""
    ledger = _ledger.get()
    _ledger.set(None)
    if ledger is None:
        return {resource: 0 for resource in Resource}
    used = dict(ledger.used)
    # 表本身也要清零：子任务和后台任务持有的是同一个对象，只在本上下文置空
    # 挡不住它在另一个上下文里被再结算一次。
    ledger.used.update({resource: 0 for resource in Resource})
    return used
