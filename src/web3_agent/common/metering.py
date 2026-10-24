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

    # MODEL：聊天、抽取或摘要模型的一次调用。
    MODEL = "model"
    # RPC：一次真实 JSON-RPC 请求，重试另计。
    RPC = "rpc"
    # EXPLORER：一次真实区块浏览器请求，重试另计。
    EXPLORER = "explorer"
    # EMBEDDING：一次向量化或重排服务请求。
    EMBEDDING = "embedding"


# 向量化与重排同样是按次消耗的自建服务，和链上请求合并成一条下游预算。
# _DOWNSTREAM：共同消耗本轮下游预算的资源类别集合。
_DOWNSTREAM = (Resource.RPC, Resource.EXPLORER, Resource.EMBEDDING)


# 数据职责：当前异步调用范围共享的可变资源账本，子任务通过对象引用累加。
class _Ledger:
    # __slots__：限制实例允许保存的属性集合，减少资源账本的额外对象开销。
    __slots__ = ("scope", "downstream", "used")

    def __init__(self, scope: str, downstream: int | None) -> None:
        """初始化本范围的资源计数和共享下游上限。

        Args:
            scope: 计量范围标记，例如 turn 或 background。
            downstream: 本范围下游请求总次数上限；为空表示不设此上限。
        """
        self.scope = scope
        self.downstream = downstream
        # used：按 Resource 分类累计的调用次数，异步子任务共享该字典。
        self.used = {resource: 0 for resource in Resource}


# _ledger：当前上下文的共享资源账本，默认 None 表示未开启计量。
_ledger: ContextVar[_Ledger | None] = ContextVar("turn_ledger", default=None)


def start_metering(*, downstream: int | None = None, scope: str = "turn") -> None:
    """在本轮上下文里开一张计费表。必须早于任何并发派发。

    Args:
        downstream: 本范围下游请求总次数上限；为空表示不设此上限。
        scope: 计量范围标记，例如 turn 或 background。
    """
    _ledger.set(_Ledger(scope, downstream))


def reserve(resource: Resource, amount: int = 1) -> bool:
    """调用前扣减。返回 False 表示这一类预算已经用完，本次调用不应发出。

    模型调用照样计数，限量由各自的硬约束负责（ReAct 步数）。

    Args:
        resource: 本次要占用的资源分类。
        amount: 本次累加或预占的资源数量。

    Returns:
        允许发出调用时为 True，超出下游预算时为 False；未初始化账本时直接放行。
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
    """取走本轮用量并清空。异常、超时、取消路径也要走到这里。

    Returns:
        各 Resource 的当前累计次数，随后本账本清零。
    """
    ledger = _ledger.get()
    _ledger.set(None)
    if ledger is None:
        return {resource: 0 for resource in Resource}
    used = dict(ledger.used)
    # 表本身也要清零：子任务和后台任务持有的是同一个对象，只在本上下文置空
    # 挡不住它在另一个上下文里被再结算一次。
    ledger.used.update({resource: 0 for resource in Resource})
    return used
