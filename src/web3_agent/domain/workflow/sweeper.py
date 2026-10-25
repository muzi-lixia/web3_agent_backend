"""写操作任务的兜底推进。

前端轮询是状态推进的主路径，这里只处理没人再轮询的那部分：用户关掉页面、
换设备、或者浏览器停在后台。按 `workflow_sweep_seconds` 扫一批仍在流程中且
静默超过一个周期的任务，逐个核对回执或判定过期。

多实例会重复扫同一批任务，但状态推进是条件更新，重复扫不会写错状态，
只会多打几次回执查询。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

from web3_agent.config.constants import TERMINAL_STATES, WorkflowState
from web3_agent.domain.workflow import service
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import workflow_repo
from web3_agent.web3.chain_adapter import ChainAdapter

logger = logging.getLogger(__name__)

# _OPEN_STATES：还没到终态、需要兜底推进的状态。等确认的也可能一直没人点。
# 已过期但报过哈希的也在内：链上结论可能晚于 TTL，没人再轮询就永远停在「未核对」。
_OPEN_STATES = (
    WorkflowState.INTENT_CONFIRM.value,
    WorkflowState.PREVIEW.value,
    WorkflowState.SUBMITTED.value,
    WorkflowState.EXPIRED_UNVERIFIED.value,
)

# _CLOSED_STATES：到了就不会再有人推动的状态；它背后挂起的图也不再有恢复的时机。
_CLOSED_STATES = frozenset(item.value for item in TERMINAL_STATES)


async def sweep_once(
    adapter: ChainAdapter,
    *,
    batch: int,
    stale_seconds: int,
    drop_plan: Callable[[str], Awaitable[None]] | None = None,
) -> int:
    """扫一批未到终态的任务，返回本次推进了状态的任务数。

    Args:
        adapter: 按网络管理 RPC 客户端与只读缓存的链适配器。
        batch: 单次最多处理的任务数。
        stale_seconds: 距上次更新超过该秒数的任务才会被扫到。
        drop_plan: 任务结案后清理其计划检查点；为空表示不清理。
    """
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    async with session_scope() as db:
        rows = await workflow_repo.list_pending(
            db, states=_OPEN_STATES, stale_before=stale_before, limit=batch
        )
    moved = 0
    for row in rows:
        try:
            result = await service.read_status(
                siwe_user_id=row.siwe_user_id,
                siwe_address=row.siwe_address,
                task_id=row.task_id,
                adapter=adapter,
            )
        except Exception:  # noqa: BLE001 单条失败不阻塞其余任务
            logger.exception("workflow sweep failed task=%s", row.task_id)
            continue
        if result["state"] == row.state:
            continue
        moved += 1
        thread = service.plan_thread_of(row)
        if drop_plan is None or not thread or result["state"] not in _CLOSED_STATES:
            continue
        try:
            await drop_plan(thread)
        except Exception:  # noqa: BLE001 清理失败只多留一条挂起记录，不影响结案
            logger.warning("plan checkpoint cleanup failed task=%s", row.task_id, exc_info=True)
    return moved
