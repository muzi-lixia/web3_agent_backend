"""审计日志写入。审计失败不能影响请求，所以这里吞掉异常并降级到 stdout。"""

from __future__ import annotations

import logging
from typing import Any

from web3_agent.common.tracing import get_run_id, get_trace_id
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import AuditLog

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, enabled: bool = True) -> None:
        """初始化 AuditLogger 所需的依赖和运行状态。

        Args:
            enabled: 是否真正写入审计或启用相关能力。
        """
        self._enabled = enabled

    async def log(
        self,
        event: str,
        *,
        siwe_user_id: str | None = None,
        siwe_address: str | None = None,
        langsmith_run_id: str | None = None,
        **payload: Any,
    ) -> None:
        """写入关联 trace_id 的审计事件；审计写库失败时仅降级记录日志。

        Args:
            event: 交易内的一条资产变动事件。
            siwe_user_id: 数据所属用户的稳定标识。
            siwe_address: 审计关联的钱包地址。
            langsmith_run_id: 审计关联的模型执行 ID。
            **payload: 当前接口请求模型或待渲染的结构化工具结果。
        """
        if not self._enabled:
            logger.info("audit event=%s payload=%s", event, payload)
            return
        try:
            async with session_scope() as db:
                db.add(
                    AuditLog(
                        trace_id=get_trace_id(),
                        siwe_user_id=siwe_user_id,
                        siwe_address=siwe_address,
                        event=event,
                        payload=payload or None,
                        langsmith_run_id=langsmith_run_id or get_run_id() or None,
                    )
                )
        except Exception:  # noqa: BLE001 审计不可阻断主流程
            logger.exception("audit write failed: event=%s", event)
