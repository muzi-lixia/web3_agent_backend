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
