"""健康检查。/health 便宜，给探针用；/health/ready 会真的问一次模型。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from web3_agent.persistence.pg.base import session_scope

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    """返回进程健康状态及当前服务的基本信息。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    container = request.app.state.container
    db_ok = True
    try:
        async with session_scope() as db:
            await db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 健康检查只报告，不抛
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "chains": container.adapter.available_chains(),
        "langsmith": container.tracing_enabled,
        "model": container.llm.model_name,
    }


@router.get("/health/ready")
async def ready(request: Request) -> dict:
    """探测模型服务就绪状态，失败时返回未就绪结果。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    container = request.app.state.container
    try:
        info = await container.llm.check_model()
        return {"status": "ready", **info}
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_ready", "reason": str(exc)}
