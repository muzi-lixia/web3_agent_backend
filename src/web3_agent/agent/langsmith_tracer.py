"""LangSmith 上报的唯一收敛点。

- 开关只在这里判断，业务代码只问 `run_config()`，拿不到配置就等于没开；
- 上报范围由 LangChain 自动埋点覆盖（模型、embedding、图节点、工具），
  纯 Python 的决策点靠 `run_config` 挂上 trace_id，按 trace_id 能搜回审计表；
- 上报失败不抛：观测系统的可用性不能高于业务系统。
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import langsmith
from langsmith import traceable
from langsmith.utils import get_env_var

from web3_agent.config.settings import Settings

logger = logging.getLogger(__name__)

_enabled = False


def configure(settings: Settings) -> bool:
    """在创建任何模型客户端之前调用。返回是否开启。"""
    global _enabled
    _enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    if not _enabled:
        langsmith.configure(enabled=False)
        return False

    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    # get_env_var 带 lru_cache：导入期如果有人读过一次，后面再设 env 就不生效了。
    get_env_var.cache_clear()
    langsmith.configure(
        enabled=True,
        project_name=settings.langsmith_project,
        tags=["web3-agent", settings.app_env],
    )
    return True


def enabled() -> bool:
    return _enabled


def run_config(name: str, **metadata: Any) -> dict[str, Any]:
    """一次调用的 RunnableConfig。关闭时返回空 dict，调用方行为不变。

    run_id 由这里生成，调用方据此写审计的 langsmith_run_id —— 事后从 run tree
    里取不可靠（run 结束上下文就没了），自己指定才是确定的。
    """
    if not _enabled:
        return {}
    return {
        "run_name": name,
        "run_id": str(uuid.uuid4()),
        "metadata": {key: value for key, value in metadata.items() if value is not None},
    }


@traceable(run_type="chain", name="tracing_heartbeat")
def _heartbeat_run() -> str:
    return "ok"


def heartbeat() -> bool:
    """确认 trace 真的发得出去，而不只是开关打开了。

    网络出口受限时上报会持续失败，故障隔离又把异常吞了 —— 不主动探一次，
    观测会安静地变成空的。
    """
    if not _enabled:
        return False
    try:
        _heartbeat_run()
        return True
    except Exception:  # noqa: BLE001 观测不可阻断业务
        logger.exception("langsmith heartbeat failed")
        return False
