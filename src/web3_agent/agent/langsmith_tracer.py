"""LangSmith 上报的唯一收敛点。

- 只有这里读写 LANGSMITH_* 环境变量，业务代码不碰；
- 上报内容分两级：L1 结构元数据全量，L2 业务内容只在需要时上报；
- 私钥/助记词命中时，本轮根本不会调用模型（见 agent_loop），因此也不会产生 trace。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langsmith import traceable

from web3_agent.agent import secret_detector
from web3_agent.config.settings import Settings

logger = logging.getLogger(__name__)


def configure(settings: Settings) -> bool:
    """在创建任何模型客户端之前调用。返回是否开启。"""
    enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_SAMPLE_RATE"] = str(settings.langsmith_sample_rate)
    return True


def metadata(trace_id: str, **extra: Any) -> dict[str, Any]:
    """L1 结构元数据：不含地址明文，可全量上报。"""
    return {"trace_id": trace_id, **extra}


def traced(name: str, run_type: str = "chain"):
    """包装纯 Python 步骤，让写路径/校验层在 trace 里可见。"""
    return traceable(name=name, run_type=run_type)


def must_hide_inputs(text: str) -> bool:
    return secret_detector.detect(text) is not None


@traced("tracing_heartbeat")
def _heartbeat_run() -> str:
    return "ok"


def heartbeat() -> bool:
    """启动与定时调用：确认 trace 真的能发出去，而不只是开关打开了。"""
    if os.environ.get("LANGSMITH_TRACING") != "true":
        return False
    try:
        _heartbeat_run()
        return True
    except Exception:  # noqa: BLE001 观测不可阻断业务
        logger.exception("langsmith heartbeat failed")
        return False
