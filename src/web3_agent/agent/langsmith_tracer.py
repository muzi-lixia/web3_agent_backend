"""LangSmith 上报的唯一收敛点。

- 开关只在这里判断，业务代码只问 `run_config()` / `root_extra()`，拿不到配置就等于没开；
- 上报范围由 LangChain 自动埋点覆盖（模型、embedding、图节点、工具），
  纯 Python 的决策点靠 `run_config` 挂上 trace_id，按 trace_id 能搜回审计表；
- 一轮对话只有一条 trace：根节点由 `@traceable` 建立，`run_config()` 只用于它内部的下级调用；
- 上报失败不抛：观测系统的可用性不能高于业务系统。
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import langsmith
from langsmith.utils import get_env_var

from web3_agent.config.settings import Settings

logger = logging.getLogger(__name__)

_enabled = False
_endpoint = ""
_api_key = ""
_probe_client: langsmith.Client | None = None


def configure(settings: Settings) -> bool:
    """在创建任何模型客户端之前调用。返回是否开启。"""
    global _enabled, _endpoint, _api_key, _probe_client
    _enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    _probe_client = None
    if not _enabled:
        # 开关关掉时也要把环境变量压成 false：否则外部 shell 里残留的 true
        # 会让 LangChain 那侧继续埋点，开关就不再是唯一入口。
        os.environ["LANGSMITH_TRACING"] = "false"
        get_env_var.cache_clear()
        langsmith.configure(enabled=False)
        return False

    _endpoint = settings.langsmith_endpoint
    _api_key = settings.langsmith_api_key
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    # configure() 里的 project / enabled 是 ContextVar，只在调用它的那个上下文生效：
    # 启动期上下文（心跳）看得见，请求上下文看不见 —— 那边会退回读环境变量，
    # 读不到就把 run 全塞进 default 项目。所以项目名必须同时落到环境变量。
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"
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
    """下级调用的 RunnableConfig。关闭时返回空 dict，调用方行为不变。

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


def root_extra(**metadata: Any) -> dict[str, Any]:
    """一轮对话根节点的 langsmith_extra。关闭时返回空 dict。

    根节点必须是 `@traceable`：只有它进入上下文后，里面的意图抽取、图、工具调用
    才会挂到同一条 trace 下。给顶层 ainvoke 传 run_config 只会得到一个新的根。
    """
    if not _enabled:
        return {}
    return {
        "run_id": str(uuid.uuid4()),
        "metadata": {key: value for key, value in metadata.items() if value is not None},
    }


def _probe() -> langsmith.Client:
    global _probe_client
    if _probe_client is None:
        _probe_client = langsmith.Client(api_url=_endpoint, api_key=_api_key)
    return _probe_client


def heartbeat() -> bool:
    """确认出口与凭证可用。

    不写 trace：心跳只回答「上报链路是否还通」，而它每次启动一条、每小时一条，
    混在业务项目里会把真正的对话 trace 淹掉。代价是它验不到 run 上传那一段，
    但每轮对话都在产生 trace，那一段坏了本来就会立刻没数据。
    """
    if not _enabled:
        return False
    try:
        _probe().list_projects(limit=1)
        return True
    except Exception:  # noqa: BLE001 观测不可阻断业务
        logger.exception("langsmith heartbeat failed: 出口或凭证不可用")
        return False
