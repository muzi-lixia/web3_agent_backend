"""FastAPI 入口：组装各层，管理生命周期。

组装只出现在这里。各层之间不互相 new 对方，避免隐式单例散落各处。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from web3_agent.agent import langsmith_tracer
from web3_agent.agent.agent_loop import AgentLoop
from web3_agent.agent.json_guard import JsonGuard
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.api.v1 import auth_router, chat_router, health_router
from web3_agent.common.exceptions import ModelUnavailable
from web3_agent.common.logging_setup import setup_logging
from web3_agent.config.constants import ToolGroup
from web3_agent.config.settings import Settings, get_settings
from web3_agent.domain.memory.embedder import Embedder
from web3_agent.domain.memory.extractor import MemoryExtractor
from web3_agent.domain.memory.summarizer import Summarizer
from web3_agent.domain.quotas import QuotaController
from web3_agent.middlewares.error_handler_middleware import register_exception_handlers
from web3_agent.middlewares.trace_middleware import TraceMiddleware
from web3_agent.persistence.conversation_context import ConversationContext
from web3_agent.persistence.pg.base import create_all, dispose_engine, init_engine, session_scope
from web3_agent.persistence.pg.repositories import nonce_repo, quota_repo
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger
from web3_agent.web3.chain_adapter import ChainAdapter
from web3_agent.web3.tools.read.get_asset_balance import build_get_asset_balance
from web3_agent.web3.tools.read.get_stake_info import build_get_stake_info
from web3_agent.web3.tools.read.get_transaction_records import build_get_transaction_records

setup_logging(get_settings().log_level)
logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SECONDS = 3600
_NONCE_CLEANUP_INTERVAL_SECONDS = 3600
_QUOTA_CLEANUP_INTERVAL_SECONDS = 3600


@dataclass
class Container:
    settings: Settings
    adapter: ChainAdapter
    llm: LlmAdapter
    agent: AgentLoop
    audit: AuditLogger
    quota: QuotaController
    summarizer: Summarizer
    memory: MemoryExtractor
    tracing_enabled: bool


def build_container(settings: Settings) -> Container:
    tracing_enabled = langsmith_tracer.configure(settings)

    adapter = ChainAdapter(settings)
    registry = ToolRegistry()
    registry.register(
        build_get_asset_balance(adapter), ToolGroup.LLM_VISIBLE, mutating=False
    )
    registry.register(
        build_get_transaction_records(adapter, settings), ToolGroup.LLM_VISIBLE, mutating=False
    )
    registry.register(
        build_get_stake_info(adapter, settings.stake_contracts),
        ToolGroup.LLM_VISIBLE,
        mutating=False,
    )

    llm = LlmAdapter(settings)
    runner = ToolRunner(registry)
    conversation = ConversationContext(
        max_messages=settings.conversation_max_messages,
        max_chars=settings.conversation_max_chars,
    )
    audit = AuditLogger()
    quota = QuotaController(settings)
    embedder = Embedder(settings)

    agent = AgentLoop(
        settings=settings,
        llm=llm,
        guard=JsonGuard(),
        conversation=conversation,
        audit=audit,
        model_with_tools=llm.bind_tools(registry.llm_tools()),
        runner=runner,
        embedder=embedder,
    )
    summarizer = Summarizer(settings=settings, llm=llm, audit=audit, quota=quota)
    memory = MemoryExtractor(
        settings=settings, llm=llm, embedder=embedder, audit=audit, quota=quota
    )
    return Container(
        settings=settings,
        adapter=adapter,
        llm=llm,
        agent=agent,
        audit=audit,
        quota=quota,
        summarizer=summarizer,
        memory=memory,
        tracing_enabled=tracing_enabled,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    container = build_container(settings)
    app.state.container = container

    init_engine(settings)
    if settings.auto_create_tables:
        await create_all()

    if settings.model_check_on_startup:
        try:
            info = await container.llm.check_model()
            logger.info("ollama ready: %s", info)
        except ModelUnavailable as exc:
            logger.error("ollama not ready: %s", exc)

    background: list[asyncio.Task] = []
    if container.tracing_enabled:
        logger.info("langsmith heartbeat: %s", langsmith_tracer.heartbeat())
        background.append(asyncio.create_task(_heartbeat_loop()))
    background.append(asyncio.create_task(_nonce_cleanup_loop()))
    background.append(asyncio.create_task(_quota_cleanup_loop()))

    try:
        yield
    finally:
        for task in background:
            task.cancel()
        await container.adapter.aclose()
        await dispose_engine()


async def _heartbeat_loop() -> None:
    """出口断了之后不能安静地没数据：每小时探一次，失败就报出来。"""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        if not langsmith_tracer.heartbeat():
            logger.error("langsmith heartbeat failed: 上报链路可能已断")


async def _cleanup_expired_nonces() -> None:
    """删除过期超过保留窗口的 nonce。失败只记日志，不影响服务。"""
    try:
        async with session_scope() as db:
            removed = await nonce_repo.cleanup(db)
    except Exception:  # noqa: BLE001
        logger.exception("nonce cleanup failed")
        return
    if removed:
        logger.info("nonce cleanup: removed %d expired rows", removed)


async def _nonce_cleanup_loop() -> None:
    """启动先清一次，之后每小时一次。"""
    while True:
        await _cleanup_expired_nonces()
        await asyncio.sleep(_NONCE_CLEANUP_INTERVAL_SECONDS)


async def _cleanup_old_quota() -> None:
    """回收过窗口的限流与配额计数。失败只记日志，不影响服务。"""
    try:
        async with session_scope() as db:
            removed = await quota_repo.cleanup(db)
    except Exception:  # noqa: BLE001
        logger.exception("quota cleanup failed")
        return
    if removed:
        logger.info("quota cleanup: removed %d old buckets", removed)


async def _quota_cleanup_loop() -> None:
    """启动先清一次，之后每小时一次。"""
    while True:
        await _cleanup_old_quota()
        await asyncio.sleep(_QUOTA_CLEANUP_INTERVAL_SECONDS)


def create_app() -> FastAPI:
    app = FastAPI(title="Web3 链上智能 Agent 助手", version="0.8", lifespan=lifespan)
    app.add_middleware(TraceMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router.router)
    app.include_router(auth_router.router, prefix="/api/v1")
    app.include_router(chat_router.router, prefix="/api/v1")
    return app


app = create_app()
