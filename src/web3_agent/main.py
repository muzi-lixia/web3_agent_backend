"""FastAPI 入口：组装各层，管理生命周期。

组装只出现在这里。各层之间不互相 new 对方，避免隐式单例散落各处。
"""

from __future__ import annotations

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
from web3_agent.api.v1 import chat_router, health_router
from web3_agent.common.exceptions import ModelUnavailable
from web3_agent.config.constants import ToolGroup
from web3_agent.config.settings import Settings, get_settings
from web3_agent.middlewares.error_handler_middleware import register_exception_handlers
from web3_agent.middlewares.trace_middleware import TraceMiddleware
from web3_agent.persistence.conversation_context import ConversationContext
from web3_agent.persistence.pg.base import create_all, dispose_engine, init_engine
from web3_agent.persistence.pg.repositories.audit_repo import AuditLogger
from web3_agent.web3.chain_adapter import ChainAdapter
from web3_agent.web3.tools.read.get_asset_balance import build_get_asset_balance
from web3_agent.web3.tools.read.get_stake_info import build_get_stake_info
from web3_agent.web3.tools.read.get_transaction_records import build_get_transaction_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    adapter: ChainAdapter
    llm: LlmAdapter
    agent: AgentLoop
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
        ttl_seconds=settings.conversation_ttl_seconds,
        max_messages=settings.conversation_max_messages,
    )
    audit = AuditLogger()

    agent = AgentLoop(
        settings=settings,
        llm=llm,
        guard=JsonGuard(),
        conversation=conversation,
        audit=audit,
        model_with_tools=llm.bind_tools(registry.llm_tools()),
        runner=runner,
    )
    return Container(
        settings=settings, adapter=adapter, llm=llm, agent=agent, tracing_enabled=tracing_enabled
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

    if container.tracing_enabled:
        ok = langsmith_tracer.heartbeat()
        logger.info("langsmith heartbeat: %s", ok)

    try:
        yield
    finally:
        await container.adapter.aclose()
        await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(title="Web3 链上智能 Agent 助手", version="0.8", lifespan=lifespan)
    app.add_middleware(TraceMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router.router)
    app.include_router(chat_router.router, prefix="/api/v1")
    return app


app = create_app()
