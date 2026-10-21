"""POST /api/v1/chat：聊天主入口。只做参数、鉴权、调 service、组装响应。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from web3_agent.api.deps import Identity, chat_rate_limit, require_identity
from web3_agent.api.schemas.chat_schema import ChatRequest, ChatResponse, HistoryResp, HistoryItem, ClearResp
from web3_agent.common.exceptions import E102_CHAIN_NOT_ALLOWED, AppError, E509_CHAIN_IDENTITY_MISMATCH
from web3_agent.config.constants import QuotaMetric
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import ConversationSummary
from web3_agent.persistence.pg.repositories import conversation_repo
from web3_agent.web3.metering import take_rpc_calls
from sqlalchemy import delete, text

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(chat_rate_limit)])

_KIND_TO_TYPE = {"answer": "answer", "ask": "ask", "reject": "answer"}

# 后台摘要任务。不 await：摘要慢，不能拖住回复。集合持有引用，否则任务可能被 GC 掉。
_BACKGROUND: set[asyncio.Task] = set()


def _spawn_summary(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


@router.post("", response_model=ChatResponse, response_model_by_alias=True)
async def chat(
    payload: ChatRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
) -> ChatResponse:
    container = request.app.state.container

    if payload.wallet and payload.wallet.chain_id not in container.settings.allowed_chain_ids:
        raise AppError(*E102_CHAIN_NOT_ALLOWED)

    chain_id = payload.wallet.chain_id if payload.wallet else None
    if chain_id is not None and chain_id != identity.chain_id:
        raise AppError(*E509_CHAIN_IDENTITY_MISMATCH)
    # 请求没带 wallet 时用登录身份里的链：档案要记代币属于哪条链，缺了就只能丢弃。
    effective_chain = chain_id if chain_id is not None else identity.chain_id

    async with session_scope() as db:
        outcome = await container.agent.handle(
            db,
            siwe_user_id=identity.user_id,
            wallet_address=identity.address,
            chain_id=effective_chain,
            user_message=payload.user_message,
        )

    rpc_calls = take_rpc_calls()
    quota_key = getattr(request.state, "quota_key", None)
    if quota_key:
        async with session_scope() as db:
            await container.quota.charge(
                db, key=quota_key, metric=QuotaMetric.STEPS, amount=outcome.steps
            )
            await container.quota.charge(
                db, key=quota_key, metric=QuotaMetric.RPC_CALLS, amount=rpc_calls
            )

    # 事务已提交，摘要任务才能看到本轮消息；它自己开会话、自己兜异常。
    _spawn_summary(
        container.summarizer.run(siwe_user_id=identity.user_id, own_address=identity.address)
    )

    return ChatResponse(
        type=_KIND_TO_TYPE.get(outcome.kind, "answer"),
        content=outcome.content,
        data={
            "steps": outcome.steps,
            "toolCalls": [call.model_dump() for call in outcome.tool_calls],
        },
    )

@router.get("/history", response_model=HistoryResp, response_model_by_alias=True)
async def history(
    before: int | None = None,
    limit: int = 20,
    identity: Identity = Depends(require_identity),
) -> HistoryResp:
    limit = max(1, min(limit, 50))
    async with session_scope() as db:
        rows = await conversation_repo.list_page(db, identity.user_id, limit, before)
    has_more = len(rows) > limit
    rows = rows[:limit]
    return HistoryResp(
        items=[
            HistoryItem(id=r.id, role=r.role, content=r.content, created_at=r.created_at)
            for r in reversed(rows)
        ],
        has_more=has_more,
    )


@router.post("/clear", response_model=ClearResp, response_model_by_alias=True)
async def clear(request: Request, identity: Identity = Depends(require_identity)) -> ClearResp:
    container = request.app.state.container
    started = datetime.now(timezone.utc)
    async with session_scope() as db:
        await db.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:u))"), {"u": identity.user_id}
        )
        cleared = await conversation_repo.clear_user_messages(db, identity.user_id)
        result = await db.execute(
            delete(ConversationSummary).where(
                ConversationSummary.siwe_user_id == identity.user_id
            )
        )
    await container.audit.log(
        "conversation_cleared", siwe_user_id=identity.user_id, cleared=cleared
    )
    return ClearResp(
        cleared_messages=cleared,
        cleared_summary=(result.rowcount or 0) > 0,
        started_at=started,
    )

