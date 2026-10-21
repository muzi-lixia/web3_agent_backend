"""POST /api/v1/chat：聊天主入口。只做参数、鉴权、调 service、组装响应。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from web3_agent.api.deps import Identity, chat_rate_limit, optional_identity
from web3_agent.api.schemas.chat_schema import ChatRequest, ChatResponse
from web3_agent.common.exceptions import E102_CHAIN_NOT_ALLOWED, AppError
from web3_agent.config.constants import QuotaMetric
from web3_agent.persistence.pg.base import session_scope
from web3_agent.web3.metering import take_rpc_calls

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(chat_rate_limit)])

_KIND_TO_TYPE = {"answer": "answer", "ask": "ask", "reject": "answer"}


@router.post("", response_model=ChatResponse, response_model_by_alias=True)
async def chat(
    payload: ChatRequest,
    request: Request,
    identity: Identity | None = Depends(optional_identity),
) -> ChatResponse:
    container = request.app.state.container

    if payload.wallet and payload.wallet.chain_id not in container.settings.allowed_chain_ids:
        code, msg, status = E102_CHAIN_NOT_ALLOWED
        raise AppError(code, msg, status)

    wallet_address = payload.wallet.address if payload.wallet else None
    if identity is not None:
        wallet_address = identity.address

    async with session_scope() as db:
        outcome = await container.agent.handle(
            db,
            session_id=payload.session_id,
            wallet_address=wallet_address,
            chain_id=payload.wallet.chain_id if payload.wallet else None,
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

    return ChatResponse(
        type=_KIND_TO_TYPE.get(outcome.kind, "answer"),
        content=outcome.content,
        data={
            "steps": outcome.steps,
            "toolCalls": [call.model_dump() for call in outcome.tool_calls],
        },
    )
