"""写链路接口：意图确认、任务状态查询、交易哈希上报。

状态推进只发生在确认接口和状态查询里。上报接口把哈希和前端观察值写进审计，
不改状态——前端上报一个 confirmed 就能把任务标成完成的话，链上回滚也看不出来。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from web3_agent.api.deps import Identity, chat_rate_limit, require_identity
from web3_agent.api.schemas.workflow_schema import (
    IntentConfirmRequest,
    IntentConfirmResp,
    TxReportRequest,
    TxReportResp,
    WorkflowStatusResp,
)
from web3_agent.domain.workflow import service

router = APIRouter(tags=["workflow"], dependencies=[Depends(chat_rate_limit)])


@router.post(
    "/intent/confirm", response_model=IntentConfirmResp, response_model_by_alias=True
)
async def confirm_intent(
    payload: IntentConfirmRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
) -> IntentConfirmResp:
    """提交用户对写操作的回显决定，接受则进入待签名，拒绝则终止任务。

    Args:
        payload: 经过字段校验的确认请求。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    container = request.app.state.container
    status, thread_id = await service.confirm(
        siwe_user_id=identity.user_id,
        siwe_address=identity.address,
        task_id=payload.task_id,
        accepted=payload.accepted,
    )
    # 多步计划建出来的任务，确认后把决定送回挂起的图；单步任务没有这个标识。
    evidence = await container.agent.resume_plan(thread_id) if thread_id else None
    extra = {"evidence": evidence} if evidence else {}
    await container.audit.log(
        "intent_confirmed",
        siwe_user_id=identity.user_id,
        siwe_address=identity.address,
        task_id=payload.task_id,
        accepted=payload.accepted,
        status=status,
        **extra,
    )
    return IntentConfirmResp(task_id=payload.task_id, status=status)


@router.get(
    "/workflow/{task_id}",
    response_model=WorkflowStatusResp,
    response_model_by_alias=True,
)
async def get_workflow(
    task_id: str,
    request: Request,
    identity: Identity = Depends(require_identity),
) -> WorkflowStatusResp:
    """读取任务当前状态；已广播的任务会顺带核对一次链上回执。

    Args:
        task_id: 对外暴露的任务标识。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    container = request.app.state.container
    data = await service.read_status(
        siwe_user_id=identity.user_id,
        siwe_address=identity.address,
        task_id=task_id,
        adapter=container.adapter,
    )
    return WorkflowStatusResp.model_validate(data)


@router.post("/tx/report", response_model=TxReportResp, response_model_by_alias=True)
async def report_tx(
    payload: TxReportRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
) -> TxReportResp:
    """记录广播哈希与前端观察到的状态，仅写审计，不推进任务状态。

    Args:
        payload: 经过字段校验的上报请求。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    container = request.app.state.container
    first = await service.report(
        siwe_user_id=identity.user_id,
        siwe_address=identity.address,
        task_id=payload.task_id,
        tx_hash=payload.tx_hash,
    )
    await container.audit.log(
        "tx_reported",
        siwe_user_id=identity.user_id,
        siwe_address=identity.address,
        task_id=payload.task_id,
        tx_hash=payload.tx_hash,
        reported_status=payload.status,
        first_report=first,
    )
    return TxReportResp(task_id=payload.task_id)
