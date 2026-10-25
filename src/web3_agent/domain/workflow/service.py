"""写操作 workflow：建任务、确认、按回执推进状态。

状态推进的触发源只有两处：`intent/confirm` 提交用户决定，`workflow/{taskId}`
拉取时核对回执。`tx/report` 只记录哈希与审计，不改状态 —— 状态必须来自链上
回执，否则前端上报一个成功就能把任务标成完成。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from web3_agent.common.evm_address import short_address
from web3_agent.common.exceptions import (
    E301_RPC_TIMEOUT,
    E504_TASK_NOT_FOUND,
    E505_TASK_FORBIDDEN,
    E506_TASK_EXPIRED,
    E609_STATE_CONFLICT,
    AppError,
    error_message,
)
from web3_agent.config.constants import TERMINAL_STATES, WorkflowState
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import WorkflowTask as Row
from web3_agent.persistence.pg.repositories import workflow_repo
from web3_agent.web3.chain_adapter import ChainAdapter


def _now() -> datetime:
    """返回当前带 UTC 时区的时间，作为过期判断基准。"""
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    """把数据库读回的时间补成带时区值。

    Args:
        value: 待补时区的数据库时间字段。
    """
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _is_expired(row: Row) -> bool:
    """判断任务是否已过期。

    Args:
        row: 写操作任务行。
    """
    return _now() >= _aware(row.expires_at)


def _receipt_state(receipt: dict | None) -> WorkflowState | None:
    """把回执换算成状态；未上链返回 None。

    Args:
        receipt: 链上交易回执对象；未上链时为空。
    """
    if receipt is None:
        return None
    try:
        succeeded = int(str(receipt.get("status")), 16) == 1
    except (TypeError, ValueError):
        return WorkflowState.FAILED
    return WorkflowState.SETTLED if succeeded else WorkflowState.FAILED


def _owned(row: Row, *, siwe_user_id: str, siwe_address: str) -> bool:
    """判断任务是否属于当前登录身份。

    Args:
        row: 写操作任务行。
        siwe_user_id: 当前登录用户的稳定标识。
        siwe_address: 当前登录钱包的完整地址。
    """
    return row.siwe_user_id == siwe_user_id and row.siwe_address == siwe_address


def _render(row: Row, *, rpc_error: str | None = None) -> dict:
    """把任务行整理成对外状态响应；txIntent 只在等待签名时出现。

    Args:
        row: 写操作任务行。
        rpc_error: 本次回执核对失败时的错误码，仅用于响应，不落库。
    """
    state = WorkflowState(row.state)
    return {
        "taskId": row.task_id,
        "state": state.value,
        "expireType": row.expire_type,
        "siweAddressShort": short_address(row.siwe_address),
        "txIntent": row.tx_intent if state is WorkflowState.PREVIEW else None,
        "txHash": row.tx_hash,
        "errorCode": rpc_error or row.error_code,
        "errorMsg": error_message(rpc_error) if rpc_error else row.error_message,
        "expiresAt": int(_aware(row.expires_at).timestamp()),
    }


async def create_transfer(
    *,
    siwe_user_id: str,
    siwe_address: str,
    chain_id: int,
    to_address: str,
    token_symbol: str,
    human_amount: str,
    tx_intent: dict,
    ttl_seconds: int,
) -> str:
    """创建一个等待用户确认的转账任务，返回 taskId。

    Args:
        siwe_user_id: 任务归属用户的稳定标识。
        siwe_address: 任务归属钱包的完整地址。
        chain_id: 写操作所在网络编号。
        to_address: 收款地址。
        token_symbol: 转账资产的符号。
        human_amount: 人类可读的转账数量。
        tx_intent: 已构造好的待签名交易参数。
        ttl_seconds: 任务有效期秒数。
    """
    task_id = str(uuid.uuid4())
    async with session_scope() as db:
        await workflow_repo.create(
            db,
            task_id=task_id,
            siwe_user_id=siwe_user_id,
            siwe_address=siwe_address,
            action="transfer",
            chain_id=chain_id,
            state=WorkflowState.INTENT_CONFIRM.value,
            params={
                "from_address": siwe_address,
                "to_address": to_address,
                "token_symbol": token_symbol,
                "human_amount": human_amount,
            },
            tx_intent=tx_intent,
            expires_at=_now() + timedelta(seconds=ttl_seconds),
        )
    return task_id


async def confirm(
    *, siwe_user_id: str, siwe_address: str, task_id: str, accepted: bool
) -> str:
    """提交用户对意图回显的决定，返回 ACCEPTED 或 REJECTED。

    同一个决定重复提交返回原结论，不报错；已确认过再提交拒绝、或已拒绝再提交
    确认属于状态冲突，按 E609 拒绝。

    Args:
        siwe_user_id: 当前登录用户的稳定标识。
        siwe_address: 当前登录钱包的完整地址。
        task_id: 对外暴露的任务标识。
        accepted: 用户是否接受本次写操作。
    """
    status: str | None = None
    failure: tuple[str, str, int] | None = None
    async with session_scope() as db:
        row = await workflow_repo.get(db, task_id)
        if row is None:
            failure = E504_TASK_NOT_FOUND
        elif not _owned(row, siwe_user_id=siwe_user_id, siwe_address=siwe_address):
            failure = E505_TASK_FORBIDDEN
        elif WorkflowState(row.state) in TERMINAL_STATES:
            # 终态不该被重新解释：已拒绝的任务过一会儿变成「过期」会让人以为还能重试。
            if not accepted and row.state == WorkflowState.ABORTED.value:
                status = "REJECTED"
            else:
                failure = E609_STATE_CONFLICT
        elif _is_expired(row):
            # 从未广播，可以安全地重新发起，所以落到 EXPIRED_UNSIGNED。
            await workflow_repo.transition(
                db,
                task_id,
                expect=row.state,
                state=WorkflowState.EXPIRED_UNSIGNED.value,
                expire_type=WorkflowState.EXPIRED_UNSIGNED.value,
                error_code=E506_TASK_EXPIRED[0],
                error_message=E506_TASK_EXPIRED[1],
            )
            failure = E506_TASK_EXPIRED
        elif row.state == WorkflowState.INTENT_CONFIRM.value:
            status = "ACCEPTED" if accepted else "REJECTED"
            await workflow_repo.transition(
                db,
                task_id,
                expect=row.state,
                state=(
                    WorkflowState.PREVIEW.value
                    if accepted
                    else WorkflowState.ABORTED.value
                ),
            )
        elif accepted and row.state in (
            WorkflowState.PREVIEW.value,
            WorkflowState.SUBMITTED.value,
        ):
            status = "ACCEPTED"
        else:
            failure = E609_STATE_CONFLICT
    if failure is not None:
        raise AppError(*failure)
    return status


async def report(
    *, siwe_user_id: str, siwe_address: str, task_id: str, tx_hash: str
) -> bool:
    """记录前端上报的广播哈希，返回本次是否为第一笔。

    只写哈希，不推进状态：状态改由 `read_status` 核对回执产生。
    已到终态的任务不再收哈希——它已经结案，写进去也不会再被核对。

    Args:
        siwe_user_id: 当前登录用户的稳定标识。
        siwe_address: 当前登录钱包的完整地址。
        task_id: 对外暴露的任务标识。
        tx_hash: 前端上报的广播交易哈希。
    """
    async with session_scope() as db:
        row = await workflow_repo.get(db, task_id)
        if row is None:
            raise AppError(*E504_TASK_NOT_FOUND)
        if not _owned(row, siwe_user_id=siwe_user_id, siwe_address=siwe_address):
            raise AppError(*E505_TASK_FORBIDDEN)
        state = WorkflowState(row.state)
        if state in TERMINAL_STATES:
            expired = state in (
                WorkflowState.EXPIRED_UNSIGNED,
                WorkflowState.EXPIRED_UNVERIFIED,
            )
            raise AppError(*(E506_TASK_EXPIRED if expired else E609_STATE_CONFLICT))
        if row.tx_hash is not None:
            return False
        return await workflow_repo.set_tx_hash(db, task_id, tx_hash)


async def read_status(
    *,
    siwe_user_id: str,
    siwe_address: str,
    task_id: str,
    adapter: ChainAdapter,
) -> dict:
    """读取任务当前状态，需要时用链上回执推进，返回对外状态响应。

    回执查询不在数据库事务里做：一次 RPC 可能等到超时，占着连接会把连接池耗尽。

    Args:
        siwe_user_id: 当前登录用户的稳定标识。
        siwe_address: 当前登录钱包的完整地址。
        task_id: 对外暴露的任务标识。
        adapter: 按网络管理 RPC 客户端与只读缓存的链适配器。
    """
    async with session_scope() as db:
        row = await workflow_repo.get(db, task_id)
    if row is None:
        raise AppError(*E504_TASK_NOT_FOUND)
    if not _owned(row, siwe_user_id=siwe_user_id, siwe_address=siwe_address):
        raise AppError(*E505_TASK_FORBIDDEN)

    state = WorkflowState(row.state)
    if state in TERMINAL_STATES:
        return _render(row)

    expired = _is_expired(row)
    # 只有用户确认过、且已上报哈希的任务才可能在链上查到回执；
    # 过期任务也要再核对一次，避免「已成功却被判过期」。
    check_receipt = row.tx_hash is not None and state in (
        WorkflowState.PREVIEW,
        WorkflowState.SUBMITTED,
    )
    resolved: WorkflowState | None = None
    rpc_error: str | None = None
    if check_receipt:
        try:
            resolved = _receipt_state(await adapter.get_receipt(row.chain_id, row.tx_hash))
        except Exception:  # noqa: BLE001 RPC 失败只影响本次推进，不动任务状态
            rpc_error = E301_RPC_TIMEOUT[0]

    async with session_scope() as db:
        if resolved is not None:
            await workflow_repo.transition(
                db,
                task_id,
                expect=row.state,
                state=resolved.value,
                error_code="E401" if resolved is WorkflowState.FAILED else None,
                error_message=(
                    error_message("E401") if resolved is WorkflowState.FAILED else None
                ),
            )
        elif expired:
            final = (
                WorkflowState.EXPIRED_UNVERIFIED
                if row.tx_hash is not None
                else WorkflowState.EXPIRED_UNSIGNED
            )
            await workflow_repo.transition(
                db,
                task_id,
                expect=row.state,
                state=final.value,
                expire_type=final.value,
                error_code=E506_TASK_EXPIRED[0],
                error_message=E506_TASK_EXPIRED[1],
            )
        elif state is WorkflowState.PREVIEW and row.tx_hash is not None:
            # 已上报哈希就说明交易已经广播，回执查不到只影响结算快慢，不影响「已广播」这个事实。
            await workflow_repo.transition(
                db,
                task_id,
                expect=row.state,
                state=WorkflowState.SUBMITTED.value,
            )
        row = await workflow_repo.get(db, task_id)
    return _render(row, rpc_error=rpc_error)
