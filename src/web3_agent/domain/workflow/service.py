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

# _TRANSFER_TOPIC：ERC20 Transfer(address,address,uint256) 的事件签名哈希。
_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# _RECEIPT_MISMATCH：回执与任务内容不符时对外报的错误码。
_RECEIPT_MISMATCH = "E406"

# _CHECKABLE_STATES：报过哈希、还需要核对回执的状态。过期态也在内：
# 用户可能正是在过期之后才签的名，链上真转了就不能停在「未签名」上。
_CHECKABLE_STATES = frozenset(
    {
        WorkflowState.PREVIEW,
        WorkflowState.SUBMITTED,
        WorkflowState.EXPIRED_UNSIGNED,
        WorkflowState.EXPIRED_UNVERIFIED,
    }
)

# _OPEN_STATES：还没结案、可能被断线重试重复建卡的状态。按自然键去重只在这组
# 状态里找：已结案的（含过期）不拦，用户真想再转一笔同样的应该放行。
_OPEN_STATES = tuple(
    s.value
    for s in (
        WorkflowState.INTENT_CONFIRM,
        WorkflowState.PREVIEW,
        WorkflowState.SUBMITTED,
    )
)


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


def _hex_int(value: object) -> int | None:
    """读回执里的十六进制数值；读不出返回 None。

    Args:
        value: 回执中的字段值，正常是 0x 前缀的十六进制文本。
    """
    try:
        return int(str(value), 16)
    except (TypeError, ValueError):
        return None


def _dec_int(value: object) -> int | None:
    """读交易意图里的十进制数量；读不出返回 None。

    Args:
        value: 建任务时存下的最小单位数量，十进制文本。
    """
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _same_address(left: object, right: object) -> bool:
    """按不区分大小写比较两个地址。

    Args:
        left: 回执里的地址。
        right: 建任务时存下的地址。
    """
    return bool(left) and str(left).lower() == str(right or "").lower()


def _content_matches(tx: dict, intent: dict) -> bool:
    """核对付款方、收款方与转账参数是不是这条任务要发起的那笔。

    Args:
        tx: 链上的交易对象，含 from / to / value / input。
        intent: 建任务时存下的待签名交易参数。
    """
    if not _same_address(tx.get("from"), intent.get("from")):
        return False
    if not _same_address(tx.get("to"), intent.get("to")):
        return False
    data = str(intent.get("data") or "")
    if data in ("", "0x"):
        return _hex_int(tx.get("value")) == _dec_int(intent.get("value"))
    # ERC20：调用数据里编着收款方和金额，逐字节相同才算同一笔。
    return str(tx.get("input") or tx.get("data") or "").lower() == data.lower()


def _transfer_logged(receipt: dict, intent: dict) -> bool:
    """回执里有没有一条把代币转给指定收款方的 Transfer 事件。

    调用数据对上只说明「调了这个合约这个入参」，事件对上才说明余额真的动了。

    Args:
        receipt: 链上交易回执对象。
        intent: 建任务时存下的待签名交易参数。
    """
    data = str(intent.get("data") or "")
    # data = 选择器(4 字节) + 收款地址(32 字节) + 金额(32 字节)。
    if len(data) < 138:
        return False
    payer = str(intent.get("from") or "").lower().removeprefix("0x")
    recipient = data[34:74].lower()
    amount = _hex_int("0x" + data[74:138])
    token = str(intent.get("to") or "").lower()
    for log in receipt.get("logs") or []:
        topics = [str(item).lower() for item in (log.get("topics") or [])]
        # topics[0] 是事件签名，[1] 付款方，[2] 收款方，都由 32 字节补零承载地址。
        if len(topics) < 3 or topics[0] != _TRANSFER_TOPIC:
            continue
        if str(log.get("address") or "").lower() != token:
            continue
        if topics[1][-40:] != payer or topics[2][-40:] != recipient:
            continue
        if _hex_int(log.get("data")) == amount:
            return True
    return False


def _receipt_state(
    receipt: dict | None, tx: dict | None, intent: dict
) -> tuple[WorkflowState | None, str | None]:
    """把回执换算成状态，并判断它是不是这条任务发起的那一笔。

    返回 (状态, 错误码)。状态为空表示这次不推进：还没上链，或者内容对不上任务。
    只看 status 的话，用户随便贴一笔成功交易的哈希就能把任务标成已完成。

    Args:
        receipt: 链上交易回执对象；未上链时为空。
        tx: 链上交易对象；回执存在时才有。
        intent: 建任务时存下的待签名交易参数。
    """
    if receipt is None:
        return None, None
    # 回执本身不带转账金额，金额与调用数据只能从交易对象上核对。
    if tx is None or not _content_matches(tx, intent):
        return None, _RECEIPT_MISMATCH
    if _hex_int(receipt.get("status")) != 1:
        # 回滚的交易没有事件可对；内容对上了就按链上的结论记失败。
        return WorkflowState.FAILED, None
    if str(intent.get("data") or "").lower() not in ("", "0x") and not _transfer_logged(
        receipt, intent
    ):
        return None, _RECEIPT_MISMATCH
    return WorkflowState.SETTLED, None


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
    plan_thread_id: str | None = None,
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
        plan_thread_id: 发起本任务的多步计划标识；单步走流程时为空。
    """
    task_id = str(uuid.uuid4())
    params = {
        "from_address": siwe_address,
        "to_address": to_address,
        "token_symbol": token_symbol,
        "human_amount": human_amount,
    }
    if plan_thread_id:
        # 确认接口凭它找到挂起的计划：前端不需要知道有图这回事。
        params["plan_thread_id"] = plan_thread_id
    async with session_scope() as db:
        # 断线重试会重跑整轮，同一句话可能再建一张待确认卡。先按交易自然键找一条
        # 未结案的同款任务，命中就复用它的 task_id，不新建第二张卡。
        existing = await workflow_repo.find_open_by_natural_key(
            db,
            siwe_user_id=siwe_user_id,
            siwe_address=siwe_address,
            chain_id=chain_id,
            to_address=to_address,
            token_symbol=token_symbol,
            human_amount=human_amount,
            open_states=_OPEN_STATES,
        )
        if existing is not None:
            return existing.task_id
        await workflow_repo.create(
            db,
            task_id=task_id,
            siwe_user_id=siwe_user_id,
            siwe_address=siwe_address,
            action="transfer",
            chain_id=chain_id,
            state=WorkflowState.INTENT_CONFIRM.value,
            params=params,
            tx_intent=tx_intent,
            expires_at=_now() + timedelta(seconds=ttl_seconds),
        )
    return task_id


def plan_thread_of(row: Row) -> str | None:
    """取出发起该任务的多步计划标识；单步建出来的任务没有这个值。

    Args:
        row: 写操作任务行。
    """
    value = (row.params or {}).get("plan_thread_id")
    return value if isinstance(value, str) and value else None


async def confirm(
    *, siwe_user_id: str, siwe_address: str, task_id: str, accepted: bool
) -> tuple[str, str | None]:
    """提交用户对意图回显的决定，返回结论与发起该任务的多步计划标识。

    同一个决定重复提交返回原结论，不报错；已确认过再提交拒绝、或已拒绝再提交
    确认属于状态冲突，按 E609 拒绝。

    Args:
        siwe_user_id: 当前登录用户的稳定标识。
        siwe_address: 当前登录钱包的完整地址。
        task_id: 对外暴露的任务标识。
        accepted: 用户是否接受本次写操作。
    """
    status: str | None = None
    thread: str | None = None
    failure: tuple[str, str, int] | None = None
    async with session_scope() as db:
        row = await workflow_repo.get(db, task_id)
        thread = None if row is None else plan_thread_of(row)
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
    return status, thread


async def report(
    *, siwe_user_id: str, siwe_address: str, task_id: str, tx_hash: str
) -> bool:
    """记录前端上报的广播哈希，返回本次是否为第一笔。

    只写哈希，不推进状态：状态改由 `read_status` 核对回执产生。已结案且不可能
    再广播的任务（已结算、已回滚、已拒绝）不再收哈希；过期的不一样 —— 用户很可能
    就是在过期之后才签的名，补登哈希后由下一次回执核对把状态纠正回来。
    已经报过哈希的任务仍接收新哈希：钱包加速或取消会换上链的那一笔，按住第一笔不放
    会一直核对一个永远不会上链的哈希。内容对不对由回执核对负责。

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
        if state in TERMINAL_STATES and state not in _CHECKABLE_STATES:
            raise AppError(*E609_STATE_CONFLICT)
        if row.tx_hash == tx_hash:
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
    # 过期任务也再核一次：哈希可能晚到，链上真转了不能被一条「已过期」盖掉。
    checking = row.tx_hash is not None and state in _CHECKABLE_STATES
    if state in TERMINAL_STATES and not checking:
        return _render(row)

    expired = _is_expired(row)
    resolved: WorkflowState | None = None
    rpc_error: str | None = None
    if checking:
        try:
            receipt = await adapter.get_receipt(row.chain_id, row.tx_hash)
            # 未上链就没有交易对象可查；已上链才有内容可核对。
            tx = (
                await adapter.get_transaction(row.chain_id, row.tx_hash)
                if receipt is not None
                else None
            )
            resolved, rpc_error = _receipt_state(receipt, tx, row.tx_intent or {})
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
        elif state in TERMINAL_STATES:
            # 已过期又核不出来就保持原样，等下一次核对；对不上账的更不能改状态。
            pass
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
