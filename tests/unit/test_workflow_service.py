"""写链路：确认、状态推进、哈希上报。

用内存替身代替仓储，专测判定分支；真库上的并发与唯一约束靠集成用例覆盖。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from web3_agent.api.v1 import workflow_router
from web3_agent.common.exceptions import AppError
from web3_agent.config.constants import WorkflowState
from web3_agent.domain.workflow import service
from web3_agent.persistence.pg.repositories import workflow_repo
from tests.unit.test_chat_idempotency_route import _Session

# _USER：登录用户的稳定标识。
_USER = "user-1"
# _ADDRESS：登录钱包的完整地址，任务归属据此判定。
_ADDRESS = "0x" + "1" * 40
# _OTHER：另一个钱包地址，用于越权用例。
_OTHER = "0x" + "9" * 40
# _TASK：被测任务标识。
_TASK = "task-abc"
# _HASH：一笔广播后的交易哈希。
_HASH = "0x" + "a" * 64
# _TOKEN：白名单 ERC20 的合约地址，ERC20 转账的 tx.to 指向它。
_TOKEN = "0x" + "3" * 40
# _TX：待签名交易参数，与 post_transfer 的输出字段一致。
_TX = {
    "from": _ADDRESS,
    "to": "0x" + "2" * 40,
    "value": "100000000000000000",
    "data": "0x",
    "chainId": 80094,
}


def _row(**overrides) -> SimpleNamespace:
    """构造一条任务行，默认处于等待确认状态。"""
    row = SimpleNamespace(
        task_id=_TASK,
        siwe_user_id=_USER,
        siwe_address=_ADDRESS,
        action="transfer",
        chain_id=80094,
        state=WorkflowState.INTENT_CONFIRM.value,
        expire_type="NONE",
        params={
            "from_address": _ADDRESS,
            "to_address": "0x" + "2" * 40,
            "token_symbol": "BERA",
            "human_amount": "0.1",
        },
        tx_intent=_TX,
        tx_hash=None,
        error_code=None,
        error_message=None,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _install(monkeypatch, *rows: SimpleNamespace) -> dict[str, SimpleNamespace]:
    """把仓储换成内存表，返回可供断言的存储。"""
    store = {row.task_id: row for row in rows}
    calls: list[tuple] = []

    async def fake_get(db, task_id):
        return store.get(task_id)

    async def fake_transition(db, task_id, *, expect, state, **kwargs):
        calls.append(("transition", task_id, expect, state))
        row = store.get(task_id)
        if row is None or row.state != expect:
            return False
        row.state = state
        for key, value in kwargs.items():
            if value is not None:
                setattr(row, key, value)
        return True

    async def fake_set_tx_hash(db, task_id, tx_hash):
        row = store.get(task_id)
        if row is None:
            return False
        first = row.tx_hash is None
        row.tx_hash = tx_hash
        return first

    monkeypatch.setattr(service, "session_scope", lambda: _Session())
    monkeypatch.setattr(workflow_repo, "get", fake_get)
    monkeypatch.setattr(workflow_repo, "transition", fake_transition)
    monkeypatch.setattr(workflow_repo, "set_tx_hash", fake_set_tx_hash)
    return {"store": store, "calls": calls}


class _Adapter:
    """只回答回执与交易查询，其它能力在写链路用不到。"""

    def __init__(
        self, receipt=None, error: Exception | None = None, calldata: str = "0x"
    ) -> None:
        self.receipt = receipt
        self.error = error
        self.calldata = calldata
        self.asked: list[tuple[int, str]] = []

    async def get_receipt(self, chain_id: int, tx_hash: str):
        self.asked.append((chain_id, tx_hash))
        if self.error is not None:
            raise self.error
        return self.receipt

    async def get_transaction(self, chain_id: int, tx_hash: str):
        # 真实节点上交易与回执是两份数据；测试里从回执推出交易，只关心比对逻辑。
        if self.receipt is None:
            return None
        return {
            "from": self.receipt.get("from"),
            "to": self.receipt.get("to"),
            "value": self.receipt.get("value"),
            "input": self.calldata,
        }


def _confirm(
    accepted: bool, *, user: str = _USER, address: str = _ADDRESS
) -> tuple[str, str | None]:
    return asyncio.run(
        service.confirm(
            siwe_user_id=user, siwe_address=address, task_id=_TASK, accepted=accepted
        )
    )


def _status(adapter: _Adapter, *, user: str = _USER, address: str = _ADDRESS) -> dict:
    return asyncio.run(
        service.read_status(
            siwe_user_id=user, siwe_address=address, task_id=_TASK, adapter=adapter
        )
    )


def test_confirm_accepted_moves_to_preview(monkeypatch):
    memory = _install(monkeypatch, _row())

    assert _confirm(True)[0] == "ACCEPTED"
    assert memory["store"][_TASK].state == WorkflowState.PREVIEW.value


def test_confirm_rejected_aborts(monkeypatch):
    memory = _install(monkeypatch, _row())

    assert _confirm(False)[0] == "REJECTED"
    assert memory["store"][_TASK].state == WorkflowState.ABORTED.value


def test_confirm_is_idempotent_while_already_accepted(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value))

    assert _confirm(True)[0] == "ACCEPTED"


def test_confirm_after_rejection_is_state_conflict(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.ABORTED.value))

    with pytest.raises(AppError) as exc:
        _confirm(True)
    assert exc.value.code == "E609"


def test_confirm_missing_task_is_not_found(monkeypatch):
    _install(monkeypatch)

    with pytest.raises(AppError) as exc:
        _confirm(True)
    assert exc.value.code == "E504"


def test_confirm_other_wallet_is_forbidden(monkeypatch):
    _install(monkeypatch, _row())

    with pytest.raises(AppError) as exc:
        _confirm(True, address=_OTHER)
    assert exc.value.code == "E505"


def test_confirm_expired_marks_unsigned_and_rejects(monkeypatch):
    memory = _install(
        monkeypatch,
        _row(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
    )

    with pytest.raises(AppError) as exc:
        _confirm(True)
    assert exc.value.code == "E506"
    row = memory["store"][_TASK]
    assert row.state == WorkflowState.EXPIRED_UNSIGNED.value
    assert row.expire_type == WorkflowState.EXPIRED_UNSIGNED.value


def test_terminal_state_is_not_reinterpreted_by_expiry(monkeypatch):
    """已拒绝的任务过期后仍是 ABORTED，不能变成「过期可重发」误导用户。"""
    memory = _install(
        monkeypatch,
        _row(
            state=WorkflowState.ABORTED.value,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        ),
    )

    assert _confirm(False)[0] == "REJECTED"
    assert memory["store"][_TASK].state == WorkflowState.ABORTED.value


def test_status_before_confirm_hides_tx_intent(monkeypatch):
    _install(monkeypatch, _row())

    data = _status(_Adapter())

    assert data["state"] == WorkflowState.INTENT_CONFIRM.value
    assert data["txIntent"] is None


def test_status_in_preview_exposes_tx_intent(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value))

    data = _status(_Adapter())

    assert data["state"] == WorkflowState.PREVIEW.value
    assert data["txIntent"] == _TX
    assert data["siweAddressShort"] == "0x111111…1111"


def _receipt(**overrides) -> dict:
    """构造一份与 _TX 对得上的成功回执。"""
    receipt = {
        "status": "0x1",
        "from": _TX["from"],
        "to": _TX["to"],
        "value": hex(int(_TX["value"])),
        "logs": [],
    }
    receipt.update(overrides)
    return receipt


def _erc20_intent(raw: int = 10**6) -> dict:
    """构造一笔 ERC20 转账意图：to 是代币合约，收款方编在 data 里。"""
    data = (
        "0xa9059cbb"
        + _TX["to"].lower().removeprefix("0x").rjust(64, "0")
        + format(raw, "x").rjust(64, "0")
    )
    return {"from": _ADDRESS, "to": _TOKEN, "value": "0", "data": data, "chainId": 80094}


def _transfer_log(*, to: str = _TX["to"], raw: int = 10**6, token: str = _TOKEN) -> dict:
    """构造一条 Transfer 事件日志。"""
    return {
        "address": token,
        "topics": [
            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
            "0x" + _ADDRESS.lower().removeprefix("0x").rjust(64, "0"),
            "0x" + to.lower().removeprefix("0x").rjust(64, "0"),
        ],
        "data": hex(raw),
    }


def test_status_with_receipt_settles(monkeypatch):
    memory = _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value, tx_hash=_HASH))
    adapter = _Adapter(receipt=_receipt(blockNumber="0x10"))

    data = _status(adapter)

    assert adapter.asked == [(80094, _HASH)]
    assert data["state"] == WorkflowState.SETTLED.value
    assert memory["store"][_TASK].state == WorkflowState.SETTLED.value


def test_status_with_reverted_receipt_fails(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value, tx_hash=_HASH))

    data = _status(_Adapter(receipt=_receipt(status="0x0")))

    assert data["state"] == WorkflowState.FAILED.value
    assert data["errorCode"] == "E401"


def test_status_receipt_from_another_wallet_is_not_settled(monkeypatch):
    """别人那笔成功交易的哈希不能把这笔记成完成。"""
    memory = _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value, tx_hash=_HASH))

    data = _status(_Adapter(receipt={**_receipt(), "from": _OTHER, "to": _OTHER}))

    assert data["state"] == WorkflowState.SUBMITTED.value
    assert data["errorCode"] == "E406"
    assert memory["store"][_TASK].state == WorkflowState.SUBMITTED.value


def test_status_receipt_with_other_amount_is_not_settled(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value, tx_hash=_HASH))

    data = _status(_Adapter(receipt=_receipt(value=hex(10**17 * 2))))

    assert data["state"] == WorkflowState.SUBMITTED.value
    assert data["errorCode"] == "E406"


def test_status_erc20_receipt_matched_by_transfer_log(monkeypatch):
    _install(
        monkeypatch,
        _row(
            state=WorkflowState.PREVIEW.value,
            tx_hash=_HASH,
            tx_intent=_erc20_intent(),
        ),
    )

    data = _status(
        _Adapter(
            receipt=_receipt(to=_TOKEN, value="0x0", logs=[_transfer_log()]),
            calldata=_erc20_intent()["data"],
        )
    )

    assert data["state"] == WorkflowState.SETTLED.value


def test_status_erc20_transfer_to_another_recipient_is_not_settled(monkeypatch):
    _install(
        monkeypatch,
        _row(
            state=WorkflowState.PREVIEW.value,
            tx_hash=_HASH,
            tx_intent=_erc20_intent(),
        ),
    )

    data = _status(
        _Adapter(
            receipt=_receipt(to=_TOKEN, value="0x0", logs=[_transfer_log(to=_OTHER)]),
            calldata=_erc20_intent()["data"],
        )
    )

    assert data["state"] == WorkflowState.SUBMITTED.value
    assert data["errorCode"] == "E406"


def test_status_broadcast_without_receipt_reports_submitted(monkeypatch):
    memory = _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value, tx_hash=_HASH))

    data = _status(_Adapter(receipt=None))

    assert data["state"] == WorkflowState.SUBMITTED.value
    assert memory["store"][_TASK].state == WorkflowState.SUBMITTED.value


def test_status_rpc_failure_keeps_task_open(monkeypatch):
    """RPC 挂了不能把任务判死：已上报哈希即已广播，状态说准、错误码如实带回。"""
    memory = _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value, tx_hash=_HASH))

    data = _status(_Adapter(error=RuntimeError("rpc down")))

    assert data["state"] == WorkflowState.SUBMITTED.value
    assert data["errorCode"] == "E301"
    assert memory["store"][_TASK].state == WorkflowState.SUBMITTED.value


def test_status_terminal_state_skips_rpc(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.SETTLED.value, tx_hash=_HASH))
    adapter = _Adapter(receipt={"status": "0x1"})

    data = _status(adapter)

    assert adapter.asked == []
    assert data["state"] == WorkflowState.SETTLED.value


def test_status_expired_before_broadcast(monkeypatch):
    _install(
        monkeypatch,
        _row(state=WorkflowState.PREVIEW.value, expires_at=datetime.now(timezone.utc)),
    )

    data = _status(_Adapter())

    assert data["state"] == WorkflowState.EXPIRED_UNSIGNED.value
    assert data["expireType"] == "EXPIRED_UNSIGNED"


def test_status_expired_unconfirmed_task_is_closed(monkeypatch):
    """没人点确认的任务也要能走到终态，否则会一直挂在 INTENT_CONFIRM。"""
    _install(monkeypatch, _row(expires_at=datetime.now(timezone.utc)))

    data = _status(_Adapter())

    assert data["state"] == WorkflowState.EXPIRED_UNSIGNED.value


def test_status_expired_after_broadcast_is_unverified(monkeypatch):
    _install(
        monkeypatch,
        _row(
            state=WorkflowState.SUBMITTED.value,
            tx_hash=_HASH,
            expires_at=datetime.now(timezone.utc),
        ),
    )

    data = _status(_Adapter(receipt=None))

    assert data["state"] == WorkflowState.EXPIRED_UNVERIFIED.value


def test_status_expired_but_receipt_arrives_still_settles(monkeypatch):
    """到期前必须最后核对一次，否则会把已成功的交易判成未知。"""
    _install(
        monkeypatch,
        _row(
            state=WorkflowState.SUBMITTED.value,
            tx_hash=_HASH,
            expires_at=datetime.now(timezone.utc),
        ),
    )

    data = _status(_Adapter(receipt=_receipt()))

    assert data["state"] == WorkflowState.SETTLED.value


def test_status_expired_with_late_hash_settles(monkeypatch):
    """过期的任务补登哈希后仍要核回执：链上真转了不能停在「未签名」上。"""
    memory = _install(
        monkeypatch,
        _row(state=WorkflowState.EXPIRED_UNSIGNED.value, tx_hash=_HASH),
    )

    data = _status(_Adapter(receipt=_receipt()))

    assert data["state"] == WorkflowState.SETTLED.value
    assert memory["store"][_TASK].state == WorkflowState.SETTLED.value


def test_status_expired_with_unrelated_hash_keeps_state(monkeypatch):
    memory = _install(
        monkeypatch,
        _row(state=WorkflowState.EXPIRED_UNVERIFIED.value, tx_hash=_HASH),
    )

    data = _status(_Adapter(receipt={**_receipt(), "to": _OTHER}))

    assert data["state"] == WorkflowState.EXPIRED_UNVERIFIED.value
    assert data["errorCode"] == "E406"
    assert memory["store"][_TASK].state == WorkflowState.EXPIRED_UNVERIFIED.value


def test_report_records_hash_without_touching_state(monkeypatch):
    memory = _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value))

    async def _call():
        return await service.report(
            siwe_user_id=_USER, siwe_address=_ADDRESS, task_id=_TASK, tx_hash=_HASH
        )

    assert asyncio.run(_call()) is True
    assert memory["store"][_TASK].tx_hash == _HASH
    assert memory["store"][_TASK].state == WorkflowState.PREVIEW.value
    assert [c for c in memory["calls"] if c[0] == "transition"] == []


def test_report_replaces_hash_when_wallet_speeds_up(monkeypatch):
    """加速/取消会换一笔新哈希，链上只会有新的那一笔，旧哈希永远等不到回执。"""
    new_hash = "0x" + "b" * 64
    memory = _install(
        monkeypatch, _row(state=WorkflowState.SUBMITTED.value, tx_hash=_HASH)
    )

    async def _call():
        return await service.report(
            siwe_user_id=_USER,
            siwe_address=_ADDRESS,
            task_id=_TASK,
            tx_hash=new_hash,
        )

    assert asyncio.run(_call()) is False
    assert memory["store"][_TASK].tx_hash == new_hash


def test_report_on_expired_task_records_late_hash(monkeypatch):
    """过期后才广播的交易也要能认领：只补登哈希，状态交给回执核对。"""
    memory = _install(monkeypatch, _row(state=WorkflowState.EXPIRED_UNSIGNED.value))

    async def _call():
        return await service.report(
            siwe_user_id=_USER, siwe_address=_ADDRESS, task_id=_TASK, tx_hash=_HASH
        )

    assert asyncio.run(_call()) is True
    assert memory["store"][_TASK].tx_hash == _HASH
    assert memory["store"][_TASK].state == WorkflowState.EXPIRED_UNSIGNED.value


def test_report_on_settled_task_is_state_conflict(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.SETTLED.value))

    async def _call():
        return await service.report(
            siwe_user_id=_USER, siwe_address=_ADDRESS, task_id=_TASK, tx_hash=_HASH
        )

    with pytest.raises(AppError) as exc:
        asyncio.run(_call())
    assert exc.value.code == "E609"


def _request(container=None) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                container=container
                or SimpleNamespace(
                    adapter=_Adapter(), audit=SimpleNamespace(log=AsyncMock())
                )
            )
        )
    )


def _identity(address: str = _ADDRESS) -> SimpleNamespace:
    return SimpleNamespace(user_id=_USER, address=address, chain_id=80094)


def test_get_workflow_renders_camel_case(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value))

    resp = asyncio.run(
        workflow_router.get_workflow(_TASK, _request(), _identity())
    ).model_dump(by_alias=True)

    assert resp["taskId"] == _TASK
    assert resp["state"] == WorkflowState.PREVIEW.value
    assert resp["expireType"] == "NONE"
    assert resp["txIntent"] == _TX
    assert isinstance(resp["expiresAt"], int)


def test_report_endpoint_logs_audit_only(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.PREVIEW.value))
    container = SimpleNamespace(
        adapter=_Adapter(), audit=SimpleNamespace(log=AsyncMock())
    )
    payload = SimpleNamespace(
        task_id=_TASK, tx_hash=_HASH, status="broadcast"
    )

    resp = asyncio.run(
        workflow_router.report_tx(payload, _request(container), _identity())
    )

    assert resp.success is True
    assert resp.msg == "event logged, status sourced from backend RPC receipt"
    container.audit.log.assert_awaited_once()
    assert container.audit.log.await_args.args[0] == "tx_reported"


def test_confirm_resumes_the_plan_behind_the_task(monkeypatch):
    row = _row(state=WorkflowState.INTENT_CONFIRM.value)
    row.params = {**row.params, "plan_thread_id": "thread-1"}
    _install(monkeypatch, row)
    evidence = [{"symbol": "BERA", "actual": "12.4", "op": "gte", "right": "10"}]
    agent = SimpleNamespace(resume_plan=AsyncMock(return_value=evidence))
    container = SimpleNamespace(
        adapter=_Adapter(), audit=SimpleNamespace(log=AsyncMock()), agent=agent
    )

    asyncio.run(
        workflow_router.confirm_intent(
            SimpleNamespace(task_id=_TASK, accepted=False),
            _request(container),
            _identity(),
        )
    )

    agent.resume_plan.assert_awaited_once_with("thread-1")
    # 审计里要留下批准时的余额依据，事后才能解释这笔钱为什么发出去。
    assert container.audit.log.await_args.kwargs["evidence"] == evidence


def test_confirm_skips_resume_for_single_step_tasks(monkeypatch):
    _install(monkeypatch, _row(state=WorkflowState.INTENT_CONFIRM.value))
    agent = SimpleNamespace(resume_plan=AsyncMock(return_value=None))
    container = SimpleNamespace(
        adapter=_Adapter(), audit=SimpleNamespace(log=AsyncMock()), agent=agent
    )

    asyncio.run(
        workflow_router.confirm_intent(
            SimpleNamespace(task_id=_TASK, accepted=True),
            _request(container),
            _identity(),
        )
    )

    agent.resume_plan.assert_not_awaited()
    assert "evidence" not in container.audit.log.await_args.kwargs
