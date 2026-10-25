"""写链路兜底扫描：只扫该扫的任务，单条失败不阻塞其余。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from web3_agent.config.constants import WorkflowState
from web3_agent.domain.workflow import sweeper
from web3_agent.persistence.pg.repositories import workflow_repo
from tests.unit.test_chat_idempotency_route import _Session

# _USER：任务归属用户。
_USER = "user-1"
# _ADDRESS：任务归属钱包。
_ADDRESS = "0x" + "1" * 40
# _HASH：已广播的交易哈希。
_HASH = "0x" + "a" * 64


def _row(task_id: str, *, state: WorkflowState = WorkflowState.SUBMITTED):
    """构造一条待核对的任务行。"""
    return SimpleNamespace(
        task_id=task_id,
        siwe_user_id=_USER,
        siwe_address=_ADDRESS,
        state=state.value,
        tx_hash=_HASH,
    )


def _install(monkeypatch, rows, *, failing=(), moved=()):
    """把仓储与状态读取换成替身，并记录被扫到的任务。"""
    asked: list[str] = []
    listed: dict = {}

    async def fake_list(db, *, states, stale_before, limit):
        listed.update(states=states, stale_before=stale_before, limit=limit)
        return list(rows)[:limit]

    async def fake_status(*, siwe_user_id, siwe_address, task_id, adapter):
        asked.append(task_id)
        if task_id in failing:
            raise RuntimeError("rpc down")
        state = WorkflowState.SETTLED if task_id in moved else WorkflowState.SUBMITTED
        return {"state": state.value}

    monkeypatch.setattr(sweeper, "session_scope", lambda: _Session())
    monkeypatch.setattr(workflow_repo, "list_pending", fake_list)
    monkeypatch.setattr(sweeper.service, "read_status", fake_status)
    return {"asked": asked, "listed": listed}


def _sweep(batch: int = 50, stale: int = 60) -> int:
    return asyncio.run(sweeper.sweep_once(object(), batch=batch, stale_seconds=stale))


def test_sweep_counts_only_tasks_that_moved(monkeypatch):
    memory = _install(
        monkeypatch, [_row("t1"), _row("t2")], moved={"t1"}
    )

    assert _sweep() == 1
    assert memory["asked"] == ["t1", "t2"]


def test_single_failure_does_not_block_the_rest(monkeypatch):
    memory = _install(
        monkeypatch, [_row("t1"), _row("t2")], failing={"t1"}, moved={"t2"}
    )

    assert _sweep() == 1
    assert memory["asked"] == ["t1", "t2"]


def test_open_states_and_batch_are_passed_through(monkeypatch):
    memory = _install(monkeypatch, [_row("t1")])

    _sweep(batch=7)

    assert memory["listed"]["states"] == (
        WorkflowState.INTENT_CONFIRM.value,
        WorkflowState.PREVIEW.value,
        WorkflowState.SUBMITTED.value,
    )
    assert memory["listed"]["limit"] == 7


def test_no_rows_means_no_receipt_calls(monkeypatch):
    memory = _install(monkeypatch, [])

    assert _sweep() == 0
    assert memory["asked"] == []
