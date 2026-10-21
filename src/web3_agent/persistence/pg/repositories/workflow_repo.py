"""workflow_persistent / workflow_tx_record 读写。领域模型与外网 ORM 的转换都在这里。

并发约定：同一 workflow 的推进必须先拿行锁（`get(..., for_update=True)`），
不依赖任何缓存里的会话标识。加锁范围只有主表一行，tx 记录随后读即可。
"""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.common.types import TxIntent, TxRecord, WorkflowStoreItem
from web3_agent.config.constants import (
    ExpireType,
    GroupStatus,
    TxKind,
    TxRecordStatus,
    WorkflowState,
)
from web3_agent.persistence.pg.orm_models import WorkflowPersistent, WorkflowTxRecord


def _to_tx_record(row: WorkflowTxRecord) -> TxRecord:
    return TxRecord(
        index=row.index,
        kind=TxKind(row.kind),
        tx_hash=row.tx_hash,
        status=TxRecordStatus(row.tx_status),
        nonce=row.nonce,
        tx_intent_snapshot=(
            TxIntent.model_validate(row.tx_intent_snapshot)
            if row.tx_intent_snapshot
            else None
        ),
    )


def _to_item(row: WorkflowPersistent, tx_rows: list[WorkflowTxRecord]) -> WorkflowStoreItem:
    return WorkflowStoreItem(
        workflow_id=row.workflow_id,
        task_id=row.task_id,
        status=WorkflowState(row.status),
        siwe_user_id=row.siwe_user_id,
        siwe_address=row.siwe_address,
        intent=row.intent,
        chain_id=row.chain_id,
        slots_snapshot=row.slots_snapshot or {},
        generated_tx_count=row.generated_tx_count,
        max_tx_count=row.max_tx_count,
        approve_fingerprints=row.approve_fingerprints or [],
        tx_records=[_to_tx_record(t) for t in sorted(tx_rows, key=lambda t: t.index)],
        transaction_group_id=row.transaction_group_id,
        group_status=GroupStatus(row.group_status),
        expire_at=row.expire_at,
        expire_type=ExpireType(row.expire_type),
    )


async def _load_tx_rows(db: AsyncSession, workflow_id: str) -> list[WorkflowTxRecord]:
    stmt = (
        select(WorkflowTxRecord)
        .where(WorkflowTxRecord.workflow_id == workflow_id)
        .order_by(WorkflowTxRecord.index)
    )
    return list((await db.execute(stmt)).scalars().all())


async def create(db: AsyncSession, item: WorkflowStoreItem) -> None:
    db.add(
        WorkflowPersistent(
            workflow_id=item.workflow_id,
            task_id=item.task_id,
            siwe_user_id=item.siwe_user_id,
            siwe_address=item.siwe_address,
            chain_id=item.chain_id,
            intent=item.intent.value,
            status=item.status.value,
            slots_snapshot=item.slots_snapshot,
            generated_tx_count=item.generated_tx_count,
            max_tx_count=item.max_tx_count,
            approve_fingerprints=item.approve_fingerprints,
            transaction_group_id=item.transaction_group_id,
            group_status=item.group_status.value,
            expire_at=item.expire_at,
            expire_type=item.expire_type.value,
        )
    )
    for record in item.tx_records:
        db.add(_new_tx_row(item.workflow_id, item.chain_id, record))
    await db.flush()


def _new_tx_row(workflow_id: str, chain_id: int, record: TxRecord) -> WorkflowTxRecord:
    return WorkflowTxRecord(
        workflow_id=workflow_id,
        index=record.index,
        kind=record.kind.value,
        tx_hash=record.tx_hash,
        tx_status=record.status.value,
        nonce=record.nonce,
        chain_id=chain_id,
        tx_intent_snapshot=(
            record.tx_intent_snapshot.model_dump(by_alias=True)
            if record.tx_intent_snapshot
            else None
        ),
    )


async def get(
    db: AsyncSession, *, task_id: str | None = None, workflow_id: str | None = None, for_update: bool = False
) -> WorkflowStoreItem | None:
    """按 taskId 或内部主键取 workflow。for_update=True 时对主表加行锁。"""
    if task_id is None and workflow_id is None:
        raise ValueError("task_id 与 workflow_id 至少给一个")

    stmt = select(WorkflowPersistent)
    stmt = stmt.where(
        WorkflowPersistent.task_id == task_id
        if task_id is not None
        else WorkflowPersistent.workflow_id == workflow_id
    )
    if for_update:
        stmt = stmt.with_for_update()
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    return _to_item(row, await _load_tx_rows(db, row.workflow_id))


async def update_state(
    db: AsyncSession,
    workflow_id: str,
    *,
    status: WorkflowState | None = None,
    group_status: GroupStatus | None = None,
    expire_type: ExpireType | None = None,
    error_code: str | None = None,
    error_msg: str | None = None,
) -> None:
    values: dict = {}
    if status is not None:
        values["status"] = status.value
    if group_status is not None:
        values["group_status"] = group_status.value
    if expire_type is not None:
        values["expire_type"] = expire_type.value
    if error_code is not None:
        values["error_code"] = error_code
    if error_msg is not None:
        values["error_msg"] = error_msg
    if not values:
        return
    await db.execute(
        update(WorkflowPersistent)
        .where(WorkflowPersistent.workflow_id == workflow_id)
        .values(**values)
    )


async def update_progress(
    db: AsyncSession,
    workflow_id: str,
    *,
    generated_tx_count: int,
    approve_fingerprints: list[str],
) -> None:
    """下发新一笔交易后更新计数与指纹。调用前应已持有行锁。"""
    await db.execute(
        update(WorkflowPersistent)
        .where(WorkflowPersistent.workflow_id == workflow_id)
        .values(
            generated_tx_count=generated_tx_count,
            approve_fingerprints=approve_fingerprints,
        )
    )


async def append_tx_record(
    db: AsyncSession, *, workflow_id: str, chain_id: int, record: TxRecord
) -> None:
    db.add(_new_tx_row(workflow_id, chain_id, record))
    await db.flush()


async def bind_tx_hash(
    db: AsyncSession, *, workflow_id: str, index: int, tx_hash: str, nonce: int | None
) -> None:
    """广播后立即回填 txHash 与 nonce，不等终态。nonce 是 REPLACED 判定的唯一依据。"""
    await db.execute(
        update(WorkflowTxRecord)
        .where(
            WorkflowTxRecord.workflow_id == workflow_id,
            WorkflowTxRecord.index == index,
        )
        .values(tx_hash=tx_hash, nonce=nonce, tx_status=TxRecordStatus.PENDING.value)
    )
    await db.execute(
        update(WorkflowPersistent)
        .where(WorkflowPersistent.workflow_id == workflow_id)
        .values(status=WorkflowState.SUBMITTED.value)
    )


async def update_tx_status(
    db: AsyncSession, *, workflow_id: str, index: int, status: TxRecordStatus
) -> None:
    await db.execute(
        update(WorkflowTxRecord)
        .where(
            WorkflowTxRecord.workflow_id == workflow_id,
            WorkflowTxRecord.index == index,
        )
        .values(tx_status=status.value)
    )


async def save_tx_intent_snapshot(
    db: AsyncSession, *, workflow_id: str, index: int, tx_intent: TxIntent
) -> None:
    """PREVIEW 幂等：快照只在首次构造时写入，重复查询直接读它，不重建。"""
    await db.execute(
        update(WorkflowTxRecord)
        .where(
            WorkflowTxRecord.workflow_id == workflow_id,
            WorkflowTxRecord.index == index,
        )
        .values(tx_intent_snapshot=tx_intent.model_dump(by_alias=True))
    )


async def list_submitted(
    db: AsyncSession, *, expire_at_gt: int, limit: int = 100
) -> list[WorkflowStoreItem]:
    """兜底扫描用：只扫 SUBMITTED，且未过期的。"""
    stmt = (
        select(WorkflowPersistent)
        .where(
            WorkflowPersistent.status == WorkflowState.SUBMITTED.value,
            WorkflowPersistent.expire_at > expire_at_gt,
        )
        .order_by(WorkflowPersistent.updated_at)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [_to_item(r, await _load_tx_rows(db, r.workflow_id)) for r in rows]
