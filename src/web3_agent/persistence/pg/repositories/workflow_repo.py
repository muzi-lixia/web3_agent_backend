"""写操作任务仓储：建任务、取任务、推进状态。

状态推进用条件更新（`WHERE state = 期望值`）：两个并发轮询同时看到 PREVIEW 时，
只有一个能把状态改到下一档，另一个影响 0 行并按新状态返回。条件更新只锁一行、
不执行 RPC，所以能在短事务里完成，不必为一次回执核对占住数据库连接。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, or_, select, update

from web3_agent.persistence.pg.orm_models import WorkflowTask as Row


async def create(
    db,
    *,
    task_id: str,
    siwe_user_id: str,
    siwe_address: str,
    action: str,
    chain_id: int,
    state: str,
    params: dict,
    tx_intent: dict | None,
    expires_at: datetime,
) -> None:
    """写入一个新的写操作任务。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        task_id: 对外暴露的任务标识。
        siwe_user_id: 任务归属用户的稳定标识。
        siwe_address: 任务归属钱包的完整地址。
        action: 业务动作名，取自写意图。
        chain_id: 写操作所在网络编号。
        state: 初始生命周期状态。
        params: 构造交易用的原始业务参数。
        tx_intent: 待签名交易参数；尚未构造时为空。
        expires_at: 任务过期时间。
    """
    db.add(
        Row(
            task_id=task_id,
            siwe_user_id=siwe_user_id,
            siwe_address=siwe_address,
            action=action,
            chain_id=chain_id,
            state=state,
            expire_type="NONE",
            params=params,
            tx_intent=tx_intent,
            expires_at=expires_at,
        )
    )


async def get(db, task_id: str) -> Row | None:
    """按任务标识读取任务行。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        task_id: 对外暴露的任务标识。
    """
    return (
        await db.execute(select(Row).where(Row.task_id == task_id))
    ).scalar_one_or_none()


async def transition(
    db,
    task_id: str,
    *,
    expect: str,
    state: str,
    expire_type: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> bool:
    """把任务从 expect 状态推进到 state，返回是否真正改到了这一行。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        task_id: 对外暴露的任务标识。
        expect: 推进前的状态；与当前值不符时不改动，用于并发去重。
        state: 目标状态。
        expire_type: 过期原因；不改动时传空。
        error_code: 稳定错误码；不改动时传空。
        error_message: 用户可读的错误文案；不改动时传空。
    """
    values: dict = {"state": state}
    if expire_type is not None:
        values["expire_type"] = expire_type
    if error_code is not None:
        values["error_code"] = error_code
    if error_message is not None:
        values["error_message"] = error_message
    result = await db.execute(
        update(Row)
        .where(Row.task_id == task_id, Row.state == expect)
        .values(**values)
    )
    return bool(result.rowcount)


async def list_pending(
    db, *, states: tuple[str, ...], stale_before: datetime, limit: int
) -> list[Row]:
    """取出一批需要兜底推进的任务：已广播的，或已过期但从未广播的。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        states: 需要推进的状态集合。
        stale_before: 只取 updated_at 早于该时刻的行，避免刚推进过又扫一遍。
        limit: 单次最多返回的行数。
    """
    stmt = (
        select(Row)
        .where(
            Row.state.in_(states),
            # 有哈希的要持续核对回执；没哈希的只在过期后才需要结案，不必每分钟空扫。
            or_(Row.tx_hash.is_not(None), Row.expires_at <= func.now()),
            Row.updated_at < stale_before,
        )
        .order_by(Row.updated_at.asc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def set_tx_hash(db, task_id: str, tx_hash: str) -> bool:
    """记录最近一次上报的广播哈希，返回本次是否为首笔。

    钱包的「加速」「取消」会用同一个 nonce 再发一笔替换交易，链上最终生效的是新
    哈希，只认第一笔就会永远等一个不会上链的哈希。哈希只是查询入口，内容是否属于
    本任务由回执核对，所以这里允许覆盖。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        task_id: 对外暴露的任务标识。
        tx_hash: 前端上报的广播交易哈希。
    """
    current = await db.execute(select(Row.tx_hash).where(Row.task_id == task_id))
    first = current.scalar_one_or_none() is None
    await db.execute(update(Row).where(Row.task_id == task_id).values(tx_hash=tx_hash))
    return first
