"""聊天幂等键：同一用户同一个 key 只跑一轮，客户端重试取回上一次的响应。

占位、写回、释放各是一次短事务。真正的去重靠主键冲突，不靠应用层判断 ——
两个并发请求同时进来时只有一方能插入成功。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert

from web3_agent.persistence.pg.orm_models import ChatIdempotency as Row

# 同一个 key 的响应保留一天。逾期未再出现的记录在下次带 key 的请求里顺手删掉。
# KEEP_SECONDS：聊天幂等响应保留时长，单位为秒。
KEEP_SECONDS = 24 * 3600


def _now() -> datetime:
    """返回当前带 UTC 时区的时间，作为仓储时间判断基准。"""
    return datetime.now(timezone.utc)


async def claim(
    db, *, siwe_user_id: str, key: str, request_hash: str, stale_after: float
) -> tuple[str, dict | None]:
    """占位。返回 (state, response)。

    - ``new``：抢到了，正常往下跑
    - ``done``：同一请求已经跑完，response 是当初的响应
    - ``running``：同一请求还在路上
    - ``conflict``：同一个 key 换了内容，不能拿旧答案顶替新问题

    ``stale_after`` 是「占位最长可以占多久」。进程在 complete 之前被杀掉时，
    行会永远停在 running，客户端拿着同一个 key 一天内都重试不了 —— 所以超过
    一轮实际耗时上限的 running 行当作孤儿，直接接管。正常并发重试不会踩到它，
    因为真实的一轮远短于这个窗口。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        key: 客户端提交的幂等键。
        request_hash: 请求内容的稳定摘要，防止幂等键内容复用。
        stale_after: 允许 running 占位存活的最大秒数，超过后可接管。

    Returns:
        tuple[str, dict | None]: 占位状态和已完成响应；状态为 new、done、running 或 conflict。
    """
    now = _now()
    await db.execute(
        delete(Row).where(
            Row.siwe_user_id == siwe_user_id,
            Row.created_at < now - timedelta(seconds=KEEP_SECONDS),
        )
    )
    inserted = await db.execute(
        insert(Row)
        .values(
            siwe_user_id=siwe_user_id,
            idempotency_key=key,
            request_hash=request_hash,
            status="running",
            created_at=now,
        )
        .on_conflict_do_nothing()
    )
    if inserted.rowcount:
        return "new", None
    row = (
        await db.execute(
            select(Row).where(
                Row.siwe_user_id == siwe_user_id, Row.idempotency_key == key
            )
        )
    ).scalar_one()
    if row.request_hash != request_hash:
        return "conflict", None
    if row.status == "done":
        return "done", row.response
    # 接管孤儿占位。用一条 UPDATE ... WHERE created_at < cutoff 定胜负：
    # 两个并发重试只有一个能改到这一行，另一个照样拿 running。
    stolen = await db.execute(
        update(Row)
        .where(
            Row.siwe_user_id == siwe_user_id,
            Row.idempotency_key == key,
            Row.status == "running",
            Row.created_at < now - timedelta(seconds=stale_after),
        )
        .values(created_at=now)
    )
    return ("new", None) if stolen.rowcount else ("running", None)


async def complete(db, *, siwe_user_id: str, key: str, response: dict) -> None:
    """把指定聊天幂等记录更新为 done，并保存完整响应快照。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        key: 客户端提交的幂等键。
        response: 要持久保存的完整响应快照。
    """
    await db.execute(
        update(Row)
        .where(Row.siwe_user_id == siwe_user_id, Row.idempotency_key == key)
        .values(status="done", response=response)
    )


async def release(db, *, siwe_user_id: str, key: str) -> None:
    """本轮没跑成，把占位撤掉 —— 客户端拿同一个 key 重试要能真的重跑。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
        key: 客户端提交的幂等键。
    """
    await db.execute(
        delete(Row).where(
            Row.siwe_user_id == siwe_user_id,
            Row.idempotency_key == key,
            Row.status == "running",
        )
    )


async def purge_user(db, *, siwe_user_id: str) -> int:
    """清空/删除对话时一起丢掉该用户的幂等记录。

    不清的话，用户删完数据还能拿旧 key 换回删除前那条答案和它的引用 ——
    generation 只拦写回，拦不住这条读路径。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        siwe_user_id: 数据所属用户的稳定标识。
    """
    result = await db.execute(delete(Row).where(Row.siwe_user_id == siwe_user_id))
    return result.rowcount or 0
