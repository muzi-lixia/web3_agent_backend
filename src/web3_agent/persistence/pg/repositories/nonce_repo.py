"""siwe_nonce 读写：签发、核销、清理。

核销必须是单条 UPDATE，不能用「先查再改」：两个请求同时带着同一个 nonce 进来时，
只有一条 UPDATE 能命中 consumed_at IS NULL，另一条返回空行——这就是单次消费的保证。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.orm_models import SiweNonce


def _now() -> datetime:
    """返回当前带 UTC 时区的时间，作为仓储时间判断基准。"""
    return datetime.now(timezone.utc)


async def issue(
    db: AsyncSession,
    *,
    nonce: str,
    ttl_seconds: int,
    address: str | None = None,
    chain_id: int | None = None,
) -> datetime:
    """将新 nonce 与有效期、钱包和网络绑定后写入当前事务。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        nonce: 一次性登录挑战字符串。
        ttl_seconds: 挑战、凭据或租约的有效秒数。
        address: 待校验、登录或查询的钱包地址。
        chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
    """
    row = SiweNonce(
        nonce=nonce,
        address=address.lower() if address else None,
        chain_id=chain_id,
        expires_at=_now() + timedelta(seconds=ttl_seconds),
    )
    db.add(row)
    await db.flush()
    return row.expires_at


async def peek(db: AsyncSession, nonce: str) -> SiweNonce | None:
    """读取 nonce 记录，供核对绑定的钱包、网络和时效。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        nonce: 一次性登录挑战字符串。
    """
    stmt = select(SiweNonce).where(SiweNonce.nonce == nonce)
    return (await db.execute(stmt)).scalar_one_or_none()


async def consume(db: AsyncSession, nonce: str) -> str | None:
    """核销成功返回绑定地址（None 表示该 nonce 未绑定地址），失败返回空串。

    返回空串与返回 None 语义不同：空串 = 核销失败，None = 核销成功但未绑定地址。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        nonce: 一次性登录挑战字符串。
    """
    stmt = (
        update(SiweNonce)
        .where(
            SiweNonce.nonce == nonce,
            SiweNonce.consumed_at.is_(None),
            SiweNonce.expires_at > _now(),
        )
        .values(consumed_at=_now())
        .returning(SiweNonce.address)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        return ""
    return row[0]


async def cleanup(db: AsyncSession, *, older_than_seconds: int = 86400) -> int:
    """删除超过保留期限的记录或配额桶，并返回影响行数。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        older_than_seconds: 保留窗口秒数，更旧的记录可被清理。
    """
    cutoff = _now() - timedelta(seconds=older_than_seconds)
    result = await db.execute(delete(SiweNonce).where(SiweNonce.expires_at < cutoff))
    return int(result.rowcount or 0)
