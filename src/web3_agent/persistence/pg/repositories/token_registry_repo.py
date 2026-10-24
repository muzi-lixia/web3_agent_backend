"""token_registry 读取。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.common.types import TokenMeta
from web3_agent.persistence.pg.orm_models import TokenRegistry


async def list_whitelisted(db: AsyncSession, chain_id: int) -> list[TokenMeta]:
    """读取指定网络的白名单代币，并转为领域 TokenMeta。

    Args:
        db: 调用方提供的异步数据库会话；事务提交由外层负责。
        chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
    """
    stmt = (
        select(TokenRegistry)
        .where(TokenRegistry.chain_id == chain_id, TokenRegistry.is_whitelisted.is_(True))
        .order_by(TokenRegistry.symbol)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        TokenMeta(
            chain_id=r.chain_id,
            symbol=r.symbol,
            decimals=r.decimals,
            token_address=r.token_address,
            is_whitelisted=r.is_whitelisted,
            name=r.name,
        )
        for r in rows
    ]
