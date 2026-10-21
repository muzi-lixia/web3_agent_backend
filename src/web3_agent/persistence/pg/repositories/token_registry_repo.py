"""token_registry 读取。数据由 scripts/init_registries.py 导入。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.common.types import TokenMeta
from web3_agent.persistence.pg.orm_models import TokenRegistry


async def list_whitelisted(db: AsyncSession, chain_id: int) -> list[TokenMeta]:
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
