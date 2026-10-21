"""PostgreSQL 连接与会话。整个进程只有这一处持有 engine。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from web3_agent.config.settings import Settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        return
    _engine = create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,
        echo=False,
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("engine 未初始化：请先调用 init_engine")
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all() -> None:
    from web3_agent.persistence.pg import orm_models  # noqa: F401  注册表定义

    if _engine is None:
        raise RuntimeError("engine 未初始化")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
