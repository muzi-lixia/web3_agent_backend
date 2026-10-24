"""PostgreSQL 连接与会话。整个进程只有这一处持有 engine。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from web3_agent.config.settings import Settings


class Base(DeclarativeBase):
    pass


# _engine：进程共享的异步数据库引擎，启动时初始化、退出时关闭。
_engine: AsyncEngine | None = None
# _sessionmaker：绑定共享引擎的会话工厂，短事务按需创建会话。
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> None:
    """创建进程共享的异步数据库引擎和会话工厂，已初始化时直接返回。

    Args:
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
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
    """提供短事务会话，正常退出提交、异常退出回滚，并释放会话资源。"""
    if _sessionmaker is None:
        raise RuntimeError("engine 未初始化：请先调用 init_engine")
    async with _sessionmaker() as session:
        try:
            yield session
            # 成功退出上下文时提交本次短事务，使本批读写作为一个整体生效。
            await session.commit()
        except Exception:
            # 业务异常时撤销本次未提交写入，再把原异常交回调用方。
            await session.rollback()
            raise


async def create_all() -> None:
    """导入表定义后创建数据库中缺失的表，不升级已有字段和索引。"""
    from web3_agent.persistence.pg import orm_models  # noqa: F401  注册表定义

    if _engine is None:
        raise RuntimeError("engine 未初始化")
    async with _engine.begin() as conn:
        # 把 ORM 元数据交给同步建表器执行，只创建不存在的结构。
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """关闭共享连接池并清空全局会话工厂。"""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
