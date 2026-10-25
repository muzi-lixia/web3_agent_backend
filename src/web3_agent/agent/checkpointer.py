"""多跳图的检查点存储。

`interrupt` 把图挂在某一步等人确认，靠 checkpoint 保存当时的状态；没有它，挂起之后
的下一个请求就接不上。连接池与应用同生命周期，由 lifespan 创建和关闭。
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from web3_agent.config.settings import Settings

logger = logging.getLogger(__name__)


def conninfo(database_url: str) -> str:
    """把 SQLAlchemy 的连接串转成 psycopg 可直接使用的形式。

    Args:
        database_url: 应用配置里的数据库地址。
    """
    return make_url(database_url).set(drivername="postgresql").render_as_string(
        hide_password=False
    )


async def init_checkpointer(
    settings: Settings,
) -> tuple[AsyncPostgresSaver, AsyncConnectionPool] | None:
    """建立检查点连接池并确保表已就绪。

    失败只记日志并返回 None：多跳路径降级为追问，其余功能不受影响。

    Args:
        settings: 运行配置，提供数据库地址、连接池上限和是否允许建表。
    """
    pool: AsyncConnectionPool | None = None
    try:
        pool = AsyncConnectionPool(
            conninfo=conninfo(settings.database_url),
            max_size=settings.checkpoint_pool_size,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        if settings.auto_create_tables:
            await saver.setup()
    except Exception:  # noqa: BLE001 检查点不可用不该拦住服务启动
        logger.exception("checkpointer init failed, multi-hop disabled")
        if pool is not None:
            try:
                await pool.close()
            except Exception:  # noqa: BLE001 关不掉也不影响启动
                logger.warning("checkpointer pool close failed", exc_info=True)
        return None
    logger.info("checkpointer ready")
    return saver, pool
