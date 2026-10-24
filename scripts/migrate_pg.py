"""建表脚本。

create_all 只建缺失的表，不改既有表，也不补索引：
结构变更需要手写 DDL 执行，向量索引另见 scripts/rag_vector_index.py。

用法：python scripts/migrate_pg.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web3_agent.config.settings import get_settings  # noqa: E402
from web3_agent.persistence.pg.base import create_all, dispose_engine, init_engine  # noqa: E402


async def main() -> None:
    """加载配置并创建数据库中缺失的表，结束后关闭引擎。"""
    settings = get_settings()
    init_engine(settings)
    await create_all()
    print(f"tables created on {settings.database_url}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
