"""给既有库补 rag_chunk 的向量索引。

create_all 只建表、不补索引，老库跑过 migrate_pg 之后仍然是全表顺序扫描。
CONCURRENTLY 不能进事务，所以单独一个脚本，用 autocommit 连接执行。

用法：python scripts/rag_vector_index.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from web3_agent.config.settings import get_settings  # noqa: E402

_DDL = """
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_rag_chunk_embedding
ON rag_chunk USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
"""


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(text(_DDL))
    finally:
        await engine.dispose()
    print("ix_rag_chunk_embedding ready")


if __name__ == "__main__":
    asyncio.run(main())
