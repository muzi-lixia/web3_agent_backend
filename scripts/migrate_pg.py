"""建表脚本。V1a 用 create_all 起步；写链路开工前换成 Alembic 管理版本。

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
    settings = get_settings()
    init_engine(settings)
    await create_all()
    print(f"tables created on {settings.database_url}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
