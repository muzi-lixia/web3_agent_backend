"""conversation_summary 读写。一个用户一行，水位线保证幂等。

写回一律带 `WHERE last_summarized_message_id = <读到的旧值>`：不匹配说明
另一个任务已经推进过水位线，本次结果作废，不覆盖也不报错。
"""

from __future__ import annotations

from sqlalchemy import delete as sql_delete
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.persistence.pg.orm_models import ConversationSummary


async def get(db: AsyncSession, siwe_user_id: str) -> ConversationSummary | None:
    return await db.get(ConversationSummary, siwe_user_id)


async def upsert_with_watermark(
    db: AsyncSession,
    *,
    siwe_user_id: str,
    content: str,
    last_summarized_message_id: int,
    expected_watermark: int | None,
) -> bool:
    """写回摘要。`expected_watermark=None` 表示首次生成。

    已有摘要时只做 UPDATE，不做 INSERT：清空接口刚把这一行删掉的话，
    这里必须写不进去，否则「刚清空的内容」会被后台任务复活。
    返回 False 表示本次结果作废。
    """
    if expected_watermark is None:
        stmt = (
            insert(ConversationSummary)
            .values(
                siwe_user_id=siwe_user_id,
                content=content,
                last_summarized_message_id=last_summarized_message_id,
                version=1,
                status="ok",
                fail_count=0,
            )
            .on_conflict_do_nothing(index_elements=["siwe_user_id"])
        )
        return ((await db.execute(stmt)).rowcount or 0) > 0

    stmt = (
        update(ConversationSummary)
        .where(
            ConversationSummary.siwe_user_id == siwe_user_id,
            ConversationSummary.last_summarized_message_id == expected_watermark,
        )
        .values(
            content=content,
            last_summarized_message_id=last_summarized_message_id,
            version=ConversationSummary.version + 1,
            status="ok",
            fail_count=0,
        )
    )
    return ((await db.execute(stmt)).rowcount or 0) > 0


async def mark_failed(db: AsyncSession, siwe_user_id: str) -> None:
    """失败只累加计数：不动水位线，也不清掉上一次成功的 content。"""
    stmt = (
        insert(ConversationSummary)
        .values(
            siwe_user_id=siwe_user_id,
            content="",
            last_summarized_message_id=0,
            version=0,
            status="failed",
            fail_count=1,
        )
        .on_conflict_do_update(
            index_elements=["siwe_user_id"],
            set_={"status": "failed", "fail_count": ConversationSummary.fail_count + 1},
        )
    )
    await db.execute(stmt)


async def delete(db: AsyncSession, siwe_user_id: str) -> int:
    """本模块有自己的 delete()，所以 sqlalchemy 的 delete 导入时改名。"""
    result = await db.execute(
        sql_delete(ConversationSummary).where(ConversationSummary.siwe_user_id == siwe_user_id)
    )
    return result.rowcount or 0
