"""user_profile 读写。单主键，只存枚举与结构化值。

偏好是「补丁式」更新：一句话里只提到风格，就不能把之前记下的语言清掉，
所以写入一律是「读当前 → 合并 → 写回」，不是整行覆盖。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.domain.memory.profile import Profile, ProfilePatch
from web3_agent.persistence.pg.orm_models import UserProfile

# 档案里留 10 个，注入时只取最近用过的几个。留多一些是为了让长尾偏好不至于立刻被挤掉。
_MAX_TOKENS_STORED = 10


async def get(db: AsyncSession, siwe_user_id: str) -> Profile:
    row = await db.get(UserProfile, siwe_user_id)
    if row is None:
        return Profile()
    return Profile(
        answer_style=row.answer_style,
        language=row.language,
        preferred_chain_id=row.preferred_chain_id,
        risk_preference=row.risk_preference,
        frequent_tokens=tuple(row.frequent_tokens or ()),
    )


async def apply_patch(
    db: AsyncSession,
    *,
    siwe_user_id: str,
    patch: ProfilePatch,
    chain_id: int | None,
) -> Profile:
    """把偏好补丁合并进档案，返回合并后的结果。

    补丁为空时只读不写 —— 绝大多数请求都走这条路径，不该为「什么都没变」付一次写。
    """
    current = await get(db, siwe_user_id)
    if patch.is_empty:
        return current

    tokens = _merge_tokens(current.frequent_tokens, patch.tokens, chain_id)
    merged = Profile(
        answer_style=patch.answer_style or current.answer_style,
        language=patch.language or current.language,
        preferred_chain_id=(
            patch.preferred_chain_id
            if patch.preferred_chain_id is not None
            else current.preferred_chain_id
        ),
        risk_preference=patch.risk_preference or current.risk_preference,
        frequent_tokens=tuple(tokens),
    )

    stmt = (
        insert(UserProfile)
        .values(
            siwe_user_id=siwe_user_id,
            answer_style=merged.answer_style,
            language=merged.language,
            preferred_chain_id=merged.preferred_chain_id,
            risk_preference=merged.risk_preference,
            frequent_tokens=list(merged.frequent_tokens),
        )
        .on_conflict_do_update(
            index_elements=["siwe_user_id"],
            set_={
                "answer_style": merged.answer_style,
                "language": merged.language,
                "preferred_chain_id": merged.preferred_chain_id,
                "risk_preference": merged.risk_preference,
                "frequent_tokens": list(merged.frequent_tokens),
            },
        )
    )
    await db.execute(stmt)
    return merged


def _merge_tokens(
    existing: tuple[dict, ...], symbols: tuple[str, ...], chain_id: int | None
) -> list[dict]:
    """同名代币留在同一链上，只更新时间戳，不重复追加。

    顺序靠「最近一次插到最前」维持，不靠时间戳排序：同一毫秒内的两次写入
    时间戳相同，排序结果就是随机的。
    """
    kept = [dict(t) for t in existing if isinstance(t, dict) and t.get("symbol")]
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    for symbol in symbols:
        target = chain_id
        if target is None:
            target = next(
                (t.get("chainId") for t in kept if t.get("symbol") == symbol), None
            )
        if target is None:
            continue  # 不知道是哪条链的代币，记下来也没有意义
        kept = [
            t for t in kept if not (t.get("symbol") == symbol and t.get("chainId") == target)
        ]
        kept.insert(0, {"chainId": target, "symbol": symbol, "lastUsedAt": now})

    return kept[:_MAX_TOKENS_STORED]
