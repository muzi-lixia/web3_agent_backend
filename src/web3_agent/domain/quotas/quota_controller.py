"""QuotaController：限流与资源配额。

两个动作分开：
- `hit`  进一次就计数，超了抛 E501（用于 QPS、RPC 这类"进门就得拦"的维度）；
- `charge` 只记账不拦（用于事后才知道用量的维度，比如 ReAct 步数、token）。

超限统一 429：PRD 12 章写的是"超限降级只读"，但降级意味着有写链路可降。
写链路上线后这里改成只拦写意图，拦截点不变。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from web3_agent.common.exceptions import E501_QUOTA_EXHAUSTED, AppError
from web3_agent.common.types import BudgetSnapshot
from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import Settings
from web3_agent.persistence.pg.repositories import quota_repo


class QuotaController:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def key_for(self, *, user_id: str | None, client_ip: str) -> str:
        """登录用户按 userId 归并（同一人多设备共享额度），匿名单 IP。"""
        return f"u:{user_id}" if user_id else f"ip:{client_ip}"

    async def hit(
        self,
        db: AsyncSession,
        *,
        key: str,
        metric: QuotaMetric,
        limit: int,
        amount: int = 1,
    ) -> int:
        used = await quota_repo.incr(db, user_key=key, metric=metric, amount=amount)
        if used > limit:
            raise AppError(*E501_QUOTA_EXHAUSTED, retryable=True)
        return used

    async def charge(
        self, db: AsyncSession, *, key: str, metric: QuotaMetric, amount: int
    ) -> None:
        if amount <= 0:
            return
        await quota_repo.incr(db, user_key=key, metric=metric, amount=amount)

    async def snapshot(self, db: AsyncSession, *, key: str) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps=await quota_repo.read(db, user_key=key, metric=QuotaMetric.STEPS),
            tokens=await quota_repo.read(db, user_key=key, metric=QuotaMetric.TOKENS),
            rpcCalls=await quota_repo.read(db, user_key=key, metric=QuotaMetric.RPC_CALLS),
            txCount=await quota_repo.read(db, user_key=key, metric=QuotaMetric.TX_COUNT),
        )

    async def check_request(
        self, db: AsyncSession, *, key: str, logged_in: bool, auth_endpoint: bool = False
    ) -> None:
        """请求入口。QPS 当场计数并拦，RPC/步数看上一轮攒下多少，已满就不放进来。"""
        rpm = (
            self._settings.quota_auth_rpm
            if auth_endpoint
            else self._settings.quota_user_rpm
            if logged_in
            else self._settings.quota_anon_rpm
        )
        await self.hit(db, key=key, metric=QuotaMetric.QPS, limit=rpm)
        for metric, limit in (
            (QuotaMetric.RPC_CALLS, self._settings.quota_rpc_per_minute),
            (QuotaMetric.STEPS, self._settings.quota_steps_per_minute),
        ):
            used = await quota_repo.read(db, user_key=key, metric=metric)
            if used >= limit:
                raise AppError(*E501_QUOTA_EXHAUSTED, retryable=True)
