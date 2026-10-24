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
from web3_agent.config.constants import QuotaMetric
from web3_agent.config.settings import Settings
from web3_agent.persistence.pg.repositories import quota_repo


# 数据职责：请求限流与资源记账的业务入口，具体分钟桶读写由配额仓储完成。
class QuotaController:
    def __init__(self, settings: Settings) -> None:
        """初始化 QuotaController 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self._settings = settings

    def key_for(self, *, user_id: str | None, client_ip: str) -> str:
        """登录用户按 userId 归并（同一人多设备共享额度），匿名单 IP。

        Args:
            user_id: 已登录用户的稳定标识；为空时按匿名主体处理。
            client_ip: 用于匿名限流的客户端地址。
        """
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
        """原子增加请求计数，并在超过限制时抛出配额错误。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            key: 用户或匿名 IP 对应的配额主体键。
            metric: 本次计数的配额维度。
            limit: 本次查询或展示返回条数上限。
            amount: 本次累加或预占的资源数量。
        """
        used = await quota_repo.incr(db, user_key=key, metric=metric, amount=amount)
        if used > limit:
            raise AppError(*E501_QUOTA_EXHAUSTED, retryable=True)
        return used

    async def charge(
        self, db: AsyncSession, *, key: str, metric: QuotaMetric, amount: int
    ) -> None:
        """仅增加已发生的资源用量，不在结算阶段再拦截请求。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            key: 用户或匿名 IP 对应的配额主体键。
            metric: 本次计数的配额维度。
            amount: 本次累加或预占的资源数量。
        """
        if amount <= 0:
            return
        await quota_repo.incr(db, user_key=key, metric=metric, amount=amount)

    async def check_request(
        self, db: AsyncSession, *, key: str, logged_in: bool, auth_endpoint: bool = False
    ) -> None:
        """请求入口。QPS 当场计数并拦，RPC/步数看上一轮攒下多少，已满就不放进来。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            key: 用户或匿名 IP 对应的配额主体键。
            logged_in: 主体是否已经登录，决定请求次数额度。
            auth_endpoint: 是否使用鉴权接口专用的请求额度。
        """
        rpm = (
            self._settings.quota_auth_rpm
            if auth_endpoint
            else self._settings.quota_user_rpm
            if logged_in
            else self._settings.quota_anon_rpm
        )
        await self.hit(db, key=key, metric=QuotaMetric.QPS, limit=rpm)
        limits = (
            (QuotaMetric.RPC_CALLS, self._settings.quota_rpc_per_minute),
            (QuotaMetric.STEPS, self._settings.quota_steps_per_minute),
        )
        counts = await quota_repo.read_many(
            db, user_key=key, metrics=tuple(metric for metric, _ in limits)
        )
        for metric, limit in limits:
            if counts[metric] >= limit:
                raise AppError(*E501_QUOTA_EXHAUSTED, retryable=True)
