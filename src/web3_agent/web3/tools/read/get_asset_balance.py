"""T1 get_asset_balance：原生币 + 白名单 ERC20 余额，多链并发查询。"""

from __future__ import annotations

import asyncio
import logging
from decimal import ROUND_DOWN, Decimal

from langchain_core.tools import StructuredTool

from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import token_registry_repo
from web3_agent.web3.chain_adapter import ChainAdapter, is_evm_address

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "查询钱包地址原生币与ERC20代币余额；后端会话注入chainId，只读场景可遍历白名单链"
)


# 展示精度。raw 字段保留原始整数，这里只决定「给人看的那一串」：
# 满精度串会被模型逐位抄进回答，对用户没有信息量。
_PRECISION = Decimal("0.000001")


def _human(raw: int, decimals: int) -> str:
    value = Decimal(raw).scaleb(-decimals)
    if value == 0:
        return "0"
    # ROUND_DOWN：宁可少说，也不能把 0.0000009 显示成 0.000001。
    shown = value.quantize(_PRECISION, rounding=ROUND_DOWN)
    if shown == 0:
        return f"<{_PRECISION}"
    # normalize 对整数值会得到 1E+2（100），必须用 format(..., "f") 落回普通写法。
    return format(shown.normalize(), "f")


async def _query_chain(adapter: ChainAdapter, chain_id: int, address: str, top_n: int) -> dict:
    balances: list[dict] = []
    errors: list[dict] = []

    # 代币列表读不到不影响原生币查询：注册表是增强，不是前置。
    try:
        async with session_scope() as db:
            tokens = await token_registry_repo.list_whitelisted(db, chain_id)
    except Exception:  # noqa: BLE001
        logger.warning("token registry unavailable chain_id=%s", chain_id, exc_info=True)
        tokens = []
        errors.append(
            {"chainId": chain_id, "scope": "token_registry", "message": "代币注册表不可用，仅返回原生币"}
        )

    try:
        native = await adapter.get_native_balance(chain_id, address)
        symbol, decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)
        balances.append(
            {
                "chainId": chain_id,
                "symbol": symbol,
                "decimals": decimals,
                "raw": str(native),
                "human": _human(native, decimals),
                "isNative": True,
            }
        )
    except RpcUnavailable:
        logger.warning("native balance unavailable chain_id=%s", chain_id, exc_info=True)
        errors.append({"chainId": chain_id, "scope": "native", "message": "链上余额服务暂时不可用"})

    # 代币余额彼此独立，串行等 N 个 RTT 只会让整轮超时。失败仍然逐条降级：
    # 一个代币查不到不能把同链其它代币一起抹掉。
    async def _balance(token) -> int | None:
        try:
            return await adapter.get_erc20_balance(chain_id, token.token_address, address)
        except RpcUnavailable:
            return None

    raws = await asyncio.gather(*(_balance(token) for token in tokens[:top_n]))

    for token, raw in zip(tokens[:top_n], raws):
        if raw is None:
            logger.warning("ERC20 balance unavailable chain_id=%s token=%s", chain_id,
                           token.symbol)
            errors.append(
                {"chainId": chain_id, "scope": token.symbol, "message": "链上余额服务暂时不可用"}
            )
            continue
        balances.append(
            {
                "chainId": chain_id,
                "symbol": token.symbol,
                "decimals": token.decimals,
                "raw": str(raw),
                "human": _human(raw, token.decimals),
                "isNative": False,
            }
        )

    return {"balances": balances, "errors": errors}


def build_get_asset_balance(adapter: ChainAdapter) -> StructuredTool:
    async def get_asset_balance(address: str, topN: int = 20) -> dict:
        if not is_evm_address(address):
            return {"error": "E101", "message": "地址格式不合法"}
        top_n = max(1, min(topN, 50))
        chains = adapter.available_chains()
        if not chains:
            return {"error": "E102", "message": "没有可用的白名单链 RPC"}

        results = await asyncio.gather(
            *[_query_chain(adapter, c, address, top_n) for c in chains],
            return_exceptions=True,
        )
        balances, errors = [], []
        for chain_id, result in zip(chains, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                logger.warning("chain balance query failed chain_id=%s", chain_id,
                               exc_info=(type(result), result, result.__traceback__))
                errors.append({"chainId": chain_id, "scope": "chain",
                               "message": "该链余额查询失败"})
            else:
                balances.extend(result["balances"])
                errors.extend(result["errors"])
        payload = {"address": address, "balances": balances, "errors": errors,
                   "status": "partial" if errors else "success"}
        if not balances:
            payload.update(error="E301", status="failed", message="所有链上余额查询均失败，请稍后重试")
        return payload

    return StructuredTool.from_function(
        coroutine=get_asset_balance,
        name="get_asset_balance",
        description=DESCRIPTION,
    )
