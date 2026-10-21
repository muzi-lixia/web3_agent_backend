"""T1 get_asset_balance：原生币 + 白名单 ERC20 余额，多链并发查询。"""

from __future__ import annotations

import asyncio
from decimal import ROUND_DOWN, Decimal

from langchain_core.tools import StructuredTool

from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import token_registry_repo
from web3_agent.web3.chain_adapter import ChainAdapter, is_evm_address

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
    except RpcUnavailable as exc:
        errors.append({"chainId": chain_id, "scope": "native", "message": str(exc)})

    for token in tokens[:top_n]:
        try:
            raw = await adapter.get_erc20_balance(chain_id, token.token_address, address)
        except RpcUnavailable as exc:
            errors.append(
                {"chainId": chain_id, "scope": token.symbol, "message": str(exc)}
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
            *[_query_chain(adapter, c, address, top_n) for c in chains]
        )
        return {
            "address": address,
            "balances": [b for r in results for b in r["balances"]],
            "errors": [e for r in results for e in r["errors"]],
        }

    return StructuredTool.from_function(
        coroutine=get_asset_balance,
        name="get_asset_balance",
        description=DESCRIPTION,
    )
