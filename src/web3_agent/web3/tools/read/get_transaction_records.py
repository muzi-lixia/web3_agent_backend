"""T2 get_transaction_records：合并区块浏览器多源记录并去重。

数据源（txlist / tokentx / txlistinternal）需要区块浏览器 API Key，
未配置时返回 E303，由 LLM 转述为"暂时查不到"，不伪造记录。
"""

from __future__ import annotations

import asyncio

import httpx
from langchain_core.tools import StructuredTool

from web3_agent.config.settings import Settings
from web3_agent.web3.chain_adapter import is_evm_address

DESCRIPTION = "获取钱包地址链上历史交易记录，limit最大50；chainId后端会话注入"

_SOURCES = ("txlist", "tokentx", "txlistinternal")


async def _fetch_source(
    client: httpx.AsyncClient, base_url: str, api_key: str, chain_id: int, address: str, source: str
) -> list[dict]:
    params = {
        "chainid": chain_id,
        "module": "account",
        "action": source,
        "address": address,
        "page": 1,
        "offset": 50,
        "sort": "desc",
        "apikey": api_key,
    }
    resp = await client.get(base_url, params=params)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "1" or not isinstance(body.get("result"), list):
        return []
    return body["result"]


def _normalize(item: dict, chain_id: int, source: str) -> dict:
    return {
        "chainId": chain_id,
        "hash": item.get("hash"),
        "blockNumber": int(item.get("blockNumber", 0)),
        "timeStamp": int(item.get("timeStamp", 0)),
        "from": item.get("from"),
        "to": item.get("to"),
        "value": item.get("value"),
        "decimals": int(item.get("tokenDecimal") or 18),
        "tokenSymbol": item.get("tokenSymbol"),
        "contractAddress": item.get("contractAddress"),
        "source": source,
    }


def build_get_transaction_records(adapter, settings: Settings) -> StructuredTool:
    async def get_transaction_records(address: str, limit: int = 10, cursor: str = "") -> dict:
        if not is_evm_address(address):
            return {"error": "E101", "message": "地址格式不合法"}
        if not settings.explorer_api_key:
            return {"error": "E303", "message": "交易记录数据源未配置"}

        limit = max(1, min(limit, 50))
        chains = adapter.available_chains()
        async with httpx.AsyncClient(timeout=settings.rpc_timeout_seconds) as client:
            batches = await asyncio.gather(
                *[
                    _fetch_source(
                        client,
                        settings.explorer_base_url,
                        settings.explorer_api_key,
                        chain_id,
                        address,
                        source,
                    )
                    for chain_id in chains
                    for source in _SOURCES
                ],
                return_exceptions=True,
            )

        merged: dict[str, dict] = {}
        per_chain = len(_SOURCES)
        for idx, batch in enumerate(batches):
            if isinstance(batch, BaseException):
                continue
            chain_id = chains[idx // per_chain]
            source = _SOURCES[idx % per_chain]
            for item in batch:
                record = _normalize(item, chain_id, source)
                if record["hash"]:
                    merged.setdefault(record["hash"], record)

        records = sorted(merged.values(), key=lambda r: r["timeStamp"], reverse=True)
        return {"address": address, "records": records[:limit]}

    return StructuredTool.from_function(
        coroutine=get_transaction_records,
        name="get_transaction_records",
        description=DESCRIPTION,
    )
