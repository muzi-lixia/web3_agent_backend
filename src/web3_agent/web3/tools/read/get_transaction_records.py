"""T2 get_transaction_records：合并区块浏览器多源记录并聚合。

数据源（txlist / tokentx / txlistinternal）需要区块浏览器 API Key，
未配置时返回 E303，由 LLM 转述为"暂时查不到"，不伪造记录。

返回粒度是「交易」：records 的每一项是一笔交易，交易内部的资产变动放在 events 里。
同一哈希的多条来源记录是同一笔交易的不同事件（原生转账 / ERC20 转账 / 内部调用），
按事件摊平会让「最近 10 笔交易」里出现同一条 hash 十次，按交易聚合才既不丢事件
也不让条数失真。
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from langchain_core.tools import StructuredTool

from web3_agent.common.exceptions import BudgetExhausted
from web3_agent.config.settings import Settings
from web3_agent.web3.chain_adapter import is_evm_address
from web3_agent.common.metering import Resource, reserve
from web3_agent.web3.net import MAX_ATTEMPTS, is_transient

DESCRIPTION = "获取钱包地址链上最近 10 笔历史交易记录；chainId后端会话注入"

logger = logging.getLogger(__name__)

_SOURCES = ("txlist", "tokentx", "txlistinternal")

# 一次只取最近 10 条：数据源按 offset 从新到旧返回，多取的部分没有用户会看，
# 却要付出三倍的下游请求和上下文。分页留到有真实需求时再说。
_QUERY_SIZE = 10


async def _get(client: httpx.AsyncClient, base_url: str, params: dict) -> httpx.Response:
    """发起一次数据源查询，传输层失败与 5xx 各重试一次。

    区块浏览器不经过 RPC 客户端，单独记一类；扣减同样在请求之前，重试算两次。
    """
    last: httpx.HTTPError | None = None
    for _ in range(MAX_ATTEMPTS):
        if not reserve(Resource.EXPLORER):
            raise BudgetExhausted("本轮交易记录查询已达上限")
        try:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            last = exc
            if not is_transient(exc):
                raise
            continue
        return resp
    raise last


async def _fetch_source(
    client: httpx.AsyncClient, base_url: str, api_key: str, chain_id: int, address: str, source: str
) -> list[dict]:
    params = {
        "chainid": chain_id,
        "module": "account",
        "action": source,
        "address": address,
        "page": 1,
        "offset": _QUERY_SIZE,
        "sort": "desc",
        "apikey": api_key,
    }
    resp = await _get(client, base_url, params)
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError("invalid explorer response")
    if body.get("status") == "1" and isinstance(body.get("result"), list):
        return body["result"]
    message = str(body.get("message", "")).casefold()
    if (body.get("status") == "0" and body.get("result") == []
            and "no" in message and "found" in message):
        return []
    # HTTP 200 may still be an API-key/rate-limit/service error, not an empty wallet.
    raise ValueError("explorer rejected query or returned invalid response")


def _normalize(item: dict, chain_id: int, source: str) -> dict:
    """一条来源记录 —— 也就是一笔交易的其中一条资产变动。"""
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


def build_get_transaction_records(settings: Settings) -> StructuredTool:
    async def get_transaction_records(address: str) -> dict:
        if not is_evm_address(address):
            return {"error": "E101", "message": "地址格式不合法"}
        if not settings.explorer_api_key:
            return {"error": "E303", "message": "交易记录数据源未配置"}

        # 区块浏览器是独立 API，只需要 chainid 和 key。按 RPC 白名单取链会让
        # 没配 RPC 的链永远查不到交易记录，而「没配置」和「没有交易」长得一样。
        chains = sorted(settings.allowed_chain_ids)
        if not chains:
            return {"error": "E102", "status": "failed", "message": "没有配置可查询的链"}
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

        merged: dict[tuple[int, str], dict] = {}
        per_chain = len(_SOURCES)
        errors, succeeded = [], 0
        for idx, batch in enumerate(batches):
            chain_id = chains[idx // per_chain]
            source = _SOURCES[idx % per_chain]
            if isinstance(batch, asyncio.CancelledError):
                raise batch
            if isinstance(batch, BudgetExhausted):
                # 预算是本轮的限制，不是数据源故障，别按异常打整段 traceback。
                errors.append({"chainId": chain_id, "scope": source,
                               "message": "本轮交易记录查询已达上限"})
                continue
            if isinstance(batch, BaseException) or not isinstance(batch, list):
                if isinstance(batch, BaseException):
                    logger.warning("transaction source failed chain_id=%s source=%s", chain_id, source,
                                   exc_info=(type(batch), batch, batch.__traceback__))
                errors.append({"chainId": chain_id, "scope": source,
                               "message": "该交易记录数据源暂时不可用"})
                continue
            valid_events, dropped = [], 0
            for item in batch:
                try:
                    event = _normalize(item, chain_id, source)
                    if not isinstance(event["hash"], str) or not event["hash"].strip():
                        raise ValueError("missing transaction hash")
                    valid_events.append(event)
                except (ValueError, TypeError, AttributeError, OverflowError):
                    dropped += 1
            if dropped:
                logger.warning("transaction records skipped chain_id=%s source=%s count=%s",
                               chain_id, source, dropped)
                errors.append({"chainId": chain_id, "scope": source, "dropped_records": dropped,
                               "message": f"跳过 {dropped} 条无法解析的交易记录"})
            # Empty successful batches are meaningful; a wholly corrupt batch isn't.
            if batch and not valid_events:
                continue
            succeeded += 1
            # 一笔交易可以包含多条资产变动：swap 的出账与入账、批量转账、内部调用。
            # 按 (链, 哈希) 聚合成一笔交易，事件全部保留 —— 只按哈希 setdefault 会留下
            # 第一条、丢掉后面的事件，金额和代币残缺却仍然显示查询成功。
            # 跨链同哈希是不同链上的两笔交易，键必须带 chainId。
            for event in valid_events:
                key = (event["chainId"], event["hash"].lower())
                group = merged.get(key)
                if group is None:
                    group = merged[key] = {
                        "chainId": event["chainId"],
                        "hash": event["hash"],
                        "blockNumber": event["blockNumber"],
                        "timeStamp": event["timeStamp"],
                        "from": event["from"],
                        "to": event["to"],
                        "events": [],
                    }
                elif event["source"] == "txlist":
                    # 交易级收发方取原生转账那条；纯代币交易没有 txlist，沿用第一条事件。
                    group["from"], group["to"] = event["from"], event["to"]
                group["events"].append({
                    "source": event["source"],
                    "from": event["from"],
                    "to": event["to"],
                    "value": event["value"],
                    "decimals": event["decimals"],
                    "tokenSymbol": event["tokenSymbol"],
                    "contractAddress": event["contractAddress"],
                })
        if not succeeded:
            return {"error": "E302", "status": "failed", "address": address,
                    "records": [], "errors": errors, "message": "所有交易记录数据源均查询失败，请稍后重试"}

        records = sorted(merged.values(), key=lambda r: r["timeStamp"], reverse=True)
        return {"address": address, "records": records[:_QUERY_SIZE], "errors": errors,
                "status": "partial" if errors else ("success" if records else "empty")}

    return StructuredTool.from_function(
        coroutine=get_transaction_records,
        name="get_transaction_records",
        description=DESCRIPTION,
    )
