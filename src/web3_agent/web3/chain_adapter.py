"""ChainAdapter：只读链访问的唯一入口。

两套客户端严格分开：
- cached_client：只读查询，允许短 TTL 缓存；
- direct_client：写路径校验用，blockTag 永远是 latest，完全绕过缓存。

V1a 只用到 cached_client。
"""

from __future__ import annotations

import time
from typing import Any

from web3_agent.common.evm_address import is_evm_address
from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.config.settings import Settings
from web3_agent.web3.clients import RpcClient

_BALANCE_OF_SELECTOR = "0x70a08231"


def encode_balance_of(holder: str) -> str:
    return _BALANCE_OF_SELECTOR + holder.lower().removeprefix("0x").rjust(64, "0")


def _to_int(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    return int(str(raw), 16)


class ChainAdapter:
    def __init__(self, settings: Settings) -> None:
        self._ttl = settings.read_cache_ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cached: dict[int, RpcClient] = {}
        self._direct: dict[int, RpcClient] = {}
        for chain_id in settings.allowed_chain_ids:
            url = settings.rpc_urls.get(chain_id)
            if not url:
                continue
            self._cached[chain_id] = RpcClient(url, settings.rpc_timeout_seconds)
            self._direct[chain_id] = RpcClient(url, settings.rpc_timeout_seconds)

    def available_chains(self) -> list[int]:
        return sorted(self._cached)

    def _require(self, chain_id: int, allow_cached: bool) -> RpcClient:
        pool = self._cached if allow_cached else self._direct
        client = pool.get(chain_id)
        if client is None:
            raise RpcUnavailable(f"chain {chain_id} 未配置 RPC")
        return client

    async def _read(self, chain_id: int, method: str, params: list) -> Any:
        key = f"{chain_id}:{method}:{params}"
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self._ttl:
            return hit[1]
        result = await self._require(chain_id, allow_cached=True).call(method, params)
        self._cache[key] = (now, result)
        return result

    async def get_native_balance(self, chain_id: int, address: str) -> int:
        return _to_int(await self._read(chain_id, "eth_getBalance", [address, "latest"]))

    async def get_erc20_balance(self, chain_id: int, token: str, holder: str) -> int:
        raw = await self._read(
            chain_id,
            "eth_call",
            [{"to": token, "data": encode_balance_of(holder)}, "latest"],
        )
        return _to_int(raw)

    async def get_block_number(self, chain_id: int) -> int:
        return _to_int(await self._read(chain_id, "eth_blockNumber", []))

    async def get_receipt(self, chain_id: int, tx_hash: str) -> dict | None:
        """写路径专用：绕过缓存。"""
        return await self._require(chain_id, allow_cached=False).call(
            "eth_getTransactionReceipt", [tx_hash]
        )

    async def aclose(self) -> None:
        for client in {**self._cached, **self._direct}.values():
            await client.aclose()
        self._cached.clear()
        self._direct.clear()
