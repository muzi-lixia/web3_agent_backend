"""ChainAdapter：只读链访问的唯一入口。

两套客户端严格分开：
- cached_client：只读查询，允许短 TTL 缓存；
- direct_client：写路径校验用，blockTag 永远是 latest，完全绕过缓存。

V1a 只用到 cached_client。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from web3_agent.common.evm_address import is_evm_address
from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.config.settings import Settings
from web3_agent.web3.clients import RpcClient

_BALANCE_OF_SELECTOR = "0x70a08231"
# 读缓存上限。键是 chain:method:params，参数一变就是新键，不设界会随运行时间无限涨。
_READ_CACHE_MAX_ENTRIES = 512


def encode_balance_of(holder: str) -> str:
    return _BALANCE_OF_SELECTOR + holder.lower().removeprefix("0x").rjust(64, "0")


def _hex_data(method: str, raw: Any) -> str:
    """eth_call 的返回值：0x 开头的十六进制串，允许是空数据。"""
    if not isinstance(raw, str) or not raw.startswith("0x"):
        raise RpcUnavailable(f"{method} -> 结果不是十六进制数据: {raw!r}")
    return raw


def _hex_int(method: str, raw: Any) -> int:
    """链上数量：0x 开头的非空十六进制串。

    不接受 None，也不接受空串 —— 把缺失折成 0，等于把上游协议错误当作
    「余额为零」交给用户，这比直接报错更难被发现。
    """
    if not isinstance(raw, str) or not raw.startswith("0x") or len(raw) == 2:
        raise RpcUnavailable(f"{method} -> 结果不是十六进制数量: {raw!r}")
    try:
        return int(raw, 16)
    except ValueError as exc:
        raise RpcUnavailable(f"{method} -> 结果不是十六进制数量: {raw!r}") from exc


class ChainAdapter:
    def __init__(self, settings: Settings) -> None:
        self._ttl = settings.read_cache_ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
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
        if hit is not None:
            if now - hit[0] < self._ttl:
                self._cache.move_to_end(key)
                return hit[1]
            # 过期就地删掉：留着只会占内存，下次写入也是同一个键。
            self._cache.pop(key, None)
        result = await self._require(chain_id, allow_cached=True).call(method, params)
        self._cache[key] = (now, result)
        self._cache.move_to_end(key)
        while len(self._cache) > _READ_CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)
        return result

    async def get_native_balance(self, chain_id: int, address: str) -> int:
        raw = await self._read(chain_id, "eth_getBalance", [address, "latest"])
        return _hex_int("eth_getBalance", raw)

    async def get_erc20_balance(self, chain_id: int, token: str, holder: str) -> int:
        raw = await self._read(
            chain_id,
            "eth_call",
            [{"to": token, "data": encode_balance_of(holder)}, "latest"],
        )
        # 空数据说明这次调用没有返回值（合约不存在或该方法不存在），
        # 与「余额为 0」是两件事。
        return _hex_int("eth_call", _hex_data("eth_call", raw))

    async def get_block_number(self, chain_id: int) -> int:
        raw = await self._read(chain_id, "eth_blockNumber", [])
        return _hex_int("eth_blockNumber", raw)

    async def get_receipt(self, chain_id: int, tx_hash: str) -> dict | None:
        """写路径专用：绕过缓存。

        未上链的交易查回执得到的是 result=null，这是合法结果，由调用方按「待确认」处理；
        缺少 result 字段则已经在客户端层失败，到这里不会出现。
        """
        raw = await self._require(chain_id, allow_cached=False).call(
            "eth_getTransactionReceipt", [tx_hash]
        )
        if raw is not None and not isinstance(raw, dict):
            raise RpcUnavailable(
                f"eth_getTransactionReceipt -> 结果不是回执对象: {type(raw).__name__}"
            )
        return raw

    async def aclose(self) -> None:
        # 两组客户端键完全相同，但各自持有独立的连接池。合并成 dict 再遍历，
        # 后者会把前者按同键覆盖掉 —— 被覆盖的那组永远不会关闭。
        for client in self._cached.values():
            await client.aclose()
        for client in self._direct.values():
            await client.aclose()
        self._cached.clear()
        self._direct.clear()
