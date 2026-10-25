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

# _BALANCE_OF_SELECTOR：ERC20 balanceOf(address) 方法的 4 字节选择器。
_BALANCE_OF_SELECTOR = "0x70a08231"
# 读缓存上限。键是 chain:method:params，参数一变就是新键，不设界会随运行时间无限涨。
# _READ_CACHE_MAX_ENTRIES：进程内链读取缓存容量上限。
_READ_CACHE_MAX_ENTRIES = 512


def encode_balance_of(holder: str) -> str:
    """将 ERC20 balanceOf 方法选择器与 32 字节左补零地址拼成调用数据。

    Args:
        holder: ERC20 balanceOf 查询的持有者地址。
    """
    return _BALANCE_OF_SELECTOR + holder.lower().removeprefix("0x").rjust(64, "0")


def _hex_data(method: str, raw: Any) -> str:
    """eth_call 的返回值：0x 开头的十六进制串，允许是空数据。

    Args:
        method: JSON-RPC 方法名，如 eth_getBalance。
        raw: 底层未归一化的值或模型候选。
    """
    if not isinstance(raw, str) or not raw.startswith("0x"):
        raise RpcUnavailable(f"{method} -> 结果不是十六进制数据: {raw!r}")
    return raw


def _hex_int(method: str, raw: Any) -> int:
    """链上数量：0x 开头的非空十六进制串。

    不接受 None，也不接受空串 —— 把缺失折成 0，等于把上游协议错误当作
    「余额为零」交给用户，这比直接报错更难被发现。

    Args:
        method: JSON-RPC 方法名，如 eth_getBalance。
        raw: 底层未归一化的值或模型候选。
    """
    if not isinstance(raw, str) or not raw.startswith("0x") or len(raw) == 2:
        raise RpcUnavailable(f"{method} -> 结果不是十六进制数量: {raw!r}")
    try:
        return int(raw, 16)
    except ValueError as exc:
        raise RpcUnavailable(f"{method} -> 结果不是十六进制数量: {raw!r}") from exc


# 数据职责：网络级只读访问入口，分离短缓存读取与绕过缓存的直连调用。
class ChainAdapter:
    def __init__(self, settings: Settings) -> None:
        # _ttl：链上读取结果的缓存有效秒数。
        """按已配置的白名单网络分别建立缓存读取和直连 RPC 客户端，并初始化有界缓存。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self._ttl = settings.read_cache_ttl_seconds
        # _cache：链上查询的 LRU 结果缓存，值保存单调时钟写入时间和结果。
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        # _cached：网络编号到只读查询 RPC 客户端的映射。
        self._cached: dict[int, RpcClient] = {}
        # _direct：网络编号到绕过缓存的 RPC 客户端的映射。
        self._direct: dict[int, RpcClient] = {}
        for chain_id in settings.allowed_chain_ids:
            url = settings.rpc_urls.get(chain_id)
            if not url:
                continue
            self._cached[chain_id] = RpcClient(url, settings.rpc_timeout_seconds)
            self._direct[chain_id] = RpcClient(url, settings.rpc_timeout_seconds)

    def available_chains(self) -> list[int]:
        """返回已实际配置 RPC 客户端的白名单网络编号。"""
        return sorted(self._cached)

    def _require(self, chain_id: int, allow_cached: bool) -> RpcClient:
        """选择缓存或直连客户端；该网络未配置时抛链服务不可用。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            allow_cached: 是否选择允许缓存的查询客户端；False 时使用独立直连客户端。

        Raises:
            RpcUnavailable: 目标网络未配置 RPC 客户端。
        """
        pool = self._cached if allow_cached else self._direct
        client = pool.get(chain_id)
        if client is None:
            raise RpcUnavailable(f"chain {chain_id} 未配置 RPC")
        return client

    async def _read(self, chain_id: int, method: str, params: list) -> Any:
        """按网络、方法和参数读取短 TTL 缓存，未命中时请求 RPC 并限制缓存容量。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            method: JSON-RPC 方法名，如 eth_getBalance。
            params: JSON-RPC 方法参数列表。
        """
        key = f"{chain_id}:{method}:{params}"
        # 缓存时效使用单调时钟，仅决定是否复用读结果，不参与链上时间判断。
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
        """调用 eth_getBalance 查询最新原生币余额，返回最小单位整数。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            address: 待校验、登录或查询的钱包地址。
        """
        raw = await self._read(chain_id, "eth_getBalance", [address, "latest"])
        return _hex_int("eth_getBalance", raw)

    async def get_erc20_balance(self, chain_id: int, token: str, holder: str) -> int:
        """编码 balanceOf 并通过 eth_call 查询 ERC20 原始整数余额。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            token: ERC20 合约地址。
            holder: ERC20 balanceOf 查询的持有者地址。
        """
        raw = await self._read(
            chain_id,
            "eth_call",
            [{"to": token, "data": encode_balance_of(holder)}, "latest"],
        )
        # 空数据说明这次调用没有返回值（合约不存在或该方法不存在），
        # 与「余额为 0」是两件事。
        return _hex_int("eth_call", _hex_data("eth_call", raw))

    async def get_block_number(self, chain_id: int) -> int:
        """查询网络最新区块高度并转换为整数。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
        """
        raw = await self._read(chain_id, "eth_blockNumber", [])
        return _hex_int("eth_blockNumber", raw)

    async def get_receipt(self, chain_id: int, tx_hash: str) -> dict | None:
        """写路径专用：绕过缓存。

        未上链的交易查回执得到的是 result=null，这是合法结果，由调用方按「待确认」处理；
        缺少 result 字段则已经在客户端层失败，到这里不会出现。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            tx_hash: 待查询的交易哈希。
        """
        raw = await self._require(chain_id, allow_cached=False).call(
            "eth_getTransactionReceipt", [tx_hash]
        )
        if raw is not None and not isinstance(raw, dict):
            raise RpcUnavailable(
                f"eth_getTransactionReceipt -> 结果不是回执对象: {type(raw).__name__}"
            )
        return raw

    async def get_transaction(self, chain_id: int, tx_hash: str) -> dict | None:
        """写路径专用：绕过缓存。

        回执里没有转账金额与调用数据，核对「这笔哈希是不是本任务的交易」只能看交易本身。
        未上链时返回 None，与回执同口径。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            tx_hash: 待查询的交易哈希。
        """
        raw = await self._require(chain_id, allow_cached=False).call(
            "eth_getTransactionByHash", [tx_hash]
        )
        if raw is not None and not isinstance(raw, dict):
            raise RpcUnavailable(
                f"eth_getTransactionByHash -> 结果不是交易对象: {type(raw).__name__}"
            )
        return raw

    async def aclose(self) -> None:
        # 两组客户端键完全相同，但各自持有独立的连接池。合并成 dict 再遍历，
        # 后者会把前者按同键覆盖掉 —— 被覆盖的那组永远不会关闭。
        """关闭当前对象持有的异步 HTTP 连接池并释放资源。"""
        for client in self._cached.values():
            await client.aclose()
        for client in self._direct.values():
            await client.aclose()
        self._cached.clear()
        self._direct.clear()
