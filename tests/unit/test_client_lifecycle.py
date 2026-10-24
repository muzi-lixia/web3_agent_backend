"""客户端生命周期：两组连接池都要关，读缓存要有界。"""
import asyncio
from types import SimpleNamespace

import web3_agent.web3.chain_adapter as chain_adapter_module
from web3_agent.web3.chain_adapter import ChainAdapter


class FakeClient:
    def __init__(self, url: str, timeout: float) -> None:
        self.url, self.closed = url, False

    async def call(self, method: str, params: list):
        return "0x1"

    async def aclose(self) -> None:
        self.closed = True


def _settings() -> SimpleNamespace:
    return SimpleNamespace(allowed_chain_ids=[1], rpc_urls={1: "http://rpc"},
                           rpc_timeout_seconds=1.0, read_cache_ttl_seconds=10.0)


def _adapter(monkeypatch) -> ChainAdapter:
    monkeypatch.setattr(chain_adapter_module, "RpcClient", FakeClient)
    return ChainAdapter(_settings())


def test_close_releases_both_client_pools(monkeypatch):
    """两组客户端键相同，合并成 dict 遍历会漏关一组。"""
    adapter = _adapter(monkeypatch)
    cached, direct = adapter._cached[1], adapter._direct[1]
    assert cached is not direct
    asyncio.run(adapter.aclose())
    assert cached.closed and direct.closed


def test_read_cache_is_bounded(monkeypatch):
    adapter = _adapter(monkeypatch)

    async def fill():
        for index in range(600):
            await adapter._read(1, "eth_call", [index])

    asyncio.run(fill())
    assert len(adapter._cache) <= chain_adapter_module._READ_CACHE_MAX_ENTRIES


def test_expired_entry_is_evicted_on_read(monkeypatch):
    adapter = _adapter(monkeypatch)

    async def run():
        await adapter._read(1, "eth_call", [0])
        key = next(iter(adapter._cache))
        adapter._cache[key] = (0.0, "stale")
        assert await adapter._read(1, "eth_call", [0]) == "0x1"
        assert adapter._cache[key][1] == "0x1"

    asyncio.run(run())
