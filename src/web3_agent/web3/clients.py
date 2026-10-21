"""JSON-RPC 客户端。一个 ChainAdapter 持有一个实例，不共享连接池配置。"""

from __future__ import annotations

import httpx

from web3_agent.common.exceptions import RpcUnavailable


class RpcClient:
    def __init__(self, url: str, timeout: float) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=timeout)
        self._id = 0

    async def call(self, method: str, params: list) -> object:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        try:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as exc:
            raise RpcUnavailable(f"{method} -> {exc}") from exc

        if "error" in body:
            raise RpcUnavailable(f"{method} -> {body['error']}")
        return body.get("result")

    async def aclose(self) -> None:
        await self._client.aclose()
