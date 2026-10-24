"""JSON-RPC 客户端。一个 ChainAdapter 持有一个实例，不共享连接池配置。

这里只校验信封：必须是 JSON-RPC 对象、id 对得上、error 与 result 有且只有一个。
「result 为 null」和「没有 result」是两件事：前者是合法返回值（未上链的交易查回执
就是这样），后者是上游协议错误，必须失败。把它们合并成同一个 None，余额查询就会
把坏响应报成 0。
"""

from __future__ import annotations

import httpx

from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.common.metering import Resource, reserve
from web3_agent.web3.net import MAX_ATTEMPTS, is_transient


# 数据职责：持有独立 HTTP 连接池的 JSON-RPC 客户端，校验响应并计量重试。
class RpcClient:
    def __init__(self, url: str, timeout: float) -> None:
        """创建独立 HTTP 连接池并初始化请求编号。

        Args:
            url: 目标 JSON-RPC 地址。
            timeout: HTTP 请求超时，单位为秒。
        """
        self._url = url
        # _client：当前对象持有的异步 HTTP 客户端及连接池。
        self._client = httpx.AsyncClient(timeout=timeout)
        # _id：当前 RPC 客户端已分配的递增请求编号。
        self._id = 0

    async def _send(self, method: str, params: list) -> tuple[int, httpx.Response]:
        """发一次请求，传输层失败重试一次。返回 (请求 id, 响应)。

        Args:
            method: JSON-RPC 方法名，如 eth_getBalance。
            params: JSON-RPC 方法参数列表。
        """
        last: RpcUnavailable | None = None
        for _ in range(MAX_ATTEMPTS):
            # 预算扣减在发请求之前：并发工具同时打到这条链上时，事后记账一定会超。
            # 用完不是链上故障，但工具层只需要「这次没拿到数据」，按 RpcUnavailable 降级。
            # 重试同样是一次真实请求，占一次预算。
            if not reserve(Resource.RPC):
                raise RpcUnavailable(f"{method} -> 本轮 RPC 调用预算已用完")
            self._id += 1
            request_id = self._id
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            try:
                resp = await self._client.post(self._url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                last = RpcUnavailable(f"{method} -> {exc}")
                if not is_transient(exc):
                    raise last from exc
                continue
            return request_id, resp
        raise last

    async def call(self, method: str, params: list) -> object:
        """发送 JSON-RPC 方法调用并校验响应结构、请求 ID 和 result 字段。

        Args:
            method: JSON-RPC 方法名，如 eth_getBalance。
            params: JSON-RPC 方法参数列表。

        Raises:
            RpcUnavailable: 下游请求失败、预算耗尽或响应协议不合法。
        """
        request_id, resp = await self._send(method, params)
        try:
            # HTTP 成功不代表协议成功，还需要逐项检查 JSON-RPC 或浏览器业务字段。
            body = resp.json()
        except ValueError as exc:  # 网关返回 HTML 错误页时会走到这里
            raise RpcUnavailable(f"{method} -> 响应不是合法 JSON") from exc

        if not isinstance(body, dict):
            raise RpcUnavailable(f"{method} -> 响应不是 JSON-RPC 对象")
        if body.get("error") is not None:
            raise RpcUnavailable(f"{method} -> {body['error']}")
        # 比对用 _send 返回的那个 id：同一个客户端会被并发调用，self._id 早已被
        # 别的协程推进，拿它比会把正常响应误判成 id 不一致。
        if str(body.get("id")) != str(request_id):
            raise RpcUnavailable(f"{method} -> 响应 id 与请求不一致")
        # 必须区别“没有 result 字段”的坏响应与“result 为 null”的合法待确认回执。
        if "result" not in body:
            raise RpcUnavailable(f"{method} -> 响应缺少 result 字段")
        return body["result"]

    async def aclose(self) -> None:
        """关闭当前对象持有的异步 HTTP 连接池并释放资源。"""
        await self._client.aclose()
