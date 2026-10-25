"""JSON-RPC 响应校验：畸形响应必须失败，不能被当成「余额 0」。

这一组的共同背景：`eth_getBalance` 返回 HTTP 200 但响应里没有 `result`，
旧代码 `body.get("result")` → None → `_to_int(None)` → 0，用户看到「余额 0」。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.config.settings import Settings
from web3_agent.web3.chain_adapter import ChainAdapter
from web3_agent.web3.clients import RpcClient

_MISSING = object()


def _rpc(handler) -> RpcClient:
    client = RpcClient("http://rpc.test", 1.0)
    old = client._client
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    asyncio.run(old.aclose())
    return client


def _adapter(handler) -> ChainAdapter:
    adapter = ChainAdapter(
        Settings(
            _env_file=None,
            allowed_chain_ids=[1],
            rpc_urls={1: "http://rpc.test"},
        )
    )
    for client in [*adapter._cached.values(), *adapter._direct.values()]:
        old = client._client
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        asyncio.run(old.aclose())
    return adapter


def _echo(result=_MISSING, *, error=None, body_id=None, raw_content=None):
    """按请求里的 id 回包，模拟真实 RPC。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if raw_content is not None:
            return httpx.Response(200, content=raw_content)
        payload = json.loads(request.content)
        body: dict = {
            "jsonrpc": "2.0",
            "id": payload["id"] if body_id is None else body_id,
        }
        if error is not None:
            body["error"] = error
        if result is not _MISSING:
            body["result"] = result
        return httpx.Response(200, json=body)

    return handler


# --- 信封层（clients.py）------------------------------------------------------


def test_missing_result_fails_instead_of_becoming_none():
    client = _rpc(_echo())
    with pytest.raises(RpcUnavailable, match="缺少 result"):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))


def test_null_result_is_a_valid_return():
    """回执查询未上链时 result 就是 null，这是合法值，不能当成协议错误。"""
    client = _rpc(_echo(None))
    assert asyncio.run(client.call("eth_getTransactionReceipt", ["0xdead"])) is None


def test_rpc_error_field_fails():
    client = _rpc(_echo(error={"code": -32000, "message": "boom"}))
    with pytest.raises(RpcUnavailable, match="boom"):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))


def test_mismatched_id_fails():
    client = _rpc(_echo("0x1", body_id=999))
    with pytest.raises(RpcUnavailable, match="id 与请求不一致"):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))


def test_response_id_missing_fails():
    client = _rpc(lambda request: httpx.Response(200, json={"jsonrpc": "2.0", "result": "0x1"}))
    with pytest.raises(RpcUnavailable, match="id 与请求不一致"):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))


def test_non_object_body_fails():
    client = _rpc(_echo(raw_content=b"[1,2,3]"))
    with pytest.raises(RpcUnavailable, match="不是 JSON-RPC 对象"):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))


def test_non_json_body_fails():
    client = _rpc(_echo(raw_content=b"<html>502 Bad Gateway</html>"))
    with pytest.raises(RpcUnavailable, match="不是合法 JSON"):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))


# --- 只读请求的一次重试 -------------------------------------------------------


def _flaky(success):
    """第一次返回 503，之后交给 success。"""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return success(request)

    return handler, calls


def test_transient_failure_is_retried_once():
    """节点偶尔 503，再发一次就拿到 —— 用户不该看到「余额查不到」。"""
    handler, calls = _flaky(_echo("0x1"))
    client = _rpc(handler)
    assert asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"])) == "0x1"
    assert len(calls) == 2


def test_retry_is_given_up_after_one_more_attempt():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(502)

    client = _rpc(handler)
    with pytest.raises(RpcUnavailable):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))
    assert len(calls) == 2


def test_client_error_is_not_retried():
    """4xx 是确定性失败：重试只是把等待时间翻倍。"""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400)

    client = _rpc(handler)
    with pytest.raises(RpcUnavailable):
        asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"]))
    assert len(calls) == 1


def test_id_is_compared_against_own_request(monkeypatch):
    """并发时 self._id 已被别的协程推进，仍要按本请求的 id 判断。"""
    client = RpcClient("http://rpc.test", 1.0)

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"jsonrpc": "2.0", "id": 1, "result": "0x1"}

    async def _post(url, json=None):  # noqa: A002 - 对应 httpx 的调用签名
        client._id += 1  # 另一个协程在本请求返回前又发了一次
        return _Resp()

    monkeypatch.setattr(client._client, "post", _post)
    assert asyncio.run(client.call("eth_getBalance", ["0xabc", "latest"])) == "0x1"


# --- 按方法的结果类型（chain_adapter.py）--------------------------------------


def test_missing_result_in_balance_query_raises_not_zero():
    adapter = _adapter(_echo())
    with pytest.raises(RpcUnavailable):
        asyncio.run(adapter.get_native_balance(1, "0xabc"))


def test_real_zero_balance_stays_zero():
    adapter = _adapter(_echo("0x0"))
    assert asyncio.run(adapter.get_native_balance(1, "0xabc")) == 0


def test_balance_with_non_string_result_raises():
    adapter = _adapter(_echo(12345))
    with pytest.raises(RpcUnavailable, match="十六进制数量"):
        asyncio.run(adapter.get_native_balance(1, "0xabc"))


def test_empty_eth_call_return_is_not_zero():
    """合约不存在或没有 balanceOf 时 eth_call 返回 0x，这不是余额 0。"""
    adapter = _adapter(_echo("0x"))
    with pytest.raises(RpcUnavailable, match="十六进制数量"):
        asyncio.run(adapter.get_erc20_balance(1, "0xtoken", "0xabc"))


def test_erc20_balance_reads_hex_word():
    adapter = _adapter(_echo("0x" + "0" * 61 + "3e8"))  # 1000
    assert asyncio.run(adapter.get_erc20_balance(1, "0xtoken", "0xabc")) == 1000


def test_receipt_allows_null_but_rejects_scalar():
    adapter = _adapter(_echo(None))
    assert asyncio.run(adapter.get_receipt(1, "0xdead")) is None

    partial = _adapter(_echo("0x1"))
    with pytest.raises(RpcUnavailable, match="回执对象"):
        asyncio.run(partial.get_receipt(1, "0xdead"))


def test_receipt_object_is_returned_as_is():
    receipt = {"transactionHash": "0xdead", "status": "0x1", "blockNumber": "0x10"}
    adapter = _adapter(_echo(receipt))
    assert asyncio.run(adapter.get_receipt(1, "0xdead")) == receipt


# --- 出卡片前的预演（chain_adapter.simulate）----------------------------------


def _sim_tx() -> dict:
    return {"from": "0x" + "1" * 40, "to": "0x" + "2" * 40, "value": str(10**17), "data": "0x"}


def test_simulate_passes_without_finding_a_reason():
    adapter = _adapter(_echo("0x"))
    assert asyncio.run(adapter.simulate(1, _sim_tx())) is None


def test_simulate_sends_value_as_hex_quantity():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["params"][0])
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x"})

    adapter = _adapter(handler)
    asyncio.run(adapter.simulate(1, _sim_tx()))

    assert seen[0]["value"] == hex(10**17)
    assert seen[0]["data"] == "0x"


def test_simulate_reports_contract_revert():
    adapter = _adapter(
        _echo(error={"code": 3, "message": "execution reverted: ERC20: blacklisted"})
    )
    assert asyncio.run(adapter.simulate(1, _sim_tx())) == "reverted"


def test_simulate_reports_gas_shortage_separately():
    adapter = _adapter(
        _echo(error={"code": -32000, "message": "insufficient funds for gas * price + value"})
    )
    assert asyncio.run(adapter.simulate(1, _sim_tx())) == "insufficient_funds"


def test_simulate_treats_unknown_errors_as_no_verdict():
    """节点抖动、方法不支持都不是「这笔交易会失败」的证据，不能据此拦下转账。"""
    adapter = _adapter(_echo(error={"code": -32000, "message": "header not found"}))
    assert asyncio.run(adapter.simulate(1, _sim_tx())) is None


def test_simulate_reads_latest_state_not_cache():
    """预演两次必须打两次 RPC：缓存里十分钟前的余额不能作为能不能转的依据。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": len(calls), "result": "0x"})

    adapter = _adapter(handler)
    asyncio.run(adapter.simulate(1, _sim_tx()))
    asyncio.run(adapter.simulate(1, _sim_tx()))

    assert len(calls) == 2
