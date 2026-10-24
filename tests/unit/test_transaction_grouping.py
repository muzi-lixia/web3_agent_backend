"""一笔交易里的多条资产变动必须全部保留。

原来的去重是 `merged.setdefault(hash, record)`：同一笔交易在 txlist / tokentx /
txlistinternal 里各有一条记录，最后只剩第一条。swap 只显示原生转账、批量空投只剩
一条代币转账，却仍然报「查询成功」。
"""

from __future__ import annotations

import asyncio
import importlib

from web3_agent.agent.render import render_transactions
from web3_agent.config.settings import Settings

transactions = importlib.import_module("web3_agent.web3.tools.read.get_transaction_records")
ADDRESS = "0x" + "1" * 40
OTHER = "0x" + "2" * 40
HASH = "0x" + "a" * 64


def _row(**overrides) -> dict:
    """区块浏览器的一条原始记录。"""
    row = {
        "hash": HASH,
        "blockNumber": "21000000",
        "timeStamp": "1700000000",
        "from": ADDRESS,
        "to": OTHER,
        "value": "0",
        "tokenDecimal": "18",
    }
    row.update(overrides)
    return row


def _tool(chains: tuple[int, ...] = (1,)):
    return transactions.build_get_transaction_records(
        Settings(_env_file=None, explorer_api_key="test", allowed_chain_ids=list(chains))
    )


def test_events_of_one_transaction_are_all_kept(monkeypatch):
    async def run():
        async def fetch(client, base_url, api_key, chain_id, address, source):
            if source == "txlist":
                return [_row(value="100000000000000000")]
            if source == "tokentx":
                return [
                    _row(value="1000000", tokenSymbol="USDC", tokenDecimal="6",
                         contractAddress="0x" + "c" * 40),
                    _row(value="20000000000000000", tokenSymbol="WETH",
                         contractAddress="0x" + "d" * 40),
                ]
            return [_row(value="5000")]

        monkeypatch.setattr(transactions, "_fetch_source", fetch)
        result = await _tool().ainvoke({"address": ADDRESS})

        assert len(result["records"]) == 1
        events = result["records"][0]["events"]
        assert [e["source"] for e in events] == ["txlist", "tokentx", "tokentx", "txlistinternal"]

        text = render_transactions(result)
        assert "0.1 ETH" in text and "1 USDC" in text and "0.02 WETH" in text

    asyncio.run(run())


def test_same_hash_on_two_chains_stays_two_transactions(monkeypatch):
    async def run():
        async def fetch(client, base_url, api_key, chain_id, address, source):
            return [_row(value="1")] if source == "txlist" else []

        monkeypatch.setattr(transactions, "_fetch_source", fetch)
        result = await _tool((1, 56)).ainvoke({"address": ADDRESS})

        assert len(result["records"]) == 2
        assert {record["chainId"] for record in result["records"]} == {1, 56}

    asyncio.run(run())


def test_records_stop_at_ten_transactions(monkeypatch):
    async def run():
        async def fetch(client, base_url, api_key, chain_id, address, source):
            if source != "tokentx":
                return []
            # 最新那笔有两项资产变动，后面再跟 11 笔：条数按交易算，多的截掉。
            rows = [
                _row(hash="0x" + "b" * 64, timeStamp="2000000000", value="1", tokenSymbol="A"),
                _row(hash="0x" + "b" * 64, timeStamp="2000000000", value="2", tokenSymbol="B"),
            ]
            rows += [
                _row(hash=f"0x{i:064x}", timeStamp=str(1900000000 - i), value="3", tokenSymbol="C")
                for i in range(11)
            ]
            return rows

        monkeypatch.setattr(transactions, "_fetch_source", fetch)
        result = await _tool().ainvoke({"address": ADDRESS})

        assert len(result["records"]) == 10
        assert result["records"][0]["hash"] == "0x" + "b" * 64
        assert len(result["records"][0]["events"]) == 2

    asyncio.run(run())
