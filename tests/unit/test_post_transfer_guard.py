"""转账构造前的两道闸：单笔等值限额与链上预演。

两道闸都必须在出卡片之前生效 —— 卡片一旦给出去，用户就会签名、广播、烧 gas。
"""

from __future__ import annotations

import asyncio

from web3_agent.web3.tools.write.transfer import build_post_transfer

# _CHAIN：测试用网络，原生币为 BERA。
_CHAIN = 80094
# _FROM：付款地址。
_FROM = "0x" + "1" * 40
# _TO：收款地址。
_TO = "0x" + "2" * 40


class _Adapter:
    def __init__(self, *, balance: int = 10**18, rejection: str | None = None) -> None:
        self.balance = balance
        self.rejection = rejection
        self.simulated: list[dict] = []

    def available_chains(self) -> list[int]:
        return [_CHAIN]

    async def get_native_balance(self, chain_id: int, address: str) -> int:
        return self.balance

    async def simulate(self, chain_id: int, tx: dict) -> str | None:
        self.simulated.append(tx)
        return self.rejection


def _run(adapter: _Adapter, *, amount: str = "0.1", **kwargs) -> dict:
    if "max_amount_usd" not in kwargs:
        kwargs["max_amount_usd"] = 0
    if "usd_prices" not in kwargs:
        kwargs["usd_prices"] = {}
    tool = build_post_transfer(adapter, **kwargs)
    return asyncio.run(
        tool.ainvoke(
            {
                "from_address": _FROM,
                "to_address": _TO,
                "token_symbol": "BERA",
                "human_amount": amount,
                "chain_id": _CHAIN,
            }
        )
    )


def test_native_transfer_is_ready_and_simulated():
    adapter = _Adapter()

    data = _run(adapter)

    assert data["status"] == "ready"
    assert data["tx"]["to"] == _TO
    assert adapter.simulated == [
        {"from": _FROM, "to": _TO, "value": str(10**17), "data": "0x"}
    ]


def test_amount_over_limit_is_rejected():
    data = _run(_Adapter(), amount="10", max_amount_usd=1, usd_prices={"BERA": 3})

    assert data["status"] == "failed"
    assert data["error"] == "E502"
    assert "30.00" in data["message"]


def test_amount_within_limit_passes():
    data = _run(_Adapter(), amount="0.1", max_amount_usd=1, usd_prices={"BERA": 3})

    assert data["status"] == "ready"


def test_asset_without_price_is_rejected():
    """折算不出来就不放行：没有单价不等于没有超额。"""
    data = _run(_Adapter(), max_amount_usd=5000, usd_prices={})

    assert data["status"] == "failed"
    assert data["error"] == "E502"
    assert "BERA" in data["message"]


def test_limit_zero_disables_the_gate():
    data = _run(_Adapter(), max_amount_usd=0, usd_prices={})

    assert data["status"] == "ready"


def test_simulated_revert_blocks_the_card():
    data = _run(_Adapter(rejection="reverted"))

    assert data["status"] == "failed"
    assert data["error"] == "E401"
    assert "拒绝" in data["message"]


def test_simulated_gas_shortage_says_so():
    data = _run(_Adapter(rejection="insufficient_funds"))

    assert data["status"] == "failed"
    assert data["error"] == "E401"
    assert "手续费" in data["message"]


def test_insufficient_balance_still_rejected_before_simulation():
    adapter = _Adapter(balance=10**17)

    data = _run(adapter, amount="0.5")

    assert data["error"] == "E405"
    assert adapter.simulated == []
