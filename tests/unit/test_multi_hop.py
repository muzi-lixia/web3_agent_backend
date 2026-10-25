from decimal import Decimal

from web3_agent.agent.multi_hop import Verdict, evaluate, format_amount, human_amount
from web3_agent.common.types import Hop

_CHAIN = 80094


def _balance(*items: dict) -> dict:
    return {"balances": list(items), "errors": []}


def _bera(amount: str, decimals: int = 18, chain_id: int = _CHAIN) -> dict:
    return {
        "chainId": chain_id,
        "symbol": "BERA",
        "decimals": decimals,
        "raw": str(int(Decimal(amount).scaleb(decimals))),
        "human": amount,
        "isNative": True,
    }


def _cond(op: str | None = "gt", right: str = "10", symbol: str = "BERA") -> Hop:
    return Hop(kind="cond", symbol=symbol, op=op, right=right)


def _eval(hop: Hop, payloads: list[dict]) -> Verdict:
    return evaluate(hop, payloads, _CHAIN)


def test_threshold_reached() -> None:
    verdict = _eval(_cond(), [_balance(_bera("12.4"))])

    assert verdict.holds is True
    assert verdict.actual == Decimal("12.4")


def test_threshold_not_reached() -> None:
    verdict = _eval(_cond(), [_balance(_bera("8.2"))])

    assert verdict.holds is False
    assert verdict.actual == Decimal("8.2")


def test_equal_value_splits_gt_and_gte() -> None:
    payloads = [_balance(_bera("10"))]

    assert _eval(_cond(op="gt"), payloads).holds is False
    assert _eval(_cond(op="gte"), payloads).holds is True


def test_compares_with_raw_not_truncated_human() -> None:
    # human 被截断成 "10"，raw 实际是 10.0000001：必须按 raw 判，超过门槛。
    item = _bera("10.0000001")
    item["human"] = "10"

    verdict = _eval(_cond(right="10"), [_balance(item)])

    assert verdict.holds is True
    assert verdict.actual == Decimal("10.0000001")


def test_symbol_case_is_ignored() -> None:
    verdict = _eval(_cond(symbol="bera"), [_balance(_bera("12"))])

    assert verdict.holds is True


def test_missing_symbol_asks_instead_of_guessing() -> None:
    verdict = _eval(_cond(), [_balance()])

    assert verdict.holds is None
    assert "BERA" in verdict.reason


def test_blank_symbol_is_rejected() -> None:
    verdict = _eval(_cond(symbol=""), [_balance(_bera("12"))])

    assert verdict.holds is None
    assert verdict.reason


def test_unknown_operator_is_rejected() -> None:
    verdict = _eval(_cond(op=None), [_balance(_bera("12"))])

    assert verdict.holds is None


def test_bad_threshold_is_rejected() -> None:
    verdict = _eval(_cond(right="十"), [_balance(_bera("12"))])

    assert verdict.holds is None


def test_threshold_with_thousands_separator() -> None:
    verdict = _eval(_cond(right="1,000"), [_balance(_bera("1200"))])

    assert verdict.holds is True


def test_other_chain_balance_is_ignored() -> None:
    # 同符号资产在别的链上也有余额；比较的只能是会话所在的那条链。
    verdict = _eval(
        _cond(),
        [_balance(_bera("12", chain_id=1), _bera("8"))],
    )

    assert verdict.holds is False
    assert verdict.actual == Decimal("8")


def test_balance_missing_on_the_session_chain_asks() -> None:
    verdict = _eval(_cond(), [_balance(_bera("12", chain_id=1))])

    assert verdict.holds is None
    assert "BERA" in verdict.reason


def test_human_amount_keeps_decimals() -> None:
    assert human_amount("1234567890123456789", 18) == Decimal("1.234567890123456789")


def test_format_amount_drops_trailing_zeros() -> None:
    assert format_amount(Decimal("12.400")) == "12.4"
    assert format_amount(Decimal("10")) == "10"


def test_verdict_defaults() -> None:
    assert Verdict(holds=True) == Verdict(holds=True, actual=None, reason="")
