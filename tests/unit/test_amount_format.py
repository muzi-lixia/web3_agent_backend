"""余额展示格式：满精度串会被模型逐位抄进回答，整数还会变成科学计数法。"""

from __future__ import annotations

from web3_agent.web3.tools.read.get_asset_balance import _human


def test_full_precision_is_truncated_to_six_decimals() -> None:
    # 这是真实返回过的余额，模型把它原样抄进了回答
    assert _human(1588110696995879326, 18) == "1.58811"


def test_whole_number_is_not_scientific_notation() -> None:
    # Decimal.normalize() 对 100 给的是 1E+2，改前就是这个输出
    assert _human(100 * 10**18, 18) == "100"


def test_zero_stays_zero() -> None:
    assert _human(0, 18) == "0"


def test_dust_is_not_shown_as_zero() -> None:
    # 截断后为 0 但确实有余额，不能显示成 0
    assert _human(900, 18) == "<0.000001"


def test_precision_follows_token_decimals() -> None:
    assert _human(1_500_000, 6) == "1.5"
