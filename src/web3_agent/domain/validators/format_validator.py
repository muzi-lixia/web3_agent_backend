"""FormatValidator：地址、金额、必填槽位的格式校验（PRD 8.1 / 8.5）。

只做"这串字符能不能变成合法的链上参数"，不做白名单、不看余额、不判风险。
金额一律 bigint（最小单位），不在这一层用 float 中转——0.1 这类值用 float 会掉精度。
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from web3_agent.common.evm_address import normalize_address
from web3_agent.config.constants import WRITE_INTENTS, Intent
from web3_agent.domain.validators.base import (
    TxDraft,
    ValidationContext,
    ValidationResult,
)

# 只接受定点十进制写法：不接受科学计数法、符号、千分位、全角字符。
_AMOUNT_RE = re.compile(r"^\d{1,78}(?:\.\d{1,36})?$")
_UINT256_LIMIT = 1 << 256


def parse_human_amount(value: str) -> Decimal | None:
    text = value.strip()
    if not _AMOUNT_RE.match(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:  # 正则已经挡掉了，留作兜底
        return None


def to_base_units(human_amount: str, decimals: int) -> int | None:
    """人类可读金额 -> 最小单位整数。精度超出代币 decimals 或溢出 uint256 返回 None。

    不做四舍五入：多舍一位就是多发一份钱，宁可让用户重填。
    """
    amount = parse_human_amount(human_amount)
    if amount is None:
        return None
    if -int(amount.as_tuple().exponent) > decimals:
        return None
    base_units = int(amount.scaleb(decimals))
    if not 0 <= base_units < _UINT256_LIMIT:
        return None
    return base_units


def validate_format(draft: TxDraft, ctx: ValidationContext) -> ValidationResult:
    if draft.intent not in WRITE_INTENTS:
        return ValidationResult.reject("E101", field="intent")

    if draft.chain_id <= 0:
        return ValidationResult.reject("E101", field="chainId")

    # stake_contract 允许是别名（PRD 的 stakeContractAliasOrAddress），解析归白名单校验。
    for name, value in (
        ("fromAddress", draft.from_address),
        ("toAddress", draft.to_address),
        ("spenderContractAddress", draft.spender),
    ):
        if value is None:
            continue
        if normalize_address(value) is None:
            return ValidationResult.reject("E103", field=name)

    if draft.intent is Intent.TRANSFER and not draft.to_address:
        return ValidationResult.reject("E101", field="toAddress")

    if draft.intent is Intent.REVOKE_APPROVE:
        return ValidationResult.ok()

    if not draft.human_amount:
        return ValidationResult.reject("E101", field="humanAmount")

    amount = parse_human_amount(draft.human_amount)
    if amount is None:
        return ValidationResult.reject("E104", field="humanAmount")
    if amount == 0 and not draft.allow_zero_amount:
        return ValidationResult.reject("E104", field="humanAmount")

    if ctx.token is not None:
        if to_base_units(draft.human_amount, ctx.token.decimals) is None:
            return ValidationResult.reject("E104", field="humanAmount")

    return ValidationResult.ok()
