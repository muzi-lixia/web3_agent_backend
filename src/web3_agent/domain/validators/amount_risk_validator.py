"""AmountRiskValidator：单笔等值金额上限（PRD 8.5 / 12 章）。

两条规则：
1. 没有价格源的代币直接拒绝写操作（E404），不做"估不出来就放行"——那等于没有上限；
2. 稳定币按 1 计价，其余按传入的 usd_price 换算，超过上限拒绝（E502）。

上限判的是"这一笔"，不是累计。approve(0) 这类 0 金额动作不参与换算。
"""

from __future__ import annotations

from decimal import Decimal

from web3_agent.config.constants import PRICE_FEED_SYMBOLS, STABLECOIN_SYMBOLS
from web3_agent.domain.validators.base import (
    TxDraft,
    ValidationContext,
    ValidationResult,
)
from web3_agent.domain.validators.format_validator import parse_human_amount


def validate_amount_risk(draft: TxDraft, ctx: ValidationContext) -> ValidationResult:
    if not draft.human_amount:
        return ValidationResult.ok()

    amount = parse_human_amount(draft.human_amount)
    if amount is None or amount <= 0:
        # 格式问题归 FormatValidator；这里只管金额大小，不重复报错。
        return ValidationResult.ok()

    token = ctx.token
    if token is None:
        return ValidationResult.ok()  # 无代币元数据时白名单校验会先拦下

    symbol = token.symbol.upper()
    if symbol not in PRICE_FEED_SYMBOLS:
        return ValidationResult.reject("E404", field="tokenSymbol")

    price = Decimal(1) if symbol in STABLECOIN_SYMBOLS else ctx.usd_price
    if price is None or price <= 0:
        return ValidationResult.reject("E404", field="tokenSymbol")

    if amount * price > ctx.max_amount_usd:
        return ValidationResult.reject("E502", field="humanAmount")

    return ValidationResult.ok()
