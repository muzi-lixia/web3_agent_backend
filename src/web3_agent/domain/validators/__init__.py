"""校验套件入口：四层依次执行，第一个非 PASS 的结果即返回。

顺序是固定的：格式 -> 白名单 -> 金额风险 -> 资源闸门。
格式在前，后面的层才能假定"地址能解析、金额能转 bigint"；
资源闸门放最后，避免一个本来就非法的请求把配额打满。

只读链路不要调这里：查询返回结果不是待执行动作，出站校验对它没有意义（PRD 8 章、架构 §537）。
"""

from __future__ import annotations

from web3_agent.domain.validators.amount_risk_validator import validate_amount_risk
from web3_agent.domain.validators.base import (
    GuardLimits,
    StakeBinding,
    TxDraft,
    ValidationContext,
    ValidationResult,
    Validator,
    Verdict,
)
from web3_agent.domain.validators.format_validator import (
    parse_human_amount,
    to_base_units,
    validate_format,
)
from web3_agent.domain.validators.runtime_guard_validator import validate_runtime_guard
from web3_agent.domain.validators.whitelist_validator import validate_whitelist

__all__ = [
    "CHAIN_OF_VALIDATORS",
    "GuardLimits",
    "StakeBinding",
    "TxDraft",
    "ValidationContext",
    "ValidationResult",
    "Verdict",
    "parse_human_amount",
    "to_base_units",
    "validate_amount_risk",
    "validate_format",
    "validate_runtime_guard",
    "validate_whitelist",
    "validate_write_request",
]

CHAIN_OF_VALIDATORS: tuple[Validator, ...] = (
    validate_format,
    validate_whitelist,
    validate_amount_risk,
    validate_runtime_guard,
)


def validate_write_request(
    draft: TxDraft,
    ctx: ValidationContext,
    validators: tuple[Validator, ...] = CHAIN_OF_VALIDATORS,
) -> ValidationResult:
    for validator in validators:
        result = validator(draft, ctx)
        if not result.passed:
            return result
    return ValidationResult.ok()
