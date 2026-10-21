"""RuntimeGuardValidator：资源闸门（PRD 8.6 / 架构 §488）。

输入是 QuotaController 聚合好的 BudgetSnapshot，不自己去多处读计数——
分散读会让"到底算到第几步"出现两个答案，也没法单测。

语义分两类（架构 §556 / §555）：
- 步数与配额耗尽属于非故障终止（TERMINATE），给提示、不跳人工；
- 交易笔数用尽同样终止，因为再下发就是并发两笔，前端无法正确串联。
"""

from __future__ import annotations

from web3_agent.config.constants import MSG_STEP_LIMIT
from web3_agent.domain.validators.base import (
    TxDraft,
    ValidationContext,
    ValidationResult,
)


def validate_runtime_guard(draft: TxDraft, ctx: ValidationContext) -> ValidationResult:
    budget = ctx.budget
    if budget is None:
        return ValidationResult.ok()

    limits = ctx.limits

    if budget.steps >= limits.max_steps:
        return ValidationResult.terminate(message=MSG_STEP_LIMIT, field="steps")

    if limits.max_tokens is not None and budget.tokens >= limits.max_tokens:
        return ValidationResult.terminate("E501", field="tokens")

    if limits.max_rpc_calls is not None and budget.rpc_calls >= limits.max_rpc_calls:
        return ValidationResult.terminate("E501", field="rpcCalls")

    if limits.max_tx_count is not None and budget.tx_count >= limits.max_tx_count:
        return ValidationResult.terminate("E501", field="txCount")

    return ValidationResult.ok()
