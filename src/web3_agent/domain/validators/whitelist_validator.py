"""WhiteListValidator：链、代币、合约、spender、身份绑定（PRD 8.2–8.4、8.7）。

这一层的每一条都是拒绝式的：命中就停，不改写参数、不降级放行。
其中 spender 必须等于本次 workflow 绑定的适配器地址（E204）——该值由状态机注入，
不接受调用方或 LLM 传入，否则"授权给谁"就成了可被话术改写的字段。

原生币转账没有 token_registry 条目，调用方用 `TokenMeta.native(chainId)` 合成后再进来。
"""

from __future__ import annotations

from web3_agent.config.constants import Intent
from web3_agent.domain.validators.base import (
    TxDraft,
    ValidationContext,
    ValidationResult,
)

_SPENDER_INTENTS = frozenset({Intent.APPROVE, Intent.REVOKE_APPROVE})


def validate_whitelist(draft: TxDraft, ctx: ValidationContext) -> ValidationResult:
    if draft.chain_id not in ctx.allowed_chain_ids:
        return ValidationResult.reject("E102", field="chainId")

    if not ctx.siwe_address:
        return ValidationResult.reject("E503", field="siweAddress")
    if draft.from_address.lower() != ctx.siwe_address.lower():
        # 越权尝试按安全事件处理：拒绝 + 告警（架构 §555），不静默放行。
        return ValidationResult.escalate("E508", field="fromAddress")

    stake = ctx.stake
    adapter_symbols: set[str] | None = None

    if draft.intent is Intent.STAKE:
        if stake is None:
            return ValidationResult.reject("E202", field="stakeContractAliasOrAddress")
        if (draft.stake_contract or "").lower() != stake.contract_address.lower():
            return ValidationResult.reject("E202", field="stakeContractAliasOrAddress")
        if not stake.supported_tokens:
            return ValidationResult.reject("E203", field="tokenSymbol")
        adapter_symbols = {symbol.upper() for symbol in stake.supported_tokens}

    token = ctx.token
    if token is None or not token.is_whitelisted:
        return ValidationResult.reject("E201", field="tokenSymbol")
    if token.chain_id != draft.chain_id:
        return ValidationResult.reject("E201", field="tokenSymbol")
    if token.is_fee_on_transfer:
        return ValidationResult.reject("E205", field="tokenSymbol")
    # 请求里声明的代币必须与解析出来的注册表条目一致，否则两边各说一套
    if token.symbol.upper() != draft.token_symbol.upper():
        return ValidationResult.reject("E201", field="tokenSymbol")
    # 质押代币取自适配器，不接受 LLM 传入（PRD 8.3）
    if adapter_symbols is not None and token.symbol.upper() not in adapter_symbols:
        return ValidationResult.reject("E201", field="tokenSymbol")

    if draft.intent in _SPENDER_INTENTS:
        if not draft.spender or stake is None:
            return ValidationResult.reject("E204", field="spenderContractAddress")
        if draft.spender.lower() != stake.contract_address.lower():
            return ValidationResult.reject("E204", field="spenderContractAddress")

    return ValidationResult.ok()
