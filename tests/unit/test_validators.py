"""Validators 四层套件。全部绕过 LLM 直接调校验层（PRD 16 章验收第 2 条的要求）。"""

from decimal import Decimal

from web3_agent.common.types import BudgetSnapshot, TokenMeta
from web3_agent.config.constants import MSG_STEP_LIMIT, Intent
from web3_agent.domain.validators import (
    GuardLimits,
    StakeBinding,
    TxDraft,
    ValidationContext,
    Verdict,
    parse_human_amount,
    to_base_units,
    validate_amount_risk,
    validate_format,
    validate_runtime_guard,
    validate_whitelist,
    validate_write_request,
)

USER = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
POOL = "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359"
OTHER = "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB"

USDT = TokenMeta(
    chain_id=1,
    symbol="USDT",
    decimals=6,
    token_address=OTHER,
    is_whitelisted=True,
    name="Tether USD",
)
WETH = TokenMeta(
    chain_id=1,
    symbol="WETH",
    decimals=18,
    token_address="0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
    is_whitelisted=True,
    name="Wrapped Ether",
)


def _ctx(**overrides) -> ValidationContext:
    base: dict = {
        "allowed_chain_ids": frozenset({1}),
        "siwe_address": USER,
        "token": USDT,
        "stake": StakeBinding(
            contract_address=POOL, adapter_name="generic", supported_tokens=("USDT",)
        ),
    }
    base.update(overrides)
    return ValidationContext(**base)


def _approve(**overrides) -> TxDraft:
    base: dict = {
        "intent": Intent.APPROVE,
        "chain_id": 1,
        "from_address": USER,
        "token_symbol": "USDT",
        "human_amount": "10",
        "spender": POOL,
    }
    base.update(overrides)
    return TxDraft(**base)


def _stake(**overrides) -> TxDraft:
    base: dict = {
        "intent": Intent.STAKE,
        "chain_id": 1,
        "from_address": USER,
        "token_symbol": "USDT",
        "human_amount": "10",
        "stake_contract": POOL,
    }
    base.update(overrides)
    return TxDraft(**base)


# --- FormatValidator -------------------------------------------------------


def test_amount_converts_to_base_units_exactly() -> None:
    assert to_base_units("1.5", 6) == 1_500_000
    assert to_base_units("10", 6) == 10_000_000
    assert to_base_units("0.000001", 6) == 1


def test_amount_rejects_excess_precision_and_overflow() -> None:
    assert to_base_units("1.0000001", 6) is None  # 超出 decimals，不做四舍五入
    assert to_base_units("1e18", 18) is None  # 科学计数法
    assert to_base_units("-1", 6) is None
    assert to_base_units("1" + "0" * 78, 18) is None  # 溢出 uint256
    assert parse_human_amount("1,000") is None


def test_wrong_checksum_rejected_as_e103() -> None:
    bad = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD"
    result = validate_format(_approve(to_address=bad), _ctx())
    assert result.code == "E103"
    assert result.field == "toAddress"
    assert result.verdict is Verdict.REJECT


def test_zero_amount_from_llm_rejected() -> None:
    result = validate_format(_approve(human_amount="0"), _ctx())
    assert result.code == "E104"


def test_zero_amount_allowed_for_state_machine_approve_zero() -> None:
    result = validate_format(_approve(human_amount="0", allow_zero_amount=True), _ctx())
    assert result.passed


# --- WhiteListValidator ----------------------------------------------------


def test_spender_mismatch_is_e204() -> None:
    result = validate_whitelist(_approve(spender=OTHER), _ctx())
    assert result.code == "E204"
    assert result.message == "授权目标校验失败"
    assert result.verdict is Verdict.REJECT


def test_spender_missing_is_e204() -> None:
    result = validate_whitelist(_approve(spender=None), _ctx())
    assert result.code == "E204"


def test_bound_spender_passes() -> None:
    assert validate_whitelist(_approve(), _ctx()).passed


def test_chain_not_in_whitelist_is_e102() -> None:
    result = validate_whitelist(_approve(chain_id=137), _ctx())
    assert result.code == "E102"


def test_anonymous_write_is_e503() -> None:
    result = validate_whitelist(_approve(), _ctx(siwe_address=None))
    assert result.code == "E503"


def test_identity_mismatch_escalates() -> None:
    result = validate_whitelist(_approve(from_address=OTHER), _ctx())
    assert result.code == "E508"
    assert result.verdict is Verdict.ESCALATE


def test_token_not_whitelisted_is_e201() -> None:
    token = USDT.model_copy(update={"is_whitelisted": False})
    result = validate_whitelist(_approve(), _ctx(token=token))
    assert result.code == "E201"


def test_fee_on_transfer_token_is_e205() -> None:
    token = USDT.model_copy(update={"is_fee_on_transfer": True})
    result = validate_whitelist(_approve(), _ctx(token=token))
    assert result.code == "E205"


def test_stake_without_adapter_token_is_e203() -> None:
    stake = StakeBinding(contract_address=POOL, adapter_name="generic", supported_tokens=())
    result = validate_whitelist(_stake(), _ctx(stake=stake))
    assert result.code == "E203"


def test_stake_contract_must_match_binding() -> None:
    result = validate_whitelist(_stake(stake_contract=OTHER), _ctx())
    assert result.code == "E202"


def test_stake_token_must_come_from_adapter() -> None:
    result = validate_whitelist(_stake(token_symbol="WETH"), _ctx())
    assert result.code == "E201"


# --- AmountRiskValidator ---------------------------------------------------


def test_stablecoin_over_limit_is_e502() -> None:
    result = validate_amount_risk(_approve(human_amount="200"), _ctx(max_amount_usd=Decimal("100")))
    assert result.code == "E502"


def test_token_without_price_feed_is_e404() -> None:
    token = TokenMeta(
        chain_id=1,
        symbol="SHIB",
        decimals=18,
        token_address=OTHER,
        is_whitelisted=True,
        name="Shiba Inu",
    )
    result = validate_amount_risk(_approve(human_amount="1"), _ctx(token=token))
    assert result.code == "E404"


def test_non_stablecoin_uses_price_and_rejects_without_it() -> None:
    draft = _approve(token_symbol="WETH", human_amount="2", spender=POOL)
    assert validate_amount_risk(draft, _ctx(token=WETH)).code == "E404"

    priced = _ctx(token=WETH, usd_price=Decimal("3000"), max_amount_usd=Decimal("5000"))
    assert validate_amount_risk(draft, priced).code == "E502"

    small = _approve(token_symbol="WETH", human_amount="1", spender=POOL)
    assert validate_amount_risk(small, priced).passed


def test_zero_approve_skips_amount_risk() -> None:
    result = validate_amount_risk(_approve(human_amount="0"), _ctx(max_amount_usd=Decimal("0.01")))
    assert result.passed


# --- RuntimeGuardValidator -------------------------------------------------


def test_step_limit_terminates() -> None:
    result = validate_runtime_guard(_approve(), _ctx(budget=BudgetSnapshot(steps=5)))
    assert result.verdict is Verdict.TERMINATE
    assert result.message == MSG_STEP_LIMIT


def test_tx_count_limit_terminates() -> None:
    ctx = _ctx(budget=BudgetSnapshot(tx_count=1), limits=GuardLimits(max_tx_count=1))
    result = validate_runtime_guard(_approve(), ctx)
    assert result.code == "E501"


def test_no_budget_means_no_guard() -> None:
    assert validate_runtime_guard(_approve(), _ctx()).passed


# --- 整链串联 --------------------------------------------------------------


def test_usdt_three_tx_path_is_not_blocked() -> None:
    """D1 的坑：USDT 三笔在 Validators 层必须全部放行，否则状态机第一笔就死。"""
    ctx = _ctx()
    drafts = (
        _approve(human_amount="0", allow_zero_amount=True),  # approve(0)
        _approve(human_amount="10"),  # approve(needed)
        _stake(human_amount="10"),  # stake
    )
    for draft in drafts:
        result = validate_write_request(draft, ctx)
        assert result.passed, (draft.intent, result.code)


def test_native_transfer_passes_with_synthesized_token() -> None:
    """原生币没有 registry 条目，靠 TokenMeta.native() 合成，不能被白名单当"未知代币"拦掉。"""
    ctx = _ctx(
        allowed_chain_ids=frozenset({80094}),
        token=TokenMeta.native(80094),  # BERA
        stake=None,
        usd_price=Decimal("5"),
    )
    draft = TxDraft(
        intent=Intent.TRANSFER,
        chain_id=80094,
        from_address=USER,
        token_symbol="BERA",
        human_amount="1.5",
        to_address=OTHER,
    )
    result = validate_write_request(draft, ctx)
    assert result.passed, result.code
    assert ctx.token is not None and ctx.token.decimals == 18


def test_first_failure_short_circuits() -> None:
    draft = _approve(spender=OTHER, human_amount="999999")
    result = validate_write_request(draft, _ctx(max_amount_usd=Decimal("1")))
    assert result.code == "E204"  # 白名单先拦，不会先报金额超限
