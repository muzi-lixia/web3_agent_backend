from web3_agent.agent.intent_router import route_intent
from web3_agent.common.types import LlmOutput
from web3_agent.config.constants import Intent, Route


def _route(
    intent: Intent, confidence: float, message: str, clarify: bool = False, **slots
):
    return route_intent(
        LlmOutput(
            intent=intent,
            confidence=confidence,
            needs_clarification=clarify,
            slots=slots,
        ),
        user_message=message,
    )


def test_read_intent_routes_to_read() -> None:
    decision = _route(Intent.QUERY_ASSET, 0.9, "我的资产")
    assert decision.route is Route.READ
    assert decision.reason == "query_asset"


def test_needs_clarification_forces_clarify() -> None:
    decision = _route(Intent.QUERY_KNOWLEDGE, 1.0, "什么是 Gas", clarify=True)
    assert decision.route is Route.CLARIFY
    assert decision.reason == "needs_clarification"


def test_low_confidence_alone_does_not_clarify() -> None:
    # confidence 只作观测：模型自报 0.1 但信息齐全时照常执行。
    decision = _route(Intent.QUERY_ASSET, 0.1, "我的资产")
    assert decision.route is Route.READ


def test_low_confidence_write_with_all_slots_still_writes() -> None:
    decision = _route(
        Intent.TRANSFER,
        0.3,
        "把 10 USDC 转给 0x1111111111111111111111111111111111111111",
        toAddress="0x" + "1" * 40,
        tokenSymbol="USDC",
        humanAmount="10",
    )
    assert decision.route is Route.WRITE


def test_out_of_scope_rejects() -> None:
    decision = _route(Intent.OUT_OF_SCOPE, 0.95, "BTC 现在多少钱")
    assert decision.route is Route.REJECT


def test_write_intent_with_all_slots_routes_to_write() -> None:
    decision = _route(
        Intent.TRANSFER,
        0.9,
        "把 10 USDC 转给 0x1111111111111111111111111111111111111111",
        toAddress="0x" + "1" * 40,
        tokenSymbol="USDC",
        humanAmount="10",
    )
    assert decision.route is Route.WRITE


def test_write_intent_missing_slots_asks() -> None:
    decision = _route(
        Intent.STAKE, 0.9, "质押 ETH 到 A 池", tokenSymbol="ETH"
    )
    assert decision.route is Route.CLARIFY
    assert set(decision.missing_slots) == {"stakeContractAliasOrAddress", "humanAmount"}


def test_query_intents_do_not_require_slots() -> None:
    # 读意图不在 REQUIRED_SLOTS 里：质押合约缺了也放行，由工具层如实报未接入。
    decision = _route(Intent.QUERY_STAKE, 0.9, "我质押了多少")
    assert decision.route is Route.READ
    assert decision.missing_slots == []


def test_chitchat_does_not_enter_write_path() -> None:
    decision = _route(Intent.CHITCHAT, 0.99, "你好")
    assert decision.route is Route.READ


def test_empty_slot_value_counts_as_missing() -> None:
    decision = _route(
        Intent.TRANSFER,
        0.9,
        "把 1 USDC 转给某地址",
        toAddress="",
        tokenSymbol="USDC",
        humanAmount="1",
    )
    assert decision.route is Route.CLARIFY
    assert decision.missing_slots == ["toAddress"]


def test_vague_query_without_object_asks() -> None:
    for message in ("查询", "查一下", "帮我看看", "看一下我的", "这个呢", "那个怎么样", "怎么算"):
        decision = _route(Intent.QUERY_KNOWLEDGE, 0.9, message)
        assert decision.route is Route.CLARIFY, message
        assert decision.reason == "vague_input", message


def test_substantive_query_still_reads() -> None:
    cases = [
        ("什么是 Gas", Intent.QUERY_KNOWLEDGE),
        ("滑点是什么", Intent.QUERY_KNOWLEDGE),
        ("我的资产", Intent.QUERY_ASSET),
        ("我最近 10 笔交易", Intent.QUERY_TX),
        ("授权失败通常是什么原因", Intent.QUERY_KNOWLEDGE),
    ]
    for message, intent in cases:
        assert _route(intent, 0.9, message).route is Route.READ, message


def test_short_chitchat_is_not_vague() -> None:
    # 闲聊的短句本身是完整表达，不能套用查询意图的"有没有对象"判据。
    assert _route(Intent.CHITCHAT, 0.9, "嗯").route is Route.READ


def test_plan_with_write_step_skips_missing_slots():
    # 多步计划把参数放在写步里，slots 为空不代表信息不足。
    decision = route_intent(
        LlmOutput(
            intent=Intent.TRANSFER,
            hops=[
                {"kind": "read", "tool": "get_asset_balance", "args": {"symbol": "BERA"}},
                {"kind": "cond", "symbol": "BERA", "op": "gte", "right": "10"},
                {
                    "kind": "write",
                    "tool": "post_transfer",
                    "args": {
                        "toAddress": "0x" + "1" * 40,
                        "tokenSymbol": "BERA",
                        "humanAmount": "5",
                    },
                },
            ],
        ),
        user_message="余额超过 10 BERA 就转 5 BERA",
    )

    assert decision.route is Route.WRITE
    assert decision.missing_slots == []


def test_single_step_write_without_slots_still_clarifies():
    decision = _route(Intent.TRANSFER, 0.9, "把一些 USDC 转给他")
    assert decision.route is Route.CLARIFY
    assert decision.reason == "missing_slots"


def test_complete_plan_is_not_blocked_by_clarification_flag():
    # 计划已经把「查什么、比什么、做什么」写全，模型的自我怀疑不能把它拦成追问。
    decision = route_intent(
        LlmOutput(
            intent=Intent.TRANSFER,
            needs_clarification=True,
            hops=[
                {"kind": "read", "tool": "get_asset_balance"},
                {"kind": "cond", "symbol": "BERA", "op": "gte", "right": "10"},
                {"kind": "write", "tool": "post_transfer", "args": {"humanAmount": "5"}},
            ],
        ),
        user_message="余额够 10 就转 5 BERA",
    )

    assert decision.route is Route.WRITE
    assert decision.reason == "plan"


def test_plan_with_two_writes_still_clarifies():
    # 形状不完整的计划不豁免：追问的是真正缺的信息。
    decision = route_intent(
        LlmOutput(
            intent=Intent.TRANSFER,
            needs_clarification=True,
            hops=[
                {"kind": "read", "tool": "get_asset_balance"},
                {"kind": "write", "tool": "post_transfer", "args": {"humanAmount": "5"}},
                {"kind": "write", "tool": "post_transfer", "args": {"humanAmount": "5"}},
            ],
        ),
        user_message="转 5 给 A，再转 5 给 B",
    )

    assert decision.route is Route.CLARIFY
