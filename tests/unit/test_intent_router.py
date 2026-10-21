from web3_agent.agent.intent_router import route_intent
from web3_agent.common.types import LlmOutput
from web3_agent.config.constants import Intent, Route


def _out(intent: Intent, confidence: float, **slots) -> LlmOutput:
    return LlmOutput(intent=intent, confidence=confidence, slots=slots)


def test_read_intent_routes_to_read() -> None:
    decision = route_intent(_out(Intent.QUERY_ASSET, 0.9), confidence_threshold=0.75)
    assert decision.route is Route.READ
    assert decision.reason == "query_asset"


def test_low_confidence_forces_clarify() -> None:
    decision = route_intent(_out(Intent.QUERY_ASSET, 0.5), confidence_threshold=0.75)
    assert decision.route is Route.CLARIFY
    assert decision.reason == "low_confidence"


def test_out_of_scope_rejects() -> None:
    decision = route_intent(_out(Intent.OUT_OF_SCOPE, 0.95), confidence_threshold=0.75)
    assert decision.route is Route.REJECT


def test_write_intent_with_all_slots_routes_to_write() -> None:
    decision = route_intent(
        _out(
            Intent.TRANSFER,
            0.9,
            toAddress="0x" + "1" * 40,
            tokenSymbol="USDC",
            humanAmount="10",
        ),
        confidence_threshold=0.75,
    )
    assert decision.route is Route.WRITE


def test_write_intent_missing_slots_asks() -> None:
    decision = route_intent(
        _out(Intent.STAKE, 0.9, tokenSymbol="ETH"),
        confidence_threshold=0.75,
    )
    assert decision.route is Route.CLARIFY
    assert set(decision.missing_slots) == {"stakeContractAliasOrAddress", "humanAmount"}


def test_chitchat_does_not_enter_write_path() -> None:
    decision = route_intent(_out(Intent.CHITCHAT, 0.99), confidence_threshold=0.75)
    assert decision.route is Route.READ
