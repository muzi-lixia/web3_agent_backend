"""IntentRouter：唯一路由决策点，纯函数。

只决定走哪条大路径，不校验参数合法性——那是 Validators 的事。
把这两件事分开，是为了避免"边校验边改路由"导致同一条输入走出两条路。
"""

from __future__ import annotations

from web3_agent.common.types import LlmOutput, RouteDecision
from web3_agent.config.constants import (
    READ_INTENTS,
    REQUIRED_SLOTS,
    WRITE_INTENTS,
    Intent,
    Route,
)


def route_intent(
    output: LlmOutput,
    *,
    confidence_threshold: float,
    required_slots: dict[Intent, tuple[str, ...]] | None = None,
) -> RouteDecision:
    if output.confidence < confidence_threshold:
        return RouteDecision(route=Route.CLARIFY, reason="low_confidence")

    if output.intent is Intent.OUT_OF_SCOPE:
        return RouteDecision(route=Route.REJECT, reason="out_of_scope")

    if output.intent is Intent.CHITCHAT or output.intent in READ_INTENTS:
        return RouteDecision(route=Route.READ, reason=output.intent.value)

    if output.intent in WRITE_INTENTS:
        slots = required_slots or REQUIRED_SLOTS
        missing = [k for k in slots.get(output.intent, ()) if not output.slots.get(k)]
        if missing:
            return RouteDecision(
                route=Route.CLARIFY, reason="missing_slots", missing_slots=missing
            )
        return RouteDecision(route=Route.WRITE, reason=output.intent.value)

    return RouteDecision(route=Route.REJECT, reason="unknown_intent")
