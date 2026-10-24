"""IntentRouter：唯一路由决策点，纯函数。

只决定走哪条大路径，不校验参数合法性——那是 Validators 的事。
把这两件事分开，是为了避免"边校验边改路由"导致同一条输入走出两条路。
"""

from __future__ import annotations

import re

from web3_agent.common.types import LlmOutput, RouteDecision
from web3_agent.config.constants import (
    READ_INTENTS,
    REQUIRED_SLOTS,
    WRITE_INTENTS,
    Intent,
    Route,
)

# _VAGUE：只表达动作或语气、不指向任何查询对象的片段。
# 模型看不出"自己该不该追问"：七条信息不足的输入里它只报了两条（见
# tests/golden_suite/BASELINE.md），所以这里只兜"剥完什么都不剩"这一类，
# 不做语义理解，也不试图判断"信息够不够详细"。
_VAGUE = re.compile(
    r"帮我|帮忙|麻烦|请问|请|一下|查看|查询|查|看看|看|我的|我|这个|那个|这些|那些|"
    r"怎么样|怎么|如何|什么|呢|吗|吧|啊|的|是|有|在|了|下|[\s，。？！,.?!、]"
)
# _MIN_SUBSTANCE：剥离语气与动作词后至少剩下的字符数。
_MIN_SUBSTANCE = 2


def _has_substance(text: str) -> bool:
    """判断输入是否还有查询对象可指。空串本身属于输入校验问题，这里放行。

    Args:
        text: 用户本轮原始输入。
    """
    if not text:
        return True
    return len(_VAGUE.sub("", text)) >= _MIN_SUBSTANCE


def route_intent(
    output: LlmOutput,
    *,
    user_message: str,
    required_slots: dict[Intent, tuple[str, ...]] | None = None,
) -> RouteDecision:
    # 返回值：RouteDecision，包含最终路由、原因和缺失槽位。
    """依据澄清信号、输入实质、意图类别和必填槽位选择处理路径，不执行工具。

    confidence 不参与判定：它是模型自报的数字，实测在"信息不足"和"信息明确"
    两类输入上无法区分（见 tests/golden_suite/BASELINE.md），据此澄清只会误伤。

    Args:
        output: 已通过结构化校验的 LlmOutput 意图结果。
        user_message: 用户本轮原始输入，用于判断查询意图有没有可执行的对象。
        required_slots: 各业务意图必须具备的参数名集合。
    """
    if output.needs_clarification:
        return RouteDecision(route=Route.CLARIFY, reason="needs_clarification")

    if output.intent is Intent.OUT_OF_SCOPE:
        return RouteDecision(route=Route.REJECT, reason="out_of_scope")

    # 查询意图要能直接形成一个检索或查询动作；闲聊和越界不适用（"嗯""你好"是完整表达）。
    if output.intent in READ_INTENTS and not _has_substance(user_message):
        return RouteDecision(route=Route.CLARIFY, reason="vague_input")

    # 缺参数就执行不了，追问比分派到下游更早。
    slots = required_slots or REQUIRED_SLOTS
    missing = [k for k in slots.get(output.intent, ()) if not output.slots.get(k)]
    if missing:
        return RouteDecision(
            route=Route.CLARIFY, reason="missing_slots", missing_slots=missing
        )

    if output.intent is Intent.CHITCHAT or output.intent in READ_INTENTS:
        return RouteDecision(route=Route.READ, reason=output.intent.value)

    if output.intent in WRITE_INTENTS:
        return RouteDecision(route=Route.WRITE, reason=output.intent.value)

    return RouteDecision(route=Route.REJECT, reason="unknown_intent")
