"""全局领域模型。api 层做 schema 转换，不直接对外返回这里的类型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META, Intent, Route


class LlmOutput(BaseModel):
    """LLM 的闭合枚举输出，也是结构化输出（format=json_schema）的约束对象。"""

    intent: Intent = Intent.OUT_OF_SCOPE
    confidence: float = 0.0
    slots: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    route: Route
    reason: str
    missing_slots: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    latency_ms: int = 0


class ChatOutcome(BaseModel):
    kind: str  # answer | ask | reject
    content: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    steps: int = 0
    degraded: bool = False
    citations: list[dict[str, Any]] = Field(default_factory=list)
    # 读历史之后会话被清空：答案基于已删除的内容，既不落库也不该交给调用方渲染。
    discarded: bool = False


class TokenMeta(BaseModel):
    chain_id: int
    symbol: str
    decimals: int
    token_address: str
    is_whitelisted: bool
    name: str
    # 写链路 E205 的判定依据（PRD 8.3）：转账带税代币一律拒绝授权/质押。
    is_fee_on_transfer: bool = False

    @classmethod
    def native(cls, chain_id: int) -> TokenMeta:
        """原生币没有 token_registry 条目（PRD 6.4 的 transfer 支持原生币/ERC20），
        由调用方用这个合成一个，符号与精度按链取，不要一律当 ETH。"""
        symbol, decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)
        return cls(
            chain_id=chain_id,
            symbol=symbol,
            decimals=decimals,
            token_address="",
            is_whitelisted=True,
            name=symbol,
        )
