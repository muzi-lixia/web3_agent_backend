"""全局领域模型。api 层做 schema 转换，不直接对外返回这里的类型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META, Intent, Route


class Hop(BaseModel):
    """多步指令里的一步。kind 决定哪些字段有意义，其余留空。"""

    # kind：这一步做什么：read 取数、cond 比较已取到的数、write 发起写操作。
    kind: Literal["read", "cond", "write"]
    # tool：read 与 write 步要调用的工具名；cond 步为空。
    tool: str = ""
    # args：read 与 write 步的调用参数；cond 步为空。
    args: dict[str, Any] = Field(default_factory=dict)
    # subject：cond 比较的对象，当前只支持资产数量。
    subject: Literal["balance"] = "balance"
    # symbol：cond 比较哪个资产的数量。
    symbol: str = ""
    # op：cond 的运算符；缺省为空表示模型没给出可用的比较方式。
    op: Literal["gt", "gte", "lt", "lte", "eq"] | None = None
    # right：cond 的阈值，人类可读数量字符串。
    right: str = ""
    # rightText：阈值在原话里的原文，如「十」「10 BERA」；用于核对门槛是不是这一轮说的。
    rightText: str = ""


class LlmOutput(BaseModel):
    """LLM 的闭合枚举输出，也是结构化输出（format=json_schema）的约束对象。"""

    # intent：模型识别的闭合业务意图；实际路由由后端计算。
    intent: Intent = Intent.OUT_OF_SCOPE
    # confidence：模型自报的把握程度，只作观测记录，不参与路由判定。
    confidence: float = 0.0
    # needs_clarification：输入缺少确定查询对象所需的信息时为真，由后端直接转澄清。
    needs_clarification: bool = False
    # slots：模型抽取的业务参数字典，例如 tokenSymbol、humanAmount、toAddress。
    slots: dict[str, Any] = Field(default_factory=dict)
    # hops：多步指令的执行计划；为空表示单步，参数全在 slots 里。
    hops: list[Hop] = Field(default_factory=list)


# 数据职责：后端路由决策结果，包含处理路径、原因及需补充的参数。
class RouteDecision(BaseModel):
    # route：后端决定的处理路径：read 查询、write 写操作、clarify 澄清、reject 拒绝。
    route: Route
    # reason：决策或校验失败的原因，供调用方判断和审计使用。
    reason: str
    # missing_slots：本次操作缺失的必填业务参数名。
    missing_slots: list[str] = Field(default_factory=list)


# 数据职责：一次工具执行的可持久化轨迹，不保存完整原始数据响应。
class ToolCallRecord(BaseModel):
    # name：对象的展示名称；具体对象由所属模型决定。
    name: str
    # args：工具执行参数的快照，便于返回轨迹和复盘调用。
    args: dict[str, Any] = Field(default_factory=dict)
    # ok：本次调用是否取得可用结果；失败结果不能作为回答证据。
    ok: bool = True
    # error：失败错误码或错误信息；无错误时为空。
    error: str | None = None
    # latency_ms：本次工具执行耗时，单位为毫秒。
    latency_ms: int = 0


# 数据职责：Agent 内部的一轮处理结果，由 HTTP 层转换为对外响应。
class ChatOutcome(BaseModel):
    # kind：内部回答类型：answer 普通回答、ask 追问、reject 拒答、intent_confirm 待确认。
    kind: str  # answer | ask | reject | intent_confirm
    # content：本轮最终回复正文；失效轮次会被清空。
    content: str
    # intent_preview：写操作待确认的参数快照，非写意图为空。
    intent_preview: dict[str, Any] | None = None
    # task_id：写操作建出的 workflow 任务标识，供确认与状态轮询使用。
    task_id: str | None = None
    # tool_calls：本轮工具调用轨迹，包含名称、参数及执行结果；未调用时可为空。
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # steps：只读图模型调用步数；意图模型等额外调用由独立账本计量。
    steps: int = 0
    # degraded：是否使用降级结果，例如缺少数据证据或结构化输出失败。
    degraded: bool = False
    # citations：当前回答使用的文档证据快照，供引用展示和历史恢复。
    citations: list[dict[str, Any]] = Field(default_factory=list)
    # 读历史之后会话被清空：答案基于已删除的内容，既不落库也不该交给调用方渲染。
    # discarded：生成期间会话已被清空时置真，表示本轮结果应丢弃。
    discarded: bool = False


# 数据职责：网络内的代币元数据，供余额查询和金额精度换算使用。
class TokenMeta(BaseModel):
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int
    # symbol：代币符号，例如 ETH、USDC。
    symbol: str
    # decimals：代币精度：原始整数金额需要除以 10 的该次方才是人类可读金额。
    decimals: int
    # token_address：ERC20 合约地址；合成原生币元数据时使用空字符串。
    token_address: str
    # is_whitelisted：代币是否在系统允许查询的白名单内。
    is_whitelisted: bool
    # name：代币的完整名称；原生币合成元数据时取符号。
    name: str
    # 写链路 E205 的判定依据（PRD 8.3）：转账带税代币一律拒绝授权/质押。
    # is_fee_on_transfer：是否为转账扣税代币；用于描述代币的转账特性。
    is_fee_on_transfer: bool = False

    @classmethod
    def native(cls, chain_id: int) -> TokenMeta:
        """原生币没有 token_registry 条目（PRD 6.4 的 transfer 支持原生币/ERC20），

        由调用方用这个合成一个，符号与精度按链取，不要一律当 ETH。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
        """
        symbol, decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)
        return cls(
            chain_id=chain_id,
            symbol=symbol,
            decimals=decimals,
            token_address="",
            is_whitelisted=True,
            name=symbol,
        )
