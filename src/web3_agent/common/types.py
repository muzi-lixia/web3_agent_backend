"""全局领域模型。api 层做 schema 转换，不直接对外返回这里的类型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from web3_agent.config.constants import (
    NATIVE_FALLBACK,
    NATIVE_META,
    ExpireType,
    GroupStatus,
    Intent,
    Route,
    SimulationStatus,
    TxKind,
    TxRecordStatus,
    WorkflowState,
)


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


class StakePosition(BaseModel):
    staked_amount: int
    pending_reward: int
    unlock_at: int | None = None
    status: str = "active"


class BalanceDiff(BaseModel):
    """模拟前后余额差。金额一律 bigint（wei / 最小单位），出接口时再转十进制字符串。"""

    model_config = ConfigDict(populate_by_name=True)

    token_symbol: str = Field(alias="tokenSymbol")
    before: int
    after: int


class TxIntentMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    human_readable_summary: str = Field(default="", alias="humanReadableSummary")
    risk_tags: list[str] = Field(default_factory=list, alias="riskTags")
    balance_diff: list[BalanceDiff] = Field(default_factory=list, alias="balanceDiff")
    created_at: int = Field(alias="createdAt")
    expires_at: int = Field(alias="expiresAt")
    intent_fingerprint: str = Field(default="", alias="intentFingerprint")


class TxIntent(BaseModel):
    """待签名交易意图。后端禁止填充 nonce / gasPrice / maxFeePerGas，全部交给前端钱包。

    PRD 6.3 与架构 7.3 对 estimatedGasLimit 的层级不一致（meta 内 vs 顶层），
    领域模型取顶层 bigint；对外 schema 序列化时再落到 PRD 的 meta 位置。
    """

    model_config = ConfigDict(populate_by_name=True)

    chain_id: int = Field(alias="chainId")
    from_address: str = Field(alias="from")
    to: str
    value: int  # wei
    data: str
    method: str  # transfer | approve | stake | revoke_approve
    estimated_gas_limit: int | None = Field(default=None, alias="estimatedGasLimit")
    meta: TxIntentMeta


class TxRecord(BaseModel):
    """单笔交易记录。nonce 用于 REPLACED 判定：receipt 为空且已过 N 个区块时，
    拿这里存的 nonce 与链上已用 nonce 比对。"""

    model_config = ConfigDict(populate_by_name=True)

    index: int
    kind: TxKind
    tx_hash: str | None = Field(default=None, alias="txHash")
    status: TxRecordStatus = TxRecordStatus.PENDING
    nonce: int | None = None
    tx_intent_snapshot: TxIntent | None = Field(default=None, alias="txIntentSnapshot")


class WorkflowStoreItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_id: str = Field(alias="workflowId")
    task_id: str = Field(alias="taskId")
    status: WorkflowState
    siwe_user_id: str = Field(alias="siweUserId")
    siwe_address: str = Field(alias="siweAddress")
    intent: Intent
    chain_id: int = Field(alias="chainId")
    slots_snapshot: dict[str, Any] = Field(default_factory=dict, alias="slotsSnapshot")
    generated_tx_count: int = Field(default=0, alias="generatedTxCount")
    max_tx_count: int = Field(default=1, alias="maxTxCount")
    approve_fingerprints: list[str] = Field(default_factory=list, alias="approveFingerprints")
    tx_records: list[TxRecord] = Field(default_factory=list, alias="txRecords")
    transaction_group_id: str = Field(alias="transactionGroupId")
    group_status: GroupStatus = Field(default=GroupStatus.IDLE, alias="groupStatus")
    expire_at: int = Field(alias="expireAt")
    expire_type: ExpireType = Field(default=ExpireType.NONE, alias="expireType")


class SimulationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: SimulationStatus
    revert_type: str | None = Field(default=None, alias="revertType")
    revert_raw: str | None = Field(default=None, alias="revertRaw")
    gas_used: int | None = Field(default=None, alias="gasUsed")
    balance_diff: list[BalanceDiff] = Field(default_factory=list, alias="balanceDiff")


class BudgetSnapshot(BaseModel):
    """资源配额快照，喂给 RuntimeGuardValidator。只做防滥用，不是资金风控。"""

    model_config = ConfigDict(populate_by_name=True)

    steps: int = 0
    tokens: int = 0
    rpc_calls: int = Field(default=0, alias="rpcCalls")
    tx_count: int = Field(default=0, alias="txCount")

