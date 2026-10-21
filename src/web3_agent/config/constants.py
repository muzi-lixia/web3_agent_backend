"""全局常量：枚举字面量、工具名、固定文案。不随环境变化。

环境相关的值一律放在 settings.py。
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    QUERY_ASSET = "query_asset"
    QUERY_TX = "query_tx"
    QUERY_STAKE = "query_stake"
    TRANSFER = "transfer"
    APPROVE = "approve"
    STAKE = "stake"
    REVOKE_APPROVE = "revoke_approve"
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"


class Route(str, Enum):
    READ = "read"
    WRITE = "write"
    CLARIFY = "clarify"
    REJECT = "reject"


class ToolGroup(str, Enum):
    LLM_VISIBLE = "llm_visible"
    STATEMACHINE_ONLY = "statemachine_only"


class CallSource(str, Enum):
    LLM = "llm"
    STATE_MACHINE = "state_machine"


READ_INTENTS: frozenset[Intent] = frozenset(
    {Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE}
)
WRITE_INTENTS: frozenset[Intent] = frozenset(
    {Intent.TRANSFER, Intent.APPROVE, Intent.STAKE, Intent.REVOKE_APPROVE}
)

# 写意图必须由 LLM 提供的槽位。chainId 与各类地址由后端注入，不在其中。
REQUIRED_SLOTS: dict[Intent, tuple[str, ...]] = {
    Intent.TRANSFER: ("toAddress", "tokenSymbol", "humanAmount"),
    Intent.APPROVE: ("spenderContractAddress", "tokenSymbol", "humanAmount"),
    Intent.STAKE: ("stakeContractAliasOrAddress", "tokenSymbol", "humanAmount"),
    Intent.REVOKE_APPROVE: ("spenderContractAddress", "tokenSymbol"),
}

TOOL_GET_ASSET_BALANCE = "get_asset_balance"
TOOL_GET_TRANSACTION_RECORDS = "get_transaction_records"
TOOL_GET_STAKE_INFO = "get_stake_info"


class WorkflowState(str, Enum):
    """写链路状态全集（架构 §4 / PRD 9 章）。CHECK_ALLOWANCE、APPROVE_REQUIRED 仅在 stake 子流程出现。"""

    VALIDATE = "VALIDATE"
    INTENT_CONFIRM = "INTENT_CONFIRM"
    CHECK_ALLOWANCE = "CHECK_ALLOWANCE"
    APPROVE_REQUIRED = "APPROVE_REQUIRED"
    SIMULATE = "SIMULATE"
    PREVIEW = "PREVIEW"
    SUBMITTED = "SUBMITTED"
    SETTLED = "SETTLED"
    FAILED = "FAILED"
    REPLACED = "REPLACED"
    ABORTED = "ABORTED"
    EXPIRED_UNSIGNED = "EXPIRED_UNSIGNED"
    EXPIRED_UNVERIFIED = "EXPIRED_UNVERIFIED"


# 终态：到达即停止轮询。SETTLED 的语义是"该 workflow 全部交易完成"，不是"本笔回执成功"。
TERMINAL_STATES: frozenset[WorkflowState] = frozenset(
    {
        WorkflowState.SETTLED,
        WorkflowState.FAILED,
        WorkflowState.REPLACED,
        WorkflowState.ABORTED,
        WorkflowState.EXPIRED_UNSIGNED,
        WorkflowState.EXPIRED_UNVERIFIED,
    }
)


class TxKind(str, Enum):
    APPROVE = "approve"
    STAKE = "stake"
    TRANSFER = "transfer"


class TxRecordStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    REPLACED = "REPLACED"


class GroupStatus(str, Enum):
    IDLE = "idle"
    PARTIAL = "partial"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    REVOKED = "revoked"


class ExpireType(str, Enum):
    NONE = "NONE"
    EXPIRED_UNSIGNED = "EXPIRED_UNSIGNED"
    EXPIRED_UNVERIFIED = "EXPIRED_UNVERIFIED"


class SimulationStatus(str, Enum):
    OK = "ok"
    REVERT = "revert"
    ESTIMATE_FAILED = "estimate_failed"
    RPC_ERROR = "rpc_error"


class QuotaMetric(str, Enum):
    """资源配额维度。落 quota_counter 时拼进 subject：`{userId}|{metric}`。"""

    STEPS = "steps"
    TOKENS = "tokens"
    RPC_CALLS = "rpcCalls"
    TX_COUNT = "txCount"
    QPS = "qps"


# 单 workflow 的 TxIntent 上限：stake 走 USDT 时是 approve(0) + approve(needed) + stake。
# 其余意图一律只允许一笔。
MAX_TX_COUNT_BY_INTENT: dict[Intent, int] = {
    Intent.STAKE: 3,
    Intent.TRANSFER: 1,
    Intent.APPROVE: 1,
    Intent.REVOKE_APPROVE: 1,
}


# 固定文案：错误与降级回复一律不走 LLM 生成。
MSG_JSON_DEGRADED = "我暂时没法稳定理解这句话，换个说法再试一次？"
MSG_SECRET_REFUSED = "检测到可能泄露的私钥或助记词，已拒绝本次请求。请勿在对话中粘贴私钥或助记词。"
MSG_WRITE_NOT_ENABLED = "写操作链路（转账 / 授权 / 质押）尚未接入，当前只能查询。"
MSG_CLARIFY = "我需要再确认一点信息才能继续，能补充一下吗？"
MSG_OUT_OF_SCOPE = "这个问题超出我的能力范围，我只能处理链上资产、交易记录与质押相关的问题。"
MSG_STEP_LIMIT = "查询步骤已达到上限，先给你目前拿到的结果。可以把问题拆小一点再问。"

# 原生币元数据：(symbol, decimals)。未命中的链标 NATIVE，不要错标成 ETH。
NATIVE_META: dict[int, tuple[str, int]] = {
    1: ("ETH", 18),
    56: ("BNB", 18),
    137: ("POL", 18),
    80094: ("BERA", 18),
}
NATIVE_FALLBACK: tuple[str, int] = ("NATIVE", 18)

# 有价格源、可做等值金额判定的代币。不在这张表里的代币一律拒绝写操作（E404），不放行。
# 真实 price feed 接入前，这就是"能判定 / 不能判定"的分界线；表本身是配置，不是业务规则。
PRICE_FEED_SYMBOLS: frozenset[str] = frozenset(
    {"ETH", "BNB", "POL", "BERA", "WETH", "WBTC", "USDT", "USDC", "DAI"}
)
# 稳定币价格按 1 计，不依赖外部喂价，避免没接 feed 时把正常授权全拦掉。
STABLECOIN_SYMBOLS: frozenset[str] = frozenset({"USDT", "USDC", "DAI"})

# 单笔写操作等值金额上限（PRD 12 章）。运行期可用同名 settings 覆盖，这里是默认值。
DEFAULT_MAX_AMOUNT_USD = 5000
