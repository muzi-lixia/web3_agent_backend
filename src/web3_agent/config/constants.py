"""全局常量：枚举字面量、工具名、固定文案。不随环境变化。

环境相关的值一律放在 settings.py。
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    QUERY_ASSET = "query_asset"
    QUERY_TX = "query_tx"
    QUERY_STAKE = "query_stake"
    QUERY_KNOWLEDGE = "query_knowledge"
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
    {Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE, Intent.QUERY_KNOWLEDGE}
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


class QuotaMetric(str, Enum):
    """资源配额维度。落 quota_counter 时拼进 subject：`{userId}|{metric}`。"""

    STEPS = "steps"
    TOKENS = "tokens"
    RPC_CALLS = "rpcCalls"
    TX_COUNT = "txCount"
    QPS = "qps"


# 固定文案：错误与降级回复一律不走 LLM 生成。
MSG_JSON_DEGRADED = "我暂时没法稳定理解这句话，换个说法再试一次？"
MSG_SECRET_REFUSED = "检测到可能泄露的私钥或助记词，已拒绝本次请求。请勿在对话中粘贴私钥或助记词。"
MSG_WRITE_NOT_ENABLED = "写操作链路（转账 / 授权 / 质押）尚未接入，当前只能查询。"
MSG_CLARIFY = "我需要再确认一点信息才能继续，能补充一下吗？"
MSG_OUT_OF_SCOPE = "这个问题超出我的能力范围，我只能处理链上资产、交易记录与质押相关的问题。"
MSG_STEP_LIMIT = "查询步骤已达到上限，先给你目前拿到的结果。可以把问题拆小一点再问。"
MSG_NO_DATA = "暂时查不到这条链上的数据，稍后再试一次。"

# 原生币元数据：(symbol, decimals)。未命中的链标 NATIVE，不要错标成 ETH。
NATIVE_META: dict[int, tuple[str, int]] = {
    1: ("ETH", 18),
    56: ("BNB", 18),
    137: ("POL", 18),
    80094: ("BERA", 18),
}
NATIVE_FALLBACK: tuple[str, int] = ("NATIVE", 18)

# 链的常用名 → chainId。只用于把「以后都用 BERA 链」这类显式偏好落成数字。
# 不放 bare 的 "bera"：它和代币符号同名，会误判。
CHAIN_ALIASES: dict[str, int] = {
    "ethereum": 1,
    "以太坊": 1,
    "主网": 1,
    "bsc": 56,
    "币安智能链": 56,
    "polygon": 137,
    "berachain": 80094,
    "bera链": 80094,
}

# 有价格源、可做等值金额判定的代币。不在这张表里的代币一律拒绝写操作（E404），不放行。
# 真实 price feed 接入前，这就是"能判定 / 不能判定"的分界线；表本身是配置，不是业务规则。
PRICE_FEED_SYMBOLS: frozenset[str] = frozenset(
    {"ETH", "BNB", "POL", "BERA", "WETH", "WBTC", "USDT", "USDC", "DAI"}
)
# 单笔写操作等值金额上限（PRD 12 章）。运行期可用同名 settings 覆盖，这里是默认值。
DEFAULT_MAX_AMOUNT_USD = 5000
