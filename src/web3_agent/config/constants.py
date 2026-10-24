"""全局常量：枚举字面量、工具名、固定文案。不随环境变化。

环境相关的值一律放在 settings.py。
"""

from __future__ import annotations

from enum import Enum


# 数据职责：模型可输出的闭合业务意图枚举。
class Intent(str, Enum):
    # QUERY_ASSET：查询链上资产余额。
    QUERY_ASSET = "query_asset"
    # QUERY_TX：查询当前钱包的最近交易记录。
    QUERY_TX = "query_tx"
    # QUERY_STAKE：查询白名单质押合约信息。
    QUERY_STAKE = "query_stake"
    # QUERY_KNOWLEDGE：查询已审核的静态 Web3 文档知识。
    QUERY_KNOWLEDGE = "query_knowledge"
    # TRANSFER：转账写意图，当前主链路仅识别并提示未接入。
    TRANSFER = "transfer"
    # APPROVE：代币授权写意图，当前主链路不执行。
    APPROVE = "approve"
    # STAKE：质押写意图，当前主链路不执行。
    STAKE = "stake"
    # REVOKE_APPROVE：撤销授权写意图，当前主链路不执行。
    REVOKE_APPROVE = "revoke_approve"
    # CHITCHAT：无需链上或文档取证的闲聊。
    CHITCHAT = "chitchat"
    # OUT_OF_SCOPE：超出系统支持范围的意图。
    OUT_OF_SCOPE = "out_of_scope"


# 数据职责：后端业务路径枚举。
class Route(str, Enum):
    # READ：进入只读查询或闲聊处理路径。
    READ = "read"
    # WRITE：进入写操作分支，当前版本只返回未接入提示。
    WRITE = "write"
    # CLARIFY：信息不足或置信度偏低，要求用户补充说明。
    CLARIFY = "clarify"
    # REJECT：意图超出范围，返回固定拒答。
    REJECT = "reject"


# 数据职责：工具对不同执行器的可见性分组。
class ToolGroup(str, Enum):
    # LLM_VISIBLE：可绑定给模型的只读工具分组。
    LLM_VISIBLE = "llm_visible"
    # STATEMACHINE_ONLY：仅允许确定性状态机使用的工具分组。
    STATEMACHINE_ONLY = "statemachine_only"


# 数据职责：工具调用者来源枚举，用于约束写工具执行。
class CallSource(str, Enum):
    # LLM：由模型或只读图发起的调用。
    LLM = "llm"
    # STATE_MACHINE：由写操作状态机发起的调用。
    STATE_MACHINE = "state_machine"


# READ_INTENTS：路由器支持的只读查询意图集合。
READ_INTENTS: frozenset[Intent] = frozenset(
    {Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE, Intent.QUERY_KNOWLEDGE}
)
# WRITE_INTENTS：路由器识别的写意图集合，当前不会执行链上写操作。
WRITE_INTENTS: frozenset[Intent] = frozenset(
    {Intent.TRANSFER, Intent.APPROVE, Intent.STAKE, Intent.REVOKE_APPROVE}
)

# 写意图必须由 LLM 提供的槽位。chainId 与各类地址由后端注入，不在其中。
# REQUIRED_SLOTS：各写意图需要用户提供的业务参数名，钱包身份及网络由服务端约束。
REQUIRED_SLOTS: dict[Intent, tuple[str, ...]] = {
    Intent.TRANSFER: ("toAddress", "tokenSymbol", "humanAmount"),
    Intent.APPROVE: ("spenderContractAddress", "tokenSymbol", "humanAmount"),
    Intent.STAKE: ("stakeContractAliasOrAddress", "tokenSymbol", "humanAmount"),
    Intent.REVOKE_APPROVE: ("spenderContractAddress", "tokenSymbol"),
}

# TOOL_GET_ASSET_BALANCE：工具在注册表和调用轨迹中使用的稳定名称。
TOOL_GET_ASSET_BALANCE = "get_asset_balance"
# TOOL_GET_TRANSACTION_RECORDS：工具在注册表和调用轨迹中使用的稳定名称。
TOOL_GET_TRANSACTION_RECORDS = "get_transaction_records"
# TOOL_GET_STAKE_INFO：工具在注册表和调用轨迹中使用的稳定名称。
TOOL_GET_STAKE_INFO = "get_stake_info"


class QuotaMetric(str, Enum):
    """资源配额维度。落 quota_counter 时拼进 subject：`{userId}|{metric}`。"""

    # STEPS：模型调用次数的记账维度。
    STEPS = "steps"
    # TOKENS：token 用量的预留维度，当前模型入口没有回传 usage。
    TOKENS = "tokens"
    # RPC_CALLS：RPC、浏览器、向量化及重排请求合计的下游消耗维度，沿用 rpcCalls 名称。
    RPC_CALLS = "rpcCalls"
    # TX_COUNT：交易生成数量的预留维度。
    TX_COUNT = "txCount"
    # QPS：每分钟请求次数；沿用 qps 字段名，实际按分钟分桶。
    QPS = "qps"


# 固定文案：错误与降级回复一律不走 LLM 生成。
# MSG_JSON_DEGRADED：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_JSON_DEGRADED = "我暂时没法稳定理解这句话，换个说法再试一次？"
# MSG_SECRET_REFUSED：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_SECRET_REFUSED = "检测到可能泄露的私钥或助记词，已拒绝本次请求。请勿在对话中粘贴私钥或助记词。"
# MSG_WRITE_NOT_ENABLED：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_WRITE_NOT_ENABLED = "写操作链路（转账 / 授权 / 质押）尚未接入，当前只能查询。"
# MSG_CLARIFY：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_CLARIFY = "我需要再确认一点信息才能继续，能补充一下吗？"
# MSG_OUT_OF_SCOPE：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_OUT_OF_SCOPE = "这个问题超出我的能力范围，我只能处理链上资产、交易记录与质押相关的问题。"
# MSG_STEP_LIMIT：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_STEP_LIMIT = "查询步骤已达到上限，先给你目前拿到的结果。可以把问题拆小一点再问。"
# MSG_NO_DATA：当前场景使用的固定用户提示，不交给模型生成或改写。
MSG_NO_DATA = "暂时查不到这条链上的数据，稍后再试一次。"

# 原生币元数据：(symbol, decimals)。未命中的链标 NATIVE，不要错标成 ETH。
# NATIVE_META：网络编号到原生币 (符号, 精度) 的映射。
NATIVE_META: dict[int, tuple[str, int]] = {
    1: ("ETH", 18),
    56: ("BNB", 18),
    137: ("POL", 18),
    80094: ("BERA", 18),
}
# NATIVE_FALLBACK：未知网络使用的原生币符号和精度回退值。
NATIVE_FALLBACK: tuple[str, int] = ("NATIVE", 18)

# 链的常用名 → chainId。只用于把「以后都用 BERA 链」这类显式偏好落成数字。
# 不放 bare 的 "bera"：它和代币符号同名，会误判。
# CHAIN_ALIASES：用户明确指定默认网络时使用的自然语言别名映射。
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
# PRICE_FEED_SYMBOLS：预留美元等值风控支持的代币符号集合。
PRICE_FEED_SYMBOLS: frozenset[str] = frozenset(
    {"ETH", "BNB", "POL", "BERA", "WETH", "WBTC", "USDT", "USDC", "DAI"}
)
# 单笔写操作等值金额上限（PRD 12 章）。运行期可用同名 settings 覆盖，这里是默认值。
# DEFAULT_MAX_AMOUNT_USD：预留写链路的默认单笔美元等值金额上限。
DEFAULT_MAX_AMOUNT_USD = 5000
