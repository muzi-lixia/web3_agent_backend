"""聊天接口的 HTTP 契约。

PRD 7.1 的请求体带 `sessionId`，架构 7.7 却是单线性会话：一个用户一条历史，
`conversation_message` 只按 siwe_user_id 归属，没有第二个分区键。V1 按架构实现，
因此不收 `sessionId`。其余字段名与 PRD 7.1 一致。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# 数据职责：客户端钱包引用；服务端仍以已验证的登录身份校验归属。
class WalletRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # address：钱包地址；登录接口用它绑定签名身份，查询接口用它指定账户。
    address: str
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int = Field(alias="chainId")


# 数据职责：单轮聊天请求，包含问题、可选钱包、知识库和幂等键。
class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # wallet：客户端传入的钱包信息；未提供时由接口按登录身份处理。
    wallet: WalletRef | None = None
    # user_message：本次用户输入的原文，最大长度由字段校验限制。
    user_message: str = Field(alias="userMessage", min_length=1, max_length=4000)
    # knowledge_base_id：本轮知识问答选择的知识库标识，默认 default。
    knowledge_base_id: str = Field(default="default",alias="knowledgeBaseId",pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    # 客户端重试时带同一个值，服务端只跑一轮并复用上次的响应。
    # idempotency_key：客户端重试标识；同一主体同一个键用于复用已有结果。
    idempotency_key: str | None = Field(
        default=None, alias="idempotencyKey", min_length=8, max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


# 数据职责：预留的用户确认视图，展示动作、代币、金额和目标。
class IntentPreview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # action：待确认的业务动作名称，例如 transfer 或 approve。
    action: str
    # token_symbol：待操作的代币符号。
    token_symbol: str = Field(alias="tokenSymbol")
    # human_amount：按代币精度展示的十进制金额字符串，避免浮点数损失。
    human_amount: str = Field(alias="humanAmount")
    # target：待操作的目标地址或合约标识。
    target: str


# 数据职责：聊天统一响应；通过别名向前端输出驼峰字段。
class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # type：前端渲染类型：answer、ask，以及预留的 intent_confirm、tx_preview。
    type: Literal["answer", "ask", "intent_confirm", "tx_preview"]
    # content：前端本次展示的回复正文。
    content: str
    # data：响应附加数据，如步数、工具调用和引用。
    data: dict[str, Any] = Field(default_factory=dict)
    # task_id：关联业务任务标识；尚无任务时为空。
    task_id: str | None = Field(default=None, alias="taskId")
    # intent_preview：待用户确认的意图预览；当前回复不需要确认时为空。
    intent_preview: IntentPreview | None = Field(default=None, alias="intentPreview")
    # tx_intent：交易意图的结构化数据；当前只读流程通常为空。
    tx_intent: dict[str, Any] | None = Field(default=None, alias="txIntent")
    # risk_tags：当前响应附带的风险标签列表。
    risk_tags: list[str] = Field(default_factory=list, alias="riskTags")


# 数据职责：统一业务错误响应，区分错误码、文案、HTTP 状态及重试能力。
class ErrorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # error_code：稳定的业务错误码，供前端判断错误类别。
    error_code: str = Field(alias="errorCode")
    # error_msg：展示给用户的固定错误文案。
    error_msg: str = Field(alias="errorMsg")
    # retryable：客户端是否可以稍后重试当前请求。
    retryable: bool = False
    # http_status：该错误映射的 HTTP 状态码。
    http_status: int = Field(alias="httpStatus")

# 数据职责：一条可展示的历史消息及其工具轨迹和文档引用。
class HistoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # id：消息表的自增 ID，同时作为历史分页游标。
    id: int
    # role：消息角色：user 用户、assistant 助手、tool 工具、summary 摘要。
    role: str
    # content：该条历史消息的正文。
    content: str
    # tool_calls：本轮工具调用轨迹，包含名称、参数及执行结果；未调用时可为空。
    tool_calls: list[dict[str, Any]] | None = Field(default=None, alias="toolCalls")
    # citations：当前回答使用的文档证据快照，供引用展示和历史恢复。
    citations: list[dict[str, Any]] = Field(default_factory=list)
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: datetime = Field(alias="createdAt")


# 数据职责：聊天历史分页结果；游标以消息 ID 为依据。
class HistoryResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # items：本页历史消息，按前端需要的时间正序返回。
    items: list[HistoryItem]
    # has_more：本页之后是否还有更多记录。
    has_more: bool = Field(alias="hasMore")


# 数据职责：清空对话后的统计结果，长期档案和记忆不在本操作内。
class ClearResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # 摘要就是 conversation_message 里的一行，已计入 cleared_messages。
    # cleared_messages：本次清空删除的消息行数，包含存放在消息表中的摘要。
    cleared_messages: int = Field(alias="clearedMessages")
    # started_at：清空操作开始时间。
    started_at: datetime = Field(alias="startedAt")


# 数据职责：彻底删除用户会话相关数据后的统计结果。
class PurgeResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # deleted_messages：本次彻底删除的消息行数，包含摘要。
    deleted_messages: int = Field(alias="deletedMessages")
    # deleted_profile：本次是否删除了该用户的结构化档案。
    deleted_profile: bool = Field(alias="deletedProfile")
    # deleted_memories：本次删除的长期记忆记录数。
    deleted_memories: int = Field(alias="deletedMemories")
    # deleted_at：本次彻底删除操作的时间。
    deleted_at: datetime = Field(alias="deletedAt")
