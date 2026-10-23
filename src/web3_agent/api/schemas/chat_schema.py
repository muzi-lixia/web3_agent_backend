"""聊天接口的 HTTP 契约。字段名与 PRD 7.1 一致，与领域模型分开维护。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WalletRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    address: str
    chain_id: int = Field(alias="chainId")


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    wallet: WalletRef | None = None
    user_message: str = Field(alias="userMessage", min_length=1, max_length=4000)
    knowledge_base_id: str = Field(default="default",alias="knowledgeBaseId",pattern=r"^[A-Za-z0-9_.:-]{1,64}$")


class IntentPreview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    action: str
    token_symbol: str = Field(alias="tokenSymbol")
    human_amount: str = Field(alias="humanAmount")
    target: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["answer", "ask", "intent_confirm", "tx_preview"]
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = Field(default=None, alias="taskId")
    intent_preview: IntentPreview | None = Field(default=None, alias="intentPreview")
    tx_intent: dict[str, Any] | None = Field(default=None, alias="txIntent")
    risk_tags: list[str] = Field(default_factory=list, alias="riskTags")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    error_code: str = Field(alias="errorCode")
    error_msg: str = Field(alias="errorMsg")
    retryable: bool = False
    http_status: int = Field(alias="httpStatus")

class HistoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = Field(default=None, alias="toolCalls")
    citations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(alias="createdAt")


class HistoryResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    items: list[HistoryItem]
    has_more: bool = Field(alias="hasMore")


class ClearResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    cleared_messages: int = Field(alias="clearedMessages")
    cleared_summary: bool = Field(alias="clearedSummary")
    started_at: datetime = Field(alias="startedAt")


class PurgeResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    deleted_messages: int = Field(alias="deletedMessages")
    deleted_summaries: int = Field(alias="deletedSummaries")
    deleted_profile: bool = Field(alias="deletedProfile")
    deleted_memories: int = Field(alias="deletedMemories")
    deleted_at: datetime = Field(alias="deletedAt")
