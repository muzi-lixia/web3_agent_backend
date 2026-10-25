"""写链路三个接口的 HTTP 契约：意图确认、任务状态、交易哈希上报。

字段名与 PRD 7.2 / 6.4 / 7.5 一致。PRD 请求体带 `sessionId`，本项目单线性会话
没有第二个分区键，因此不收该字段（与 chat 接口的处理一致）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# 数据职责：用户对意图回显的确认或拒绝。
class IntentConfirmRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # task_id：写操作任务标识，由 chat 响应下发。
    task_id: str = Field(alias="taskId", min_length=1, max_length=36)
    # accepted：用户是否接受本次写操作。
    accepted: bool


# 数据职责：确认提交结果；拒绝也是一种正常结果，不按错误返回。
class IntentConfirmResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # success：请求是否被受理。
    success: bool = True
    # task_id：本次确认对应的任务标识。
    task_id: str = Field(alias="taskId")
    # status：受理后的业务结论：接受或拒绝。
    status: Literal["ACCEPTED", "REJECTED"]
    # error_code：受理失败时的稳定错误码；正常受理为空。
    error_code: str | None = Field(default=None, alias="errorCode")
    # error_msg：受理失败时的固定文案；正常受理为空。
    error_msg: str | None = Field(default=None, alias="errorMsg")


# 数据职责：交易哈希上报请求；状态说明只进审计，不驱动状态流转。
class TxReportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # task_id：写操作任务标识。
    task_id: str = Field(alias="taskId", min_length=1, max_length=36)
    # tx_hash：钱包广播后的交易哈希。
    tx_hash: str = Field(alias="txHash", pattern=r"^0x[0-9a-fA-F]{64}$")
    # status：前端观察到的交易状态，仅作审计备注。
    status: Literal["broadcast", "confirmed", "failed", "replaced"]


# 数据职责：哈希上报的固定应答，明确状态以链上回执为准。
class TxReportResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # success：事件是否已记录。
    success: bool = True
    # task_id：本次上报对应的任务标识。
    task_id: str = Field(alias="taskId")
    # msg：固定说明文案，避免前端把上报当成状态来源。
    msg: str = "event logged, status sourced from backend RPC receipt"


# 数据职责：任务状态查询结果；txIntent 只在等待签名时出现。
class WorkflowStatusResp(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # task_id：写操作任务标识。
    task_id: str = Field(alias="taskId")
    # state：生命周期状态，取值见 WorkflowState。
    state: str
    # expire_type：过期原因：NONE 或两个 EXPIRED_* 之一。
    expire_type: Literal["NONE", "EXPIRED_UNSIGNED", "EXPIRED_UNVERIFIED"] = Field(
        alias="expireType"
    )
    # siwe_address_short：任务归属钱包的脱敏摘要，完整地址不对外输出。
    siwe_address_short: str = Field(alias="siweAddressShort")
    # tx_intent：待签名交易参数；无待签交易时为空。
    tx_intent: dict[str, Any] | None = Field(default=None, alias="txIntent")
    # tx_hash：已上报的广播哈希；未广播时为空。
    tx_hash: str | None = Field(default=None, alias="txHash")
    # error_code：失败、过期或本次回执核对失败时的稳定错误码。
    error_code: str | None = Field(default=None, alias="errorCode")
    # error_msg：与错误码对应的固定文案。
    error_msg: str | None = Field(default=None, alias="errorMsg")
    # expires_at：任务过期时间戳，单位为秒。
    expires_at: int = Field(alias="expiresAt")
