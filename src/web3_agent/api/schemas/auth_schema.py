"""SIWE 鉴权接口的 HTTP 契约（PRD 7.4）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NonceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 地址必填：nonce 直接绑定到该地址，验签时 recovered 必须与之一致
    address: str = Field(min_length=42, max_length=42)
    chain_id: int = Field(alias="chainId")


class NonceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    nonce: str
    expires_in: int = Field(alias="expiresIn")
    # 后端生成的完整消息，前端直接让钱包签，避免两端格式不一致导致验签失败
    message: str


class VerifyRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(min_length=1, max_length=4000)
    signature: str = Field(min_length=1, max_length=200)


class VerifyResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    token: str
    siwe_address: str = Field(alias="siweAddress")
    siwe_address_short: str = Field(alias="siweAddressShort")
    expires_in: int = Field(alias="expiresIn")
