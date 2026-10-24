"""SIWE 鉴权接口的 HTTP 契约（PRD 7.4）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# 数据职责：申请登录挑战时提交的钱包与网络信息。
class NonceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 地址必填：nonce 直接绑定到该地址，验签时 recovered 必须与之一致
    # address：钱包地址；登录接口用它绑定签名身份，查询接口用它指定账户。
    address: str = Field(min_length=42, max_length=42)
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int = Field(alias="chainId")


# 数据职责：待签名的完整登录消息及一次性挑战有效期。
class NonceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # nonce：SIWE 一次性随机挑战值，验签成功后核销以防重放。
    nonce: str
    # expires_in：凭据剩余有效时长，单位为秒。
    expires_in: int = Field(alias="expiresIn")
    # 后端生成的完整消息，前端直接让钱包签，避免两端格式不一致导致验签失败
    # message：待签名的 SIWE 完整文本或业务提示文本，含义由所属模型决定。
    message: str


# 数据职责：钱包签名验证请求，需提交完整原文和签名。
class VerifyRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # message：待签名的 SIWE 完整文本或业务提示文本，含义由所属模型决定。
    message: str = Field(min_length=1, max_length=4000)
    # signature：钱包对完整 SIWE 文本生成的 personal_sign 签名。
    signature: str = Field(min_length=1, max_length=200)


# 数据职责：验签成功后返回的登录令牌与钱包展示信息。
class VerifyResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # token：登录成功后签发的 JWT，后续请求放在 Authorization Bearer 中。
    token: str
    # siwe_address：已通过 SIWE 验证的钱包地址。
    siwe_address: str = Field(alias="siweAddress")
    # siwe_address_short：钱包地址的缩略展示形式。
    siwe_address_short: str = Field(alias="siweAddressShort")
    # expires_in：凭据剩余有效时长，单位为秒。
    expires_in: int = Field(alias="expiresIn")
