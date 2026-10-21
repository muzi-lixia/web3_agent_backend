"""全局异常与错误码。错误文案固定，不由 LLM 生成。

ERROR_TABLE 照 PRD 11 章「错误码总表」原样落地，是错误码的唯一来源；
下面的三元组常量只是它的具名视图，避免各处硬编码字符串时写岔文案。
E507 / E508 是 PRD 表未覆盖的两处，在此补登（见注释）。
"""

from __future__ import annotations


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable


class ToolSourceViolation(Exception):
    """LLM 试图调用 mutating 工具。属于内部越界，不是用户错误。"""


class RpcUnavailable(Exception):
    pass


class ModelUnavailable(Exception):
    pass


# code -> (用户可见固定文案, HTTP 状态码)
ERROR_TABLE: dict[str, tuple[str, int]] = {
    "E101": ("请求参数格式不正确", 400),
    "E102": ("暂不支持该网络", 400),
    "E103": ("钱包地址格式不正确", 400),
    "E104": ("金额格式不正确", 400),
    "E105": ("查询条数超出上限", 400),
    "E201": ("暂不支持该代币", 400),
    "E202": ("暂不支持该质押合约", 400),
    "E203": ("该质押合约暂不支持质押", 400),
    "E204": ("授权目标校验失败", 400),
    "E205": ("该代币暂不支持", 400),
    "E301": ("链上服务暂时不可用，请稍后重试", 503),
    "E302": ("交易记录暂时无法获取", 502),
    "E401": ("交易被合约拒绝", 422),  # 实际文案由 RevertDecoder 补业务原因
    "E402": ("该质押合约当前不可用", 409),
    "E403": ("交易模拟未通过，链上状态可能已变化", 409),
    "E404": ("该代币暂不支持写操作", 422),
    "E501": ("请求过于频繁，请稍后再试", 429),
    "E502": ("单笔金额超过限额", 403),
    "E503": ("请先连接钱包并登录", 401),
    "E504": ("任务不存在", 404),
    "E505": ("无权访问该任务", 403),
    "E506": ("任务已过期，请重新发起", 410),
    # PRD 表未列：模型服务不可用。原先占用 E501，与「配额耗尽」冲突，改挂 E507。
    "E507": ("模型服务暂时不可用", 503),
    # PRD 表未列：PRD 8.7 要求 fromAddress 必须等于绑定的 siweAddress，但没给错误码。
    "E508": ("钱包地址与登录身份不一致", 403),
}


def error_tuple(code: str) -> tuple[str, str, int]:
    """取 (code, 固定文案, HTTP)。未登记的码直接抛，避免悄悄发出没有文案的错误。"""
    try:
        message, http_status = ERROR_TABLE[code]
    except KeyError as exc:
        raise KeyError(f"未登记的错误码: {code}") from exc
    return code, message, http_status


def error_message(code: str) -> str:
    return ERROR_TABLE[code][0]


E101_INVALID_PARAM = error_tuple("E101")
E102_CHAIN_NOT_ALLOWED = error_tuple("E102")
E301_RPC_TIMEOUT = error_tuple("E301")
E302_RPC_ERROR = error_tuple("E302")
E507_MODEL_UNAVAILABLE = error_tuple("E507")
