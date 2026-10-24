"""全局异常与错误码。错误文案固定，不由 LLM 生成。

ERROR_TABLE 照 PRD 11 章「错误码总表」原样落地，是错误码的唯一来源；
下面的三元组常量只是它的具名视图，避免各处硬编码字符串时写岔文案。
E507 / E508 是 PRD 表未覆盖的两处，在此补登（见注释）。
"""

from __future__ import annotations


# 数据职责：可直接映射为统一 HTTP 响应的业务异常。
class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        retryable: bool = False,
        retry_after: int | None = None,
    ) -> None:
        """保存固定错误码、用户文案、HTTP 状态和重试信息，交由全局异常处理器响应。

        Args:
            code: 已登记的业务错误码。
            message: 按当前函数用途传入的用户文本或已解析 SIWE 消息。
            http_status: 对外 HTTP 状态码。
            retryable: 当前失败是否允许客户端重试。
            retry_after: 建议重试等待秒数；为空则不设置提示。
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        # 秒数。只在此刻重试必然失败、过一会儿才可能成功的场合给（如上一轮还在跑）。
        self.retry_after = retry_after


class ToolSourceViolation(Exception):
    """LLM 试图调用 mutating 工具。属于内部越界，不是用户错误。"""


class BudgetExhausted(Exception):
    """本轮某类资源预算已用完。调用方按数据源降级，不当成服务故障。"""


# 数据职责：链上 RPC 不可用、协议不合法或预算不足的内部故障信号。
class RpcUnavailable(Exception):
    pass


class ModelUnavailable(Exception):
    """模型服务不可达或超时。对外统一 503，重试有意义。"""


class ModelConfigError(ModelUnavailable):
    """模型名、端点这类配置错误。重试不会变好，只是不需要单独一条兜底路径。"""


class ModelBadOutput(Exception):
    """模型返回了内容但结构不合法。这是唯一值得走一次格式修复的失败。"""


# code -> (用户可见固定文案, HTTP 状态码)
# ERROR_TABLE：全局错误码到固定文案和 HTTP 状态码的映射。
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
    "E304": ("工具执行失败，请稍后重试", 502),
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
    "E604": ("上传文件或请求体超过大小限制", 413),
    "E605": ("无知识库管理权限", 403),
    "E606": ("知识库上传服务未启用", 503),
    "E607": ("文档格式、元数据或审核信息不正确", 400),
    "E608": ("入库任务或文档不存在", 404),
    "E609": ("重复请求内容不一致，或当前状态不允许此操作", 409),
    "E611": ("知识库不存在或已停用", 404),
    "E612": ("知识库嵌入配置与当前服务不一致", 409),
    "E613": ("知识库切片数量超过关键词检索上限，请调整配置或拆分知识库", 409),
    "E610": ("知识库暂时不可用", 503),
    "E509": ("链与登录身份不一致", 403),
    # PRD 表未列：单轮预算（整轮时间 / 工具调用次数）耗尽。与 E501 的每分钟配额不是
    # 一回事，那个等一分钟就恢复，这个要把问题拆小或重问一次。
    "E510": ("本轮处理超出时间或资源上限，请把问题拆小一点再试", 504),
    # 幂等键：同一个 key 的请求还在跑。等这一轮收尾后拿同一个 key 重试就能拿到答案，
    # 所以是可重试的；换 key 等于再跑一轮，客户端不该这么做。
    "E511": ("同一请求正在处理中，请稍后重试", 409),
    # 幂等键被复用到另一条内容上。重试永远是同一个结果，只能换 key 重新发起。
    "E512": ("请求内容与上一次不一致，请重新发起", 409),
}


def error_tuple(code: str) -> tuple[str, str, int]:
    """取 (code, 固定文案, HTTP)。未登记的码直接抛，避免悄悄发出没有文案的错误。

    Args:
        code: 已登记的业务错误码。
    """
    try:
        message, http_status = ERROR_TABLE[code]
    except KeyError as exc:
        raise KeyError(f"未登记的错误码: {code}") from exc
    return code, message, http_status


def error_message(code: str) -> str:
    """按已登记错误码读取固定用户提示文案。

    Args:
        code: 已登记的业务错误码。
    """
    return ERROR_TABLE[code][0]


# E101_INVALID_PARAM：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E101_INVALID_PARAM = error_tuple("E101")
# E102_CHAIN_NOT_ALLOWED：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E102_CHAIN_NOT_ALLOWED = error_tuple("E102")
# E301_RPC_TIMEOUT：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E301_RPC_TIMEOUT = error_tuple("E301")
# E302_RPC_ERROR：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E302_RPC_ERROR = error_tuple("E302")
# E501_QUOTA_EXHAUSTED：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E501_QUOTA_EXHAUSTED = error_tuple("E501")
# E503_UNAUTHENTICATED：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E503_UNAUTHENTICATED = error_tuple("E503")
# E507_MODEL_UNAVAILABLE：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E507_MODEL_UNAVAILABLE = error_tuple("E507")
# E509_CHAIN_IDENTITY_MISMATCH：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E509_CHAIN_IDENTITY_MISMATCH = error_tuple("E509")
# E510_TURN_BUDGET：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E510_TURN_BUDGET = error_tuple("E510")
# E511_DUPLICATE_REQUEST：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E511_DUPLICATE_REQUEST = error_tuple("E511")
# E512_IDEMPOTENCY_CONFLICT：该业务错误对应的 (错误码, 固定文案, HTTP 状态码) 三元组。
E512_IDEMPOTENCY_CONFLICT = error_tuple("E512")
