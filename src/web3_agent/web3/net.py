"""只读外部调用的一次重试：判据与次数放在一处，RPC 与区块浏览器共用。

只读查询不该因为一次连接重置就报「查不到」；但也不能在请求路径里退避重试 ——
一轮对话有整轮时间上限，等待只会把预算烧在一个数据源上。
"""

from __future__ import annotations

import httpx

# 两次：一次失败就放弃太脆，再多就把整轮时间花在一个数据源上。
MAX_ATTEMPTS = 2


def is_transient(exc: httpx.HTTPError) -> bool:
    """同一请求再发一次可能成功的错误：传输层失败与 5xx。

    4xx 是确定性失败（方法不存在、参数不对、Key 无效），重试只是把等待时间翻倍。
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)
