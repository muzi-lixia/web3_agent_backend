"""trace_id 注入。生成与透传都在这里，下游只读上下文。"""

from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from web3_agent.common.tracing import new_trace_id, set_trace_id

# _TRACE_ID_RE：客户端提供的请求追踪 ID 的允许字符及长度规则。
_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # 透传的 id 会被写进日志和响应头，先按白名单过一遍：不合规就当没传。
        """为 HTTP 请求设置追踪上下文，并把追踪 ID 附到响应头。

        Args:
            request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
            call_next: Starlette 提供的下一层请求处理函数，调用后获得 HTTP 响应。
        """
        incoming = request.headers.get("X-Trace-Id", "")
        trace_id = incoming if _TRACE_ID_RE.match(incoming) else new_trace_id()
        set_trace_id(trace_id)
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
