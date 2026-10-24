"""trace_id 注入。生成与透传都在这里，下游只读上下文。"""

from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from web3_agent.common.tracing import new_trace_id, set_trace_id

_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # 透传的 id 会被写进日志和响应头，先按白名单过一遍：不合规就当没传。
        incoming = request.headers.get("X-Trace-Id", "")
        trace_id = incoming if _TRACE_ID_RE.match(incoming) else new_trace_id()
        set_trace_id(trace_id)
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
