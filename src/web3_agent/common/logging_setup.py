"""日志格式：每行都带 trace_id，出问题时能和审计表对上。

trace_id 存在 ContextVar 里，后台任务（摘要、记忆抽取）由 create_task 继承，
所以在日志里看到的仍是触发它的那一轮请求。
"""

from __future__ import annotations

import logging

from web3_agent.common.tracing import get_trace_id

_FORMAT = "%(asctime)s %(levelname)-5s [%(trace_id)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# uvicorn 自带 handler 且 propagate=False，不接管就还是它自己的格式。
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class TraceIdFilter(logging.Filter):
    """给没有 trace_id 的记录补 "-"，否则 formatter 会抛 KeyError 把日志本身弄坏。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "-"
        return True


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(TraceIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
