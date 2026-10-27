"""日志格式：每行都带 trace_id，出问题时能和审计表对上。

trace_id 存在 ContextVar 里，后台任务（摘要、记忆抽取）由 create_task 继承，
所以在日志里看到的仍是触发它的那一轮请求。
"""

from __future__ import annotations

import logging

from web3_agent.common.tracing import get_trace_id

# _FORMAT：模型响应结构异常类型集合，可进入结构化重试。
_FORMAT = "%(asctime)s %(levelname)-5s [%(trace_id)s] %(name)s: %(message)s"
# _DATE_FORMAT：日志时间字段的格式字符串。
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# uvicorn 自带 handler 且 propagate=False，不接管就还是它自己的格式。
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class HealthAccessFilter(logging.Filter):
    """省略成功的存活探针访问记录，保留失败请求和就绪检查。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            _, method, path, _, status = record.args
            if method in {"GET", "HEAD"} and str(path).partition("?")[0] == "/health":
                return not (isinstance(status, int) and 200 <= status < 300)
        return True


class TraceIdFilter(logging.Filter):
    """给没有 trace_id 的记录补 "-"，否则 formatter 会抛 KeyError 把日志本身弄坏。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """给日志记录补充当前请求的 trace_id，供统一格式化输出。

        Args:
            record: 待补充请求追踪字段的日志记录。
        """
        record.trace_id = get_trace_id() or "-"
        return True


def setup_logging(level: str) -> None:
    """配置日志级别、输出格式与请求追踪字段过滤器。

    Args:
        level: 日志级别名称。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(TraceIdFilter())
    handler.addFilter(HealthAccessFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
