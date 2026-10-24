"""全局异常映射。对外只暴露固定文案与错误码，不返回堆栈。"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from web3_agent.common.exceptions import AppError, error_tuple
from web3_agent.common.tracing import get_trace_id

logger = logging.getLogger(__name__)


def _error(
    code: str,
    msg: str,
    status: int,
    retryable: bool = False,
    retry_after: int | None = None,
) -> JSONResponse:
    headers = {"X-Trace-Id": get_trace_id()}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=status,
        content={
            "errorCode": code,
            "errorMsg": msg,
            "retryable": retryable,
            "httpStatus": status,
        },
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error(exc.code, exc.message, exc.http_status, exc.retryable, exc.retry_after)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("validation error: %s", exc.errors())
        return _error(*error_tuple("E101"))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error("E900", "服务内部错误", 500)
