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
    """把统一错误信息组装为 JSON HTTP 响应，并按需设置 Retry-After。

    Args:
        code: 已登记的业务错误码。
        msg: 对外展示的固定错误文案。
        status: 对外 HTTP 状态码。
        retryable: 当前失败是否允许客户端重试。
        retry_after: 建议重试等待秒数；为空则不设置提示。
    """
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
    """注册业务错误、参数校验错误和未知异常的统一响应处理器。

    Args:
        app: 当前 FastAPI 应用实例。
    """
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        """把 AppError 的字段映射为规范错误响应。

        Args:
            exc: 需要转换、记录或判断是否重试的原始异常。
        """
        return _error(exc.code, exc.message, exc.http_status, exc.retryable, exc.retry_after)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        """把请求参数校验失败转换为统一参数错误，避免泄露内部校验细节。

        Args:
            exc: 需要转换、记录或判断是否重试的原始异常。
        """
        logger.info("validation error: %s", exc.errors())
        return _error(*error_tuple("E101"))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        """记录未处理异常，并返回统一内部错误响应。

        Args:
            exc: 需要转换、记录或判断是否重试的原始异常。
        """
        logger.exception("unhandled error")
        return _error("E900", "服务内部错误", 500)
