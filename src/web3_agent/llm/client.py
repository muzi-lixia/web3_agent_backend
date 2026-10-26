"""模型调用的唯一出口。

接入方式、超时与异常归一化只在这里定义。业务提示词与输出契约属于各业务层，
不进入本模块 —— 这样任何一处模型调用都走同一套传输与追踪，不存在第二份实现。
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from web3_agent.common.exceptions import (
    ModelBadOutput,
    ModelConfigError,
    ModelUnavailable,
)
from web3_agent.common.metering import Resource, reserve
from web3_agent.config.settings import Settings

logger = logging.getLogger(__name__)

# 传输层故障：连不上、超时、端口不通。等一会再试有意义。
# ConnectionError / TimeoutError 都是 OSError 子类，写进元组是重复的。
# SDK 的连接异常不继承 httpx.HTTPError，只按 httpx 判断会落进「未归类」。
# _TRANSPORT：模型 SDK 的传输层故障类型集合，归一为模型不可用。
_TRANSPORT = (httpx.HTTPError, OSError, openai.APIConnectionError)
# 拿到了响应但字段不对：只有这一类值得走 JsonGuard 的格式修复。
# _FORMAT：模型响应结构异常类型集合，可进入结构化重试。
_FORMAT = (ValueError, KeyError, TypeError, AttributeError)


def normalize_model_error(exc: BaseException, stage: str) -> Exception:
    """把 SDK 异常翻成三类，本项目其他位置只认这三类。

    Args:
        exc: 需要转换、记录或判断是否重试的原始异常。
        stage: 执行阶段名称，用于进度或错误定位。
    """
    if isinstance(exc, (ModelUnavailable, ModelBadOutput)):
        return exc
    status = getattr(exc, "status_code", None)
    if status in (400, 401, 403, 404):
        # 请求本身或凭据被拒：重试和降级都救不回来。
        return ModelConfigError(f"{stage}：模型拒绝请求（{status}）")
    if isinstance(exc, _TRANSPORT):
        return ModelUnavailable(f"{stage}：{type(exc).__name__}")
    if isinstance(exc, _FORMAT):
        return ModelBadOutput(f"{stage}：响应格式不合法")
    # 未知异常也不许落成通用 500：对外按模型不可用处理，细节留在日志里。
    logger.warning("unclassified model error at %s: %r", stage, exc)
    return ModelUnavailable(f"{stage}：{type(exc).__name__}")


def _build_chat_model(settings: Settings) -> BaseChatModel:
    """按配置创建聊天模型；两个分支只差参数，不差调用契约。

    Args:
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    if settings.model_provider == "ollama":
        return ChatOllama(
            model=settings.model_name,
            base_url=settings.model_base_url,
            temperature=settings.model_temperature,
            num_ctx=settings.model_context_tokens,
            keep_alive=settings.model_keep_alive,
            # 不传就是 httpx 默认值：模型排队时请求一直挂着，先把整轮预算吃光。
            client_kwargs={"timeout": settings.model_timeout_seconds},
        )
    if not settings.model_api_key:
        raise ModelConfigError("聊天模型缺少 MODEL_API_KEY")
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
        temperature=settings.model_temperature,
        timeout=settings.model_timeout_seconds,
        # SDK 默认对 5xx 重试两次：一次调用会被记成一次，实际却发了三次请求。
        # 重试由上层按阶段预算决定，这里不开隐形重试。
        max_retries=0,
        extra_body=settings.model_extra_body or None,
    )


class GuardedModel:
    """在调用边界归一化异常。图里直接 ainvoke 也不会漏出 SDK 原始异常。"""

    def __init__(self, inner: Runnable, stage: str) -> None:
        """初始化异常边界包装。

        Args:
            inner: 需要包装异常边界的原始模型 Runnable。
            stage: 执行阶段名称，用于进度或错误定位。
        """
        self._inner = inner
        self._stage = stage

    def bind_tools(self, tools: list[BaseTool]) -> GuardedModel:
        """给模型绑定允许使用的工具列表并返回可调用对象。

        Args:
            tools: 允许绑定给模型的工具实例列表。
        """
        return GuardedModel(self._inner.bind_tools(tools), self._stage)

    def invoke(self, messages, config: RunnableConfig | None = None, **kwargs):
        """同步调用被包装模型，并把底层异常归一化。

        Args:
            messages: 按角色组织的模型输入消息列表。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。
            **kwargs: 透传给底层调用的其余关键字参数。
        """
        try:
            return self._inner.invoke(messages, config, **kwargs)
        except Exception as exc:
            raise normalize_model_error(exc, self._stage) from exc

    async def ainvoke(self, messages, config: RunnableConfig | None = None, **kwargs):
        """异步调用被包装模型，并把底层异常归一化；取消信号继续向上传播。

        Args:
            messages: 按角色组织的模型输入消息列表。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。
            **kwargs: 透传给底层调用的其余关键字参数。
        """
        try:
            return await self._inner.ainvoke(messages, config, **kwargs)
        except Exception as exc:
            raise normalize_model_error(exc, self._stage) from exc


# 数据职责：持有一个聊天模型，向业务层提供结构化、工具绑定与自由文本三种调用形态。
class ModelClient:
    def __init__(self, settings: Settings) -> None:
        """按配置创建聊天模型，并建立默认的异常边界。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self._settings = settings
        # _llm：底层聊天模型，接入方式由配置决定。
        self._llm = _build_chat_model(settings)
        # _chat：自由文本调用的异常边界包装。
        self._chat = GuardedModel(self._llm, "模型调用")

    @property
    def settings(self) -> Settings:
        """取得构造本客户端时使用的配置。"""
        return self._settings

    @property
    def model_name(self) -> str:
        """读取当前配置的聊天模型名称。"""
        return self._settings.model_name

    @property
    def chat_model(self) -> BaseChatModel:
        """取得底层聊天模型实例，供需要绑定结构化契约的调用方使用。"""
        return self._llm

    @property
    def free_form(self) -> GuardedModel:
        """取得自由文本回答模型，调用点只拿到异常已归一的包装。"""
        return self._chat

    def structured(self, schema, stage: str) -> GuardedModel:
        """按给定契约创建结构化输出模型。

        Args:
            schema: 期望模型输出的结构化契约。
            stage: 执行阶段名称，用于进度或错误定位。
        """
        return GuardedModel(
            self._llm.with_structured_output(schema, method=self._settings.structured_mode),
            stage,
        )

    def bind_tools(self, tools: list[BaseTool]) -> GuardedModel:
        """给自由文本模型绑定允许使用的工具列表。

        Args:
            tools: 允许绑定给模型的工具实例列表。
        """
        return self._chat.bind_tools(tools)

    async def chat(
        self,
        messages,
        *,
        timeout: float | None = None,
        json_mode: bool = False,
        config: RunnableConfig | None = None,
    ) -> str:
        """一次自由文本调用，返回模型输出正文。

        timeout 是调用点自己的阶段预算，与客户端默认超时无关：一次三秒的短调用
        不该继承整轮回答的尺度。json_mode 只要求服务端输出 JSON，不校验结构。

        Args:
            messages: 按角色组织的模型输入消息列表。
            timeout: 本次调用的超时秒数；为空表示只用客户端默认超时。
            json_mode: 是否要求服务端以 JSON 形式输出。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。
        """
        # 模型调用发出前计数，失败或超时仍属于已经发生的资源消耗。
        reserve(Resource.MODEL)
        runnable = GuardedModel(self._json_runnable(), "模型调用") if json_mode else self._chat
        try:
            async with asyncio.timeout(timeout):
                result = await runnable.ainvoke(messages, config=config or None)
        except TimeoutError as exc:
            # 阶段预算由本层兑现，超时也要落成三类模型错误之一。
            raise ModelUnavailable("模型调用：超出阶段预算") from exc
        return str(result.content)

    def _json_runnable(self) -> Runnable:
        """按接入方式给模型加上 JSON 输出约束。"""
        if self._settings.model_provider == "ollama":
            return self._llm.bind(format="json")
        return self._llm.bind(response_format={"type": "json_object"})

    async def check_model(self) -> dict:
        """确认模型可用：ollama 查已拉取列表与指纹，其余查模型列表。"""
        if self._settings.model_provider == "ollama":
            return await self._check_ollama()
        return await self._check_remote()

    async def _check_ollama(self) -> dict:
        url = f"{self._settings.model_base_url.rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                models = resp.json().get("models", [])
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"Ollama 不可达：{exc}") from exc

        hit = next((m for m in models if m.get("name") == self.model_name), None)
        if hit is None:
            raise ModelConfigError(f"模型未拉取：{self.model_name}")

        digest = hit.get("digest", "")
        expected = self._settings.model_digest
        if expected and digest != expected:
            raise ModelConfigError(
                f"模型 digest 不一致：期望 {expected[:12]}，实际 {digest[:12]}"
            )
        return {"model": self.model_name, "digest": digest}

    async def _check_remote(self) -> dict:
        url = f"{self._settings.model_base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self._settings.model_api_key}"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"模型服务不可达：{exc}") from exc
        if resp.status_code == 404:
            # 部分兼容实现没有模型列表接口，此时只能靠一次真实调用验证。
            return {"model": self.model_name, "verified": False}
        if resp.status_code >= 400:
            raise ModelConfigError(f"模型服务拒绝查询（{resp.status_code}）")
        available = [item.get("id") for item in resp.json().get("data", [])]
        if self.model_name not in available:
            raise ModelConfigError(f"模型不可用：{self.model_name}")
        return {"model": self.model_name, "verified": True}
