"""ChatOllama 薄封装。

所有模型调用都从这里过一遍，目的是：
- 统一 num_ctx / keep_alive / 超时（不设 num_ctx 会静默截断上下文）；
- 统一走 LangChain，保证 LangSmith 埋点不断链；
- 把各家 SDK 的异常翻成项目自己的三类，调用点不再各写一份 except。
"""

from __future__ import annotations

import logging

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from web3_agent.agent.prompts import INTENT_SYSTEM, MEMORY_SYSTEM, SUMMARY_SYSTEM
from web3_agent.common.exceptions import ModelBadOutput, ModelConfigError, ModelUnavailable
from web3_agent.common.types import LlmOutput
from web3_agent.config.settings import Settings
from web3_agent.domain.memory.memory import MemoryDraftList
from web3_agent.domain.memory.schema import SummaryFields
from web3_agent.common.metering import Resource, reserve

logger = logging.getLogger(__name__)

# 传输层故障：连不上、超时、端口不通。等一会再试有意义。
# ConnectionError / TimeoutError 都是 OSError 子类，写进元组是重复的。
_TRANSPORT = (httpx.HTTPError, OSError)
# 拿到了响应但字段不对：只有这一类值得走 JsonGuard 的格式修复。
_FORMAT = (ValueError, KeyError, TypeError, AttributeError)


def _normalize(exc: BaseException, stage: str) -> Exception:
    """把 SDK 异常翻成三类，本项目其他位置只认这三类。"""
    if isinstance(exc, (ModelUnavailable, ModelBadOutput)):
        return exc
    status = getattr(exc, "status_code", None)
    if type(exc).__name__ == "ResponseError" and status in (400, 404):
        # 模型名不对或请求被服务端拒绝：重试和降级都救不回来。
        return ModelConfigError(f"{stage}：模型拒绝请求（{status}）")
    if isinstance(exc, _TRANSPORT):
        return ModelUnavailable(f"{stage}：{type(exc).__name__}")
    if isinstance(exc, _FORMAT):
        return ModelBadOutput(f"{stage}：响应格式不合法")
    # 未知异常也不许落成通用 500：对外按模型不可用处理，细节留在日志里。
    logger.warning("unclassified model error at %s: %r", stage, exc)
    return ModelUnavailable(f"{stage}：{type(exc).__name__}")


class _GuardedModel:
    """在调用边界归一化异常。图里直接 ainvoke 也不会漏出 SDK 原始异常。"""

    def __init__(self, inner: Runnable, stage: str) -> None:
        self._inner = inner
        self._stage = stage

    def bind_tools(self, tools: list[BaseTool]) -> "_GuardedModel":
        return _GuardedModel(self._inner.bind_tools(tools), self._stage)

    def invoke(self, messages, config: RunnableConfig | None = None, **kwargs):
        try:
            return self._inner.invoke(messages, config, **kwargs)
        except Exception as exc:  # noqa: BLE001 所有异常都要归一到三类再往外走
            raise _normalize(exc, self._stage) from exc

    async def ainvoke(self, messages, config: RunnableConfig | None = None, **kwargs):
        try:
            return await self._inner.ainvoke(messages, config, **kwargs)
        except Exception as exc:  # noqa: BLE001 同上；取消是 BaseException，不受影响
            raise _normalize(exc, self._stage) from exc


class LlmAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            # 不传就是 httpx 默认值：模型排队时请求一直挂着，先把整轮预算吃光。
            client_kwargs={"timeout": settings.ollama_timeout_seconds},
        )
        self._answer_model = _GuardedModel(self._llm, "回答")
        self._intent_llm = _GuardedModel(
            self._llm.with_structured_output(LlmOutput, method="json_schema"), "意图抽取"
        )
        self._summary_llm = _GuardedModel(
            self._llm.with_structured_output(SummaryFields, method="json_schema"), "摘要"
        )
        self._memory_llm = _GuardedModel(
            self._llm.with_structured_output(MemoryDraftList, method="json_schema"), "记忆抽取"
        )

    @property
    def model_name(self) -> str:
        return self._settings.ollama_model

    def bind_tools(self, tools: list[BaseTool]) -> Runnable:
        return self._answer_model.bind_tools(tools)

    async def check_model(self) -> dict:
        """确认模型真的加载得起来，而不只是 11434 端口活着。"""
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/tags"
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
        expected = self._settings.ollama_model_digest
        if expected and digest != expected:
            raise ModelConfigError(
                f"模型 digest 不一致：期望 {expected[:12]}，实际 {digest[:12]}"
            )
        return {"model": self.model_name, "digest": digest}

    async def extract_intent(
        self,
        history: list[tuple[str, str]],
        user_message: str,
        *,
        summary: str = "",
        profile: str = "",
        memories: str = "",
        config: RunnableConfig | None = None,
    ) -> LlmOutput:
        messages: list[BaseMessage] = [SystemMessage(content=INTENT_SYSTEM)]
        # 顺序固定为 档案 → 背景 → 摘要：前两者是长期约束，摘要是一次性事实，越靠近当前输入越具体。
        if profile:
            messages.append(SystemMessage(content=profile))
        if memories:
            messages.append(SystemMessage(content=memories))
        if summary:
            messages.append(SystemMessage(content=summary))
        for role, content in history:
            messages.append(_as_message(role, content))
        messages.append(_as_message("user", user_message))
        reserve(Resource.MODEL)
        result = await self._intent_llm.ainvoke(messages, config=config or None)
        if not isinstance(result, LlmOutput):
            # 模型给了内容但字段不合法：可修复，不是服务故障。
            raise ModelBadOutput("意图抽取：结构化输出字段不合法")
        return result

    async def summarize(
        self,
        previous: str,
        messages: list[tuple[str, str]],
        *,
        config: RunnableConfig | None = None,
    ) -> SummaryFields:
        """增量压缩：只给旧摘要 + 新增区间，不给全部历史，否则 token 随会话线性增长。"""
        prompt: list[BaseMessage] = [SystemMessage(content=SUMMARY_SYSTEM)]
        if previous:
            prompt.append(HumanMessage(content=f"已有摘要（在此基础上更新，保留仍有效的条目）：\n{previous}"))
        dialogue = "\n".join(f"{role}: {content}" for role, content in messages)
        prompt.append(HumanMessage(content=f"需要压缩的对话：\n{dialogue}"))
        reserve(Resource.MODEL)
        result = await self._summary_llm.ainvoke(prompt, config=config or None)
        if not isinstance(result, SummaryFields):
            raise ModelBadOutput("摘要：结构化输出字段不合法")
        return result

    async def extract_memories(
        self,
        messages: list[tuple[str, str]],
        existing: list[str],
        *,
        config: RunnableConfig | None = None,
    ) -> MemoryDraftList:
        """抽长期记忆。existing 只用来避免重复输出，模型不负责判断矛盾。"""
        prompt: list[BaseMessage] = [SystemMessage(content=MEMORY_SYSTEM)]
        if existing:
            prompt.append(
                HumanMessage(content="已有记忆（已覆盖的不要重复输出）：\n" + "\n".join(f"- {t}" for t in existing))
            )
        dialogue = "\n".join(f"{role}: {content}" for role, content in messages)
        prompt.append(HumanMessage(content=f"需要抽取的对话：\n{dialogue}"))
        reserve(Resource.MODEL)
        result = await self._memory_llm.ainvoke(prompt, config=config or None)
        if not isinstance(result, MemoryDraftList):
            raise ModelBadOutput("记忆抽取：结构化输出字段不合法")
        return result

    async def answer(
        self, messages: list[BaseMessage], *, config: RunnableConfig | None = None
    ) -> BaseMessage:
        reserve(Resource.MODEL)
        return await self._answer_model.ainvoke(messages, config=config or None)

    @property
    def answer_model(self) -> Runnable:
        return self._answer_model

    @property
    def chat_model(self) -> BaseChatModel:
        return self._llm


def _as_message(role: str, content: str) -> BaseMessage:
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)
