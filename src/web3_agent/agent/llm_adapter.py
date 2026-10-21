"""ChatOllama 薄封装。

所有模型调用都从这里过一遍，目的是：
- 统一 num_ctx / keep_alive（不设 num_ctx 会静默截断上下文）；
- 统一走 LangChain，保证 LangSmith 埋点不断链；
- 意图抽取用 JSON Schema（format），不用 tool calling。
"""

from __future__ import annotations

import logging

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from web3_agent.agent.prompts import INTENT_SYSTEM, MEMORY_SYSTEM, SUMMARY_SYSTEM
from web3_agent.common.exceptions import ModelUnavailable
from web3_agent.common.types import LlmOutput
from web3_agent.config.settings import Settings
from web3_agent.domain.memory.memory import MemoryDraftList
from web3_agent.domain.memory.schema import SummaryFields

logger = logging.getLogger(__name__)


class LlmAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
        )
        self._intent_llm = self._llm.with_structured_output(LlmOutput, method="json_schema")
        self._summary_llm = self._llm.with_structured_output(SummaryFields, method="json_schema")
        self._memory_llm = self._llm.with_structured_output(MemoryDraftList, method="json_schema")

    @property
    def model_name(self) -> str:
        return self._settings.ollama_model

    def bind_tools(self, tools: list[BaseTool]) -> Runnable:
        return self._llm.bind_tools(tools)

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
            raise ModelUnavailable(f"模型未拉取：{self.model_name}")

        digest = hit.get("digest", "")
        expected = self._settings.ollama_model_digest
        if expected and digest != expected:
            raise ModelUnavailable(
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
        try:
            result = await self._intent_llm.ainvoke(messages, config=config or None)
        except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            raise ModelUnavailable(f"模型调用失败：{exc}") from exc
        if not isinstance(result, LlmOutput):
            raise ModelUnavailable("结构化输出解析失败")
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
        try:
            result = await self._summary_llm.ainvoke(prompt, config=config or None)
        except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            raise ModelUnavailable(f"摘要调用失败：{exc}") from exc
        if not isinstance(result, SummaryFields):
            raise ModelUnavailable("摘要结构化输出解析失败")
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
        try:
            result = await self._memory_llm.ainvoke(prompt, config=config or None)
        except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            raise ModelUnavailable(f"记忆抽取调用失败：{exc}") from exc
        if not isinstance(result, MemoryDraftList):
            raise ModelUnavailable("记忆抽取结构化输出解析失败")
        return result

    async def answer(self, messages: list[BaseMessage]) -> BaseMessage:
        return await self._llm.ainvoke(messages)

    @property
    def chat_model(self) -> BaseChatModel:
        return self._llm


def _as_message(role: str, content: str) -> BaseMessage:
    from langchain_core.messages import AIMessage, HumanMessage

    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)
