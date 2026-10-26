"""意图抽取、摘要、记忆抽取与自由回答的业务入口。

模型访问（接入方式、超时、异常归一化）在 `llm.client`。这里只负责提示词、
输出契约，以及调用发出前的资源计量，不再自己构造任何请求。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from web3_agent.agent.prompts import INTENT_SYSTEM, MEMORY_SYSTEM, SUMMARY_SYSTEM
from web3_agent.common.exceptions import ModelBadOutput
from web3_agent.common.metering import Resource, reserve
from web3_agent.common.types import LlmOutput
from web3_agent.config.settings import Settings
from web3_agent.domain.memory.memory import MemoryDraftList
from web3_agent.domain.memory.schema import SummaryFields
from web3_agent.llm.client import ModelClient


# 数据职责：意图识别、摘要、记忆抽取与自由回答的业务入口。
class LlmAdapter:
    def __init__(self, settings: Settings, client: ModelClient | None = None) -> None:
        """绑定意图、摘要与记忆的输出契约。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
            client: 统一的模型访问入口；不传时按配置自建一个，多个业务方共用同一实例即可。
        """
        self._settings = settings
        # _client：唯一的模型访问入口，超时与异常边界都在它内部。
        self._client = client or ModelClient(settings)
        # _answer_model：用于自由文本回答且统一处理异常的模型包装器。
        self._answer_model = self._client.free_form
        # _intent_llm：按 LlmOutput 契约输出的意图模型。
        self._intent_llm = self._client.structured(LlmOutput, "意图抽取")
        # _summary_llm：按 SummaryFields 契约输出的摘要模型。
        self._summary_llm = self._client.structured(SummaryFields, "摘要")
        # _memory_llm：按 MemoryDraftList 契约输出的记忆模型。
        self._memory_llm = self._client.structured(MemoryDraftList, "记忆抽取")

    @property
    def client(self) -> ModelClient:
        """取得模型访问入口，供需要文本调用的其他领域模块直接使用。"""
        return self._client

    @property
    def model_name(self) -> str:
        """读取当前配置的聊天模型名称。"""
        return self._settings.model_name

    def bind_tools(self, tools: list[BaseTool]) -> Runnable:
        """给模型绑定允许使用的工具列表并返回可调用对象。

        Args:
            tools: 允许绑定给模型的工具实例列表。
        """
        return self._client.bind_tools(tools)

    async def check_model(self) -> dict:
        """确认模型可用：ollama 查已拉取列表与指纹，其余查模型列表。"""
        return await self._client.check_model()

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
        """按档案、记忆、摘要和近期历史组装上下文，调用结构化意图模型。

        Args:
            history: 按角色组织的近期对话列表。
            user_message: 本轮待处理的用户输入。
            summary: 已渲染的历史摘要文本。
            profile: 用户偏好对象或知识库嵌入配置，取决于当前模块。
            memories: 已渲染的用户长期背景文本。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。

        Returns:
            LlmOutput: 通过输出结构校验的意图、置信度和槽位。
        """
        messages: list[BaseMessage] = [SystemMessage(content=INTENT_SYSTEM)]
        # 顺序固定为 档案 → 背景 → 摘要：前两者是长期约束，摘要是一次性事实，越靠近当前输入越具体。
        if profile:
            messages.append(SystemMessage(content=profile))
        if memories:
            messages.append(SystemMessage(content=memories))
        if summary:
            messages.append(SystemMessage(content=summary))
        # 意图判断只消解「刚才那个」「它」这类指代，不需要全部历史：只看最近一小段，
        # 否则历史 + 系统提示会把 num_ctx 挤爆，被静默截断丢掉硬约束。
        for role, content in _tail_history(
            history, self._settings.intent_history_max_chars
        ):
            messages.append(_as_message(role, content))
        messages.append(_as_message("user", user_message))
        # 模型调用发出前计数，失败或超时仍属于已经发生的资源消耗。
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
        """增量压缩：只给旧摘要 + 新增区间，不给全部历史，否则 token 随会话线性增长。

        Args:
            previous: 上一份结构化历史摘要的 JSON 正文，用于增量更新。
            messages: 待压缩或抽取的 (角色, 正文) 对话列表。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。

        Returns:
            SummaryFields: 固定五类的结构化历史摘要。
        """
        prompt: list[BaseMessage] = [SystemMessage(content=SUMMARY_SYSTEM)]
        if previous:
            prompt.append(HumanMessage(content=f"已有摘要（在此基础上更新，保留仍有效的条目）：\n{previous}"))
        dialogue = "\n".join(f"{role}: {content}" for role, content in messages)
        prompt.append(HumanMessage(content=f"需要压缩的对话：\n{dialogue}"))
        # 模型调用发出前计数，失败或超时仍属于已经发生的资源消耗。
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
        """抽长期记忆。existing 只用来避免重复输出，模型不负责判断矛盾。

        Args:
            messages: 待压缩或抽取的 (角色, 正文) 对话列表。
            existing: 已持久化的记忆文本或常用代币集合。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。

        Returns:
            MemoryDraftList: 尚需业务清洗与去重的记忆候选。
        """
        prompt: list[BaseMessage] = [SystemMessage(content=MEMORY_SYSTEM)]
        if existing:
            prompt.append(
                HumanMessage(content="已有记忆（已覆盖的不要重复输出）：\n" + "\n".join(f"- {t}" for t in existing))
            )
        dialogue = "\n".join(f"{role}: {content}" for role, content in messages)
        prompt.append(HumanMessage(content=f"需要抽取的对话：\n{dialogue}"))
        # 模型调用发出前计数，失败或超时仍属于已经发生的资源消耗。
        reserve(Resource.MODEL)
        result = await self._memory_llm.ainvoke(prompt, config=config or None)
        if not isinstance(result, MemoryDraftList):
            raise ModelBadOutput("记忆抽取：结构化输出字段不合法")
        return result

    async def answer(
        self, messages: list[BaseMessage], *, config: RunnableConfig | None = None
    ) -> BaseMessage:
        # 模型调用发出前计数，失败或超时仍属于已经发生的资源消耗。
        """调用普通回答模型，调用前计入模型资源消耗。

        Args:
            messages: 按角色组织的模型输入消息列表。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。
        """
        reserve(Resource.MODEL)
        return await self._answer_model.ainvoke(messages, config=config or None)

    @property
    def answer_model(self) -> Runnable:
        """取得具备统一异常边界的回答模型，供图节点直接调用。"""
        return self._answer_model

    @property
    def chat_model(self) -> BaseChatModel:
        """取得底层聊天模型实例，供需要原始模型的调用方使用。"""
        return self._client.chat_model


def _as_message(role: str, content: str) -> BaseMessage:
    """把持久化角色和正文转换成 LangChain 人类或助手消息。

    Args:
        role: 消息角色，如 user、assistant 或 summary。
        content: 消息正文、文档正文或待写入文本。
    """
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _tail_history(
    history: list[tuple[str, str]], max_chars: int
) -> list[tuple[str, str]]:
    """取历史末尾不超过 max_chars 字符的一段，供意图抽取消解指代。

    max_chars 为 0 时原样返回全部历史。最新一条始终保留，其余从旧到新丢弃。

    Args:
        history: 按时间排列的 (角色, 正文) 消息列表。
        max_chars: 该段历史允许的最大字符预算。
    """
    if max_chars <= 0:
        return history
    kept: list[tuple[str, str]] = []
    used = 0
    for role, content in reversed(history):
        size = len(content or "")
        if kept and used + size > max_chars:
            break
        kept.append((role, content))
        used += size
    kept.reverse()
    return kept
