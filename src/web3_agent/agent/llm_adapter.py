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
# _TRANSPORT：模型 SDK 的传输层故障类型集合，归一为模型不可用。
_TRANSPORT = (httpx.HTTPError, OSError)
# 拿到了响应但字段不对：只有这一类值得走 JsonGuard 的格式修复。
# _FORMAT：模型响应结构异常类型集合，可进入结构化重试。
_FORMAT = (ValueError, KeyError, TypeError, AttributeError)


def _normalize(exc: BaseException, stage: str) -> Exception:
    """把 SDK 异常翻成三类，本项目其他位置只认这三类。

    Args:
        exc: 需要转换、记录或判断是否重试的原始异常。
        stage: 执行阶段名称，用于进度或错误定位。
    """
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
        """初始化 _GuardedModel 所需的依赖和运行状态。

        Args:
            inner: 需要包装异常边界的原始模型 Runnable。
            stage: 执行阶段名称，用于进度或错误定位。
        """
        self._inner = inner
        self._stage = stage

    def bind_tools(self, tools: list[BaseTool]) -> "_GuardedModel":
        """给模型绑定允许使用的工具列表并返回可调用对象。

        Args:
            tools: 允许绑定给模型的工具实例列表。
        """
        return _GuardedModel(self._inner.bind_tools(tools), self._stage)

    def invoke(self, messages, config: RunnableConfig | None = None, **kwargs):
        """同步调用被包装模型，并把底层异常归一化。

        Args:
            messages: 按角色组织的模型输入消息列表。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。
            **kwargs: 透传给底层调用的其余关键字参数。
        """
        try:
            return self._inner.invoke(messages, config, **kwargs)
        except Exception as exc:  # noqa: BLE001 所有异常都要归一到三类再往外走
            raise _normalize(exc, self._stage) from exc

    async def ainvoke(self, messages, config: RunnableConfig | None = None, **kwargs):
        """异步调用被包装模型，并把底层异常归一化；取消信号继续向上传播。

        Args:
            messages: 按角色组织的模型输入消息列表。
            config: LangChain 调用配置，包含追踪名称、运行 ID 等。
            **kwargs: 透传给底层调用的其余关键字参数。
        """
        try:
            return await self._inner.ainvoke(messages, config, **kwargs)
        except Exception as exc:  # noqa: BLE001 同上；取消是 BaseException，不受影响
            raise _normalize(exc, self._stage) from exc


# 数据职责：聊天、意图识别、摘要与记忆抽取的统一模型入口。
class LlmAdapter:
    def __init__(self, settings: Settings) -> None:
        """统一创建聊天模型，并分别绑定意图、摘要与记忆的结构化输出契约。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self._settings = settings
        # _llm：底层聊天模型或统一模型适配器，由当前类职责决定。
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            # 不传就是 httpx 默认值：模型排队时请求一直挂着，先把整轮预算吃光。
            client_kwargs={"timeout": settings.ollama_timeout_seconds},
        )
        # _answer_model：用于自由文本回答且统一处理异常的模型包装器。
        self._answer_model = _GuardedModel(self._llm, "回答")
        # _intent_llm：按 LlmOutput JSON Schema 输出的意图模型。
        self._intent_llm = _GuardedModel(
            self._llm.with_structured_output(LlmOutput, method="json_schema"), "意图抽取"
        )
        # _summary_llm：按 SummaryFields JSON Schema 输出的摘要模型。
        self._summary_llm = _GuardedModel(
            self._llm.with_structured_output(SummaryFields, method="json_schema"), "摘要"
        )
        # _memory_llm：按 MemoryDraftList JSON Schema 输出的记忆模型。
        self._memory_llm = _GuardedModel(
            self._llm.with_structured_output(MemoryDraftList, method="json_schema"), "记忆抽取"
        )

    @property
    def model_name(self) -> str:
        """读取当前配置的聊天模型名称。"""
        return self._settings.ollama_model

    def bind_tools(self, tools: list[BaseTool]) -> Runnable:
        """给模型绑定允许使用的工具列表并返回可调用对象。

        Args:
            tools: 允许绑定给模型的工具实例列表。
        """
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
        """取得底层 ChatOllama 模型实例，供需要原始模型的调用方使用。"""
        return self._llm


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
