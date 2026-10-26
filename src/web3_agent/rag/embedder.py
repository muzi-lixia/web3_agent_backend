"""文本向量化，接入方式按配置选。

`vector(1024)` 是建表时写死的，模型换了维度就对不上 —— 所以这里校验维度，
不把「写库时才报错」留给调用方。换模型或换供应商之后，已入库的向量必须在
新模型下重建，否则检索侧的 query 向量和 doc 向量不在同一个空间。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from web3_agent.common.exceptions import (
    BudgetExhausted,
    ModelConfigError,
    ModelUnavailable,
)
from web3_agent.common.metering import Resource, reserve
from web3_agent.config.settings import Settings
from web3_agent.llm.client import normalize_model_error

# _DEFAULT_OLLAMA_BASE_URL：本地向量服务默认地址，向量留在本地而聊天走远端时使用。
_DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _build_embeddings(settings: Settings) -> Embeddings:
    """按配置创建向量模型客户端。

    Args:
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    if settings.embed_provider == "ollama":
        # 聊天在远端、向量在本地时，两条链路地址不同，不能回落到 MODEL_BASE_URL。
        base_url = settings.embed_base_url or (
            settings.model_base_url
            if settings.model_provider == "ollama" else _DEFAULT_OLLAMA_BASE_URL
        )
        return OllamaEmbeddings(model=settings.embed_model, base_url=base_url)
    base_url = settings.embed_base_url or settings.model_base_url
    api_key = settings.embed_api_key or settings.model_api_key
    if not api_key:
        raise ModelConfigError("向量模型缺少 EMBED_API_KEY")
    return OpenAIEmbeddings(
        model=settings.embed_model,
        base_url=base_url,
        api_key=api_key,
        dimensions=settings.embed_dim,
        chunk_size=settings.embed_batch_size,
        # 同聊天模型：隐形重试会让一次计量对应多次上游请求。
        max_retries=0,
        # 不改就是 tiktoken 切 token 再发，非 OpenAI 的兼容端点只收原文。
        check_embedding_ctx_length=False,
        # 默认 base64 编码，兼容端点普遍只认 float。
        model_kwargs={"encoding_format": "float"},
    )


# 数据职责：文本向量化适配器，统一计量并检查数据库维度兼容性。
class Embedder:
    def __init__(self, settings: Settings) -> None:
        """初始化 Embedder 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        # _model：向量模型客户端，供应商由配置决定。
        self._model = _build_embeddings(settings)
        # _dim：预期向量维度，用于与数据库向量列校验。
        self._dim = settings.embed_dim

    @property
    def dim(self) -> int:
        """返回当前向量模型配置的维度。"""
        return self._dim

    async def embed(self, text: str) -> list[float]:
        """把一条文本向量化，复用批量向量化的计量和维度检查。

        Args:
            text: 待检查、解析或处理的原始文本。
        """
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        # 返回值：与输入文本顺序一致的向量列表；输入为空时返回空列表。
        """批量调用嵌入模型，计入下游预算并校验返回向量维度。

        Args:
            texts: 按输入顺序组成的待向量化文本列表。

        Raises:
            BudgetExhausted: 本轮下游调用预算已用尽。
            ModelUnavailable: 模型不可达、调用超时或向量维度不匹配。
        """
        if not texts:
            return []
        # 在调用向量化或重排服务前检查共享下游次数，耗尽时不再发出请求。
        if not reserve(Resource.EMBEDDING):
            raise BudgetExhausted("本轮向量化次数已达上限")
        try:
            vectors = await self._model.aembed_documents(texts)
        except Exception as exc:
            # 在线接入抛的是 SDK 自己的连接异常，不是 httpx，逐类捕获会漏。
            raise normalize_model_error(exc, "向量化") from exc
        for vec in vectors:
            if len(vec) != self._dim:
                raise ModelUnavailable(
                    f"向量维度不一致：期望 {self._dim}，实际 {len(vec)}"
                )
        return vectors
