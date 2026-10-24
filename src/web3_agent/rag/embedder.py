"""文本向量化，走本地 Ollama 的 bge-m3。

`vector(1024)` 是建表时写死的，模型换了维度就对不上 —— 所以这里校验维度，
不把「写库时才报错」留给调用方。
"""

from __future__ import annotations

import httpx
from langchain_ollama import OllamaEmbeddings

from web3_agent.common.exceptions import BudgetExhausted, ModelUnavailable
from web3_agent.config.settings import Settings
from web3_agent.common.metering import Resource, reserve


# 数据职责：Ollama 文本向量化适配器，统一计量并检查数据库维度兼容性。
class Embedder:
    def __init__(self, settings: Settings) -> None:
        # _model：Ollama 嵌入模型客户端。
        """初始化 Embedder 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self._model = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
        # _dim：预期向量维度，用于与数据库向量列校验。
        self._dim = settings.ollama_embed_dim

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
        except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            raise ModelUnavailable(f"向量化失败：{exc}") from exc
        for vec in vectors:
            if len(vec) != self._dim:
                raise ModelUnavailable(
                    f"向量维度不一致：期望 {self._dim}，实际 {len(vec)}"
                )
        return vectors
