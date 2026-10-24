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


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._model = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
        self._dim = settings.ollama_embed_dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
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
