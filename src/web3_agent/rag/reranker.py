"""Optional HTTP cross-encoder adapter: self-hosted TEI and online /rerank contracts."""
import math

import httpx

from web3_agent.common.exceptions import BudgetExhausted
from web3_agent.common.metering import Resource, reserve


def _tei_body(settings, query, texts):
    """自建 TEI 的请求体。"""
    return {"query": query, "texts": texts, "truncate": True}


def _tei_rows(data):
    """TEI 直接返回裸数组。"""
    return data


def _cohere_body(settings, query, texts):
    """Cohere、Jina、硅基流动共用的请求体。"""
    return {"model": settings.rag_rerank_model, "query": query,
            "documents": texts, "top_n": len(texts)}


def _cohere_rows(data):
    """Cohere 兼容响应：结果在 results 下，分数字段是 relevance_score。"""
    return [{"index": row["index"], "score": row["relevance_score"]}
            for row in data["results"]]


def _dashscope_body(settings, query, texts):
    """百炼 text-rerank 的请求体，输入与参数分两层。"""
    return {"model": settings.rag_rerank_model,
            "input": {"query": query, "documents": texts},
            "parameters": {"return_documents": False, "top_n": len(texts)}}


def _dashscope_rows(data):
    """百炼响应：结果在 output.results 下，分数字段是 relevance_score。"""
    return [{"index": row["index"], "score": row["relevance_score"]}
            for row in data["output"]["results"]]


# _SHAPES：重排契约到（请求体构造、响应取行）的映射，只描述字段差异。
# 分数校验、排序与降级口径对三种契约完全一致，不留第二份实现。
_SHAPES = {
    "tei": (_tei_body, _tei_rows),
    "cohere": (_cohere_body, _cohere_rows),
    "dashscope": (_dashscope_body, _dashscope_rows),
}


# 数据职责：可选的 HTTP cross-encoder 重排适配器。
class HttpReranker:
    def __init__(self, settings):
        """初始化 HttpReranker 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self.settings = settings

    def _request(self, query, candidates):
        # 按 provider 取契约；凭据统一放 Bearer 头，自建 TEI 不带凭据。
        """组装本次重排的请求体、响应取行方式与鉴权头。

        Args:
            query: 当前检索问题文本。
            candidates: 召回融合后待重排的切片列表。
        """
        build_body, read_rows = _SHAPES[self.settings.rag_rerank_provider]
        headers = ({"Authorization": f"Bearer {self.settings.rag_rerank_api_key}"}
                   if self.settings.rag_rerank_api_key else {})
        return build_body(self.settings, query, [c.content for c in candidates]), read_rows, headers

    async def rerank(self, query, candidates):
        # 在调用向量化或重排服务前检查共享下游次数，耗尽时不再发出请求。
        """调用重排服务，为所有候选校验唯一且有限的分数后重新排序。

        Args:
            query: 当前检索问题文本。
            candidates: 召回融合后待重排的切片列表。
        """
        if not reserve(Resource.EMBEDDING):
            raise BudgetExhausted("本轮重排次数已达上限")
        payload, read_rows, headers = self._request(query, candidates)
        async with httpx.AsyncClient(timeout=self.settings.rag_rerank_timeout_seconds) as client:
            response = await client.post(self.settings.rag_rerank_url, json=payload, headers=headers)
            response.raise_for_status()
            rows = read_rows(response.json())
        if not isinstance(rows, list) or len(rows) != len(candidates):
            raise ValueError("incomplete rerank result")
        scores = {}
        # 校验每个候选恰好收到一个合法分数，重复序号、越界序号或无穷数都视为坏响应。
        for row in rows:
            index, score = row["index"], float(row["score"])
            if isinstance(index, bool) or not isinstance(index, int) or index not in range(len(candidates)) or index in scores or not math.isfinite(score):
                raise ValueError("invalid rerank result")
            scores[index] = score
        ordered = sorted(scores, key=lambda i: (-scores[i], candidates[i].chunk_id))
        return [(candidates[i], scores[i]) for i in ordered]
