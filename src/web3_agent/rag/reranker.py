"""Optional HTTP cross-encoder adapter (TEI-compatible /rerank contract)."""
import math
import httpx

from web3_agent.common.exceptions import BudgetExhausted
from web3_agent.common.metering import Resource, reserve

# 数据职责：可选的 HTTP cross-encoder 重排适配器。
class HttpReranker:
    def __init__(self, settings):
        """初始化 HttpReranker 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        """
        self.settings = settings

    async def rerank(self, query, candidates):
        # 在调用向量化或重排服务前检查共享下游次数，耗尽时不再发出请求。
        """调用重排服务，为所有候选校验唯一且有限的分数后重新排序。

        Args:
            query: 当前检索问题文本。
            candidates: 召回融合后待重排的切片列表。
        """
        if not reserve(Resource.EMBEDDING):
            raise BudgetExhausted("本轮重排次数已达上限")
        async with httpx.AsyncClient(timeout=self.settings.rag_rerank_timeout_seconds) as client:
            response = await client.post(self.settings.rag_rerank_url, json={
                "query": query, "texts": [c.content for c in candidates], "truncate": True,
            })
            response.raise_for_status()
            rows = response.json()
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
