"""Optional HTTP cross-encoder adapter (TEI-compatible /rerank contract)."""
import math
import httpx

from web3_agent.common.exceptions import BudgetExhausted
from web3_agent.common.metering import Resource, reserve

class HttpReranker:
    def __init__(self, settings):
        self.settings = settings

    async def rerank(self, query, candidates):
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
        for row in rows:
            index, score = row["index"], float(row["score"])
            if isinstance(index, bool) or not isinstance(index, int) or index not in range(len(candidates)) or index in scores or not math.isfinite(score):
                raise ValueError("invalid rerank result")
            scores[index] = score
        ordered = sorted(scores, key=lambda i: (-scores[i], candidates[i].chunk_id))
        return [(candidates[i], scores[i]) for i in ordered]
