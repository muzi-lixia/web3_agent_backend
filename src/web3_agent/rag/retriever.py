"""通用静态知识检索，只返回文档证据。"""
import asyncio
import logging
import hashlib
import json
from time import monotonic
from langsmith import traceable
from web3_agent.rag.cache import RetrievalCache
from web3_agent.rag.keyword_index import BM25Index
from web3_agent.rag.fusion import fuse
from web3_agent.rag.query_rewriter import QueryRewriter
from web3_agent.rag.reranker import HttpReranker
from web3_agent.rag.config import normalize_filters

logger = logging.getLogger(__name__)
NO_DOCUMENTS = "未找到相关文档信息，请核对链或协议信息，当前无法确认该问题。"
UNAVAILABLE = "知识库暂时不可用，请稍后重试；当前无法确认文档信息。"


class KnowledgeRetriever:
    def __init__(self, settings, embedder, store):
        self.settings, self.embedder, self.store = settings, embedder, store
        self.cache = RetrievalCache(settings.rag_cache_max_entries)
        self.index_cache = RetrievalCache(max_entries=8)
        self.rewriter = QueryRewriter(settings)
        self.reranker = HttpReranker(settings)

    @traceable(name="rag.retrieve", run_type="chain")
    async def search(self, query, *, chain_id=None, protocol_id=None, doc_category=None, top_k=None, kb_id="default", conversation_context=""):
        if not self.settings.rag_enabled:
            return {"error": "E603", "status": "unavailable", "message": UNAVAILABLE, "hits": []}
        try:
            filters = normalize_filters(chain_id, protocol_id, doc_category, kb_id)
        except ValueError:
            return {"error": "E101", "status": "invalid", "message": "链、协议或分类过滤参数格式错误", "hits": []}
        if not query.strip() or len(query) > 4000:
            return {"error": "E101", "status": "invalid", "message": "知识查询内容不能为空或超过 4000 字符", "hits": []}
        limit = min(max(1, top_k or self.settings.rag_top_k), self.settings.rag_top_k)
        try:
            async with asyncio.timeout(self.settings.rag_timeout_seconds):
                if not await self.store.has_documents(**filters):
                    return {"error": "E602", "status": "empty", "message": NO_DOCUMENTS, "hits": []}
                if self.settings.rag_hybrid_enabled:
                    return await self._hybrid(query, filters, limit, conversation_context)
                vector = await self.embedder.embed(query)
                hits = await self.store.search(vector, **filters, limit=limit,
                                               max_distance=self.settings.rag_max_distance)
        except Exception:
            logger.exception("knowledge retrieval failed filters=%s", filters)
            return {"error": "E603", "status": "unavailable", "message": UNAVAILABLE, "hits": []}
        remaining, selected = self.settings.rag_max_context_chars, []
        for hit in hits:
            if len(hit.content) > remaining:
                continue
            selected.append(hit.model_dump())
            remaining -= len(hit.content)
        if not selected:
            return {"error": "E602", "status": "empty", "message": NO_DOCUMENTS, "hits": []}
        return {"status": "success", "filters": filters, "hits": selected}

    @traceable(name="rag.hybrid", run_type="chain")
    async def _hybrid(self, query, filters, limit, conversation_context=""):
        start = monotonic()
        info = {"cache_hit": False, "degraded_stages": []}
        # Scope is revalidated before cache lookup; removed/expired chunks change revision.
        revision, corpus = await self.store.keyword_snapshot(
            max_chunks=self.settings.rag_keyword_max_chunks, **filters)
        if not corpus:
            return {"error": "E602", "status": "empty", "message": NO_DOCUMENTS, "hits": []}
        config = {k: v for k, v in self.settings.model_dump().items() if k.startswith("rag_")}
        config["embedding_model"] = self.settings.ollama_embed_model
        config["rewrite_model"] = self.settings.ollama_model
        key = hashlib.sha256(json.dumps([query, conversation_context[-2000:], filters, limit, revision, config],
                                       sort_keys=True, default=str).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            cached["retrieval"]["cache_hit"] = True
            cached["retrieval"]["elapsed_ms"] = round((monotonic()-start)*1000)
            return cached
        queries = [query]
        if self.settings.rag_rewrite_enabled:
            try:
                queries = await self.rewriter.rewrite(query, conversation_context[-2000:])
            except Exception:
                logger.warning("RAG rewrite failed; using original query", exc_info=True)
                info["degraded_stages"].append("rewrite")
        info["queries"] = queries
        scope_key = repr((filters, revision))
        index = self.index_cache.get(scope_key)
        if index is None:
            index = BM25Index(corpus)
            self.index_cache.put(scope_key, index, self.settings.rag_cache_ttl_seconds)
        keyword_rankings = [[chunk for chunk, _ in index.search(q, self.settings.rag_candidate_k)] for q in queries]
        # Dense failures must not discard a useful exact-term match.
        async def dense(q):
            try:
                vector = await self.embedder.embed(q)
                return await self.store.search(vector, **filters, limit=self.settings.rag_candidate_k,
                                               max_distance=self.settings.rag_max_distance)
            except Exception:
                logger.warning("RAG dense recall failed", exc_info=True)
                info["degraded_stages"].append("dense")
                return []
        dense_rankings = await asyncio.gather(*(dense(q) for q in queries))
        # Only retain chunks present in the live scoped snapshot, even if a version changes mid-request.
        live = {c.chunk_id: c for c in corpus}
        rankings = [[c for c in ranking if c.chunk_id in live] for ranking in dense_rankings+keyword_rankings]
        candidates, scores = fuse(rankings, limit=self.settings.rag_fusion_k)
        info["dense_candidates"] = [[c.chunk_id for c in r] for r in dense_rankings]
        info["keyword_candidates"] = [[c.chunk_id for c in r] for r in keyword_rankings]
        info["fusion_scores"] = scores
        if candidates and self.settings.rag_rerank_enabled:
            try:
                ranked = await self.reranker.rerank(query, candidates)
                info["rerank_scores"] = {c.chunk_id: score for c, score in ranked}
                threshold = self.settings.rag_rerank_min_score
                candidates = [c for c, score in ranked if threshold is None or score >= threshold]
            except Exception:
                logger.warning("RAG rerank failed; retaining RRF ordering", exc_info=True)
                info["degraded_stages"].append("rerank")
        remaining, selected = self.settings.rag_max_context_chars, []
        for chunk in candidates:
            if len(chunk.content) <= remaining:
                item = chunk.model_dump()
                item["id"] = f"S{len(selected)+1}"
                selected.append(item)
                remaining -= len(chunk.content)
            if len(selected) >= limit:
                break
        info["selected"] = [c["chunk_id"] for c in selected]
        info["elapsed_ms"] = round((monotonic()-start)*1000)
        if not selected:
            status = "unavailable" if "dense" in info["degraded_stages"] else "empty"
            return {"error": "E603" if status == "unavailable" else "E602", "status": status,
                    "message": UNAVAILABLE if status == "unavailable" else NO_DOCUMENTS,
                    "hits": [], "retrieval": info}
        result = {"status": "success", "filters": filters, "hits": selected, "retrieval": info}
        # Do not cache partial failures; a transient model outage should recover immediately.
        if not info["degraded_stages"]:
            self.cache.put(key, result, self.settings.rag_cache_ttl_seconds)
        return result
