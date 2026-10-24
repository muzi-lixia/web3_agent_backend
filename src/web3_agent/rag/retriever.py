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
from web3_agent.common.exceptions import AppError

logger = logging.getLogger(__name__)
# NO_DOCUMENTS：检索范围内没有可用证据时使用的固定答复。
NO_DOCUMENTS = "未找到相关文档信息，请核对链或协议信息，当前无法确认该问题。"
# UNAVAILABLE：知识检索不可用时使用的固定降级答复。
UNAVAILABLE = "知识库暂时不可用，请稍后重试；当前无法确认文档信息。"

# 召回与可选阶段共用一条总预算，但都必须在总截止线之前收工：融合与装配是本地计算，
# 一旦被某个分支吃光预算，已经算出来的关键词结果会跟着一起作废。
# _FUSION_RESERVE_SECONDS：给本地融合与证据组装预留的时间，单位为秒。
_FUSION_RESERVE_SECONDS = 0.5


def _remaining(deadline):
    """按单调时钟计算当前阶段距离截止时间的剩余秒数。

    Args:
        deadline: 当前阶段基于单调时钟的截止时间。
    """
    return max(0.0, deadline - monotonic())


def _discard(branch):
    """取走被回收分支的结果：取消的不算失败，真出错的也要消费掉，别留 never-retrieved。

    Args:
        branch: 需要消费结果或异常的单个并发任务。
    """
    if not branch.cancelled():
        branch.exception()


async def _settle(branches, deadline):
    """在 deadline 前收齐一组召回分支：超时的取消并回收，已完成的结果照常返回。

    返回 (按输入顺序的结果, 未完成的分支集合)；未完成的位置是 None。
    分支异常沿用 gather 语义上抛，不让召回故障变成空结果。

    Args:
        branches: 需要在截止时间内收集的并发任务列表。
        deadline: 当前阶段基于单调时钟的截止时间。
    """
    done, pending = set(), set(branches)
    try:
        done, pending = await asyncio.wait(branches, timeout=_remaining(deadline))
        return [(branch.result() if branch in done else None) for branch in branches], pending
    finally:
        # 子任务不是父任务的后代，父任务被取消时取消不会向下传播，必须在这里收。
        # 漏掉这条路径，客户端断开后 Embedding 和 SQL 会继续跑完，
        # 既脱离本轮结算，晚到的异常也没人取。
        # 这里不等它们结束：等待会吃掉融合与回答的预算，取消本身才是收敛手段。
        for branch in branches:
            if not branch.done():
                branch.cancel()
                branch.add_done_callback(_discard)


# 进缓存键的配置：取全部 rag_* 字段，只排掉不改变召回结果的那些。
# 用白名单时要逐项登记，加了新配置忘了登记，改配置就会静默复用旧结果。
_CACHE_KEY_EXCLUDED = frozenset({
    # 只影响取数节奏与缓存自身
    "rag_timeout_seconds", "rag_rewrite_timeout_seconds", "rag_rerank_timeout_seconds",
    "rag_cache_ttl_seconds", "rag_cache_max_entries", "rag_degraded_cache_ttl_seconds",
    # 建库、上传与后台任务，与在线检索无关
    "rag_enabled", "rag_upload_enabled", "rag_admin_addresses", "rag_upload_max_bytes",
    "rag_max_text_chars", "rag_max_chunks", "rag_chunk_tokens", "rag_chunk_overlap",
    "rag_worker_poll_seconds", "rag_job_timeout_seconds", "rag_job_max_attempts",
})


def _select_evidence(chunks, *, max_chars, limit):
    """按字符预算挑证据并连续编号。

    纯向量和混合两条召回路径共用：编号规则、字符预算、超长片段跳过的口径
    只要有一份，改上下文预算或引用格式时不会只改到其中一条。

    Args:
        chunks: 本批文档切片，或任务的切片统计数量。
        max_chars: 本段文本允许的最大字符预算。
        limit: 本次查询或展示返回条数上限。

    Returns:
        未超字符与条数预算、编号为 S1 起的证据字典列表。
    """
    remaining, selected = max_chars, []
    for chunk in chunks:
        if len(chunk.content) <= remaining:
            item = chunk.model_dump()
            item["id"] = f"S{len(selected) + 1}"
            selected.append(item)
            remaining -= len(chunk.content)
        if len(selected) >= limit:
            break
    return selected


# 数据职责：知识检索编排器，统一处理预算、缓存、召回、融合与降级。
class KnowledgeRetriever:
    def __init__(self, settings, embedder, store):
        # settings：运行配置对象，提供本模块所需开关、阈值和服务地址。
        # embedder：把文本转换为检索向量的服务。
        # store：文档和切片的持久仓储。
        """初始化 KnowledgeRetriever 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
            embedder: 把文本转换为检索向量的服务。
            store: 文档和切片的持久仓储。
        """
        self.settings, self.embedder, self.store = settings, embedder, store
        # cache：按问题、范围和索引版本保存的检索结果缓存。
        self.cache = RetrievalCache(settings.rag_cache_max_entries)
        # 索引对象只读，共用即可；逐次深拷贝等于每个请求都把整个知识库再铺一遍。
        # index_cache：按范围、版本和有效期边界缓存的只读 BM25 索引。
        self.index_cache = RetrievalCache(max_entries=8, clone=False)
        # rewriter：可选的问题指代消解与同义改写适配器。
        self.rewriter = QueryRewriter(settings)
        # reranker：可选的候选证据重排适配器。
        self.reranker = HttpReranker(settings)

    @traceable(name="rag.retrieve", run_type="chain")
    async def search(self, query, *, chain_id=None, protocol_id=None, doc_category=None, top_k=None, kb_id="default", conversation_context=""):
        # 检索结果 status 为 success、empty、invalid 或 unavailable；失败时含 error、message，hits 为空。
        # 成功结果 hits 为引用证据列表，filters 为实际精确过滤范围；混合检索另给 resolved_question 与 retrieval 诊断信息。
        """校验检索范围并应用总时间预算，执行向量或混合检索，返回证据或明确降级状态。

        Args:
            query: 当前检索问题文本。
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            protocol_id: 协议精确过滤，None 表示不限制。
            doc_category: 文档分类精确过滤，None 表示不限制。
            top_k: 期望返回的证据条数，实际仍受配置上限限制。
            kb_id: 目标知识库标识，默认 default。
            conversation_context: 受限的最近历史，只用于理解当前问题指代。

        Returns:
            dict: 包含 status、hits，以及成功过滤范围或失败错误信息的检索结果。
        """
        if not self.settings.rag_enabled:
            return {"error": "E603", "status": "unavailable", "message": UNAVAILABLE, "hits": []}
        try:
            filters = normalize_filters(chain_id, protocol_id, doc_category, kb_id)
        except ValueError:
            return {"error": "E101", "status": "invalid", "message": "链、协议或分类过滤参数格式错误", "hits": []}
        if not query.strip() or len(query) > 4000:
            return {"error": "E101", "status": "invalid", "message": "知识查询内容不能为空或超过 4000 字符", "hits": []}
        limit = min(max(1, top_k or self.settings.rag_top_k), self.settings.rag_top_k)
        deadline = monotonic() + self.settings.rag_timeout_seconds
        try:
            # 外层只做最后一道兜底：各阶段自带截止时间，正常不会走到这里。
            async with asyncio.timeout_at(deadline):
                if not await self.store.has_documents(**filters):
                    return {"error": "E602", "status": "empty", "message": NO_DOCUMENTS, "hits": []}
                if self.settings.rag_hybrid_enabled:
                    return await self._hybrid(query, filters, limit, conversation_context, deadline)
                vector = await self.embedder.embed(query)
                hits = await self.store.search(vector, **filters, limit=limit,
                                               max_distance=self.settings.rag_max_distance,
                                               validate=False)
        except AppError:
            raise
        except Exception:
            logger.exception("knowledge retrieval failed filters=%s", filters)
            return {"error": "E603", "status": "unavailable", "message": UNAVAILABLE, "hits": []}
        # 最后统一裁剪证据条数与字符预算，并生成本轮连续引用编号。
        selected = _select_evidence(
            hits, max_chars=self.settings.rag_max_context_chars, limit=limit)
        if not selected:
            return {"error": "E602", "status": "empty", "message": NO_DOCUMENTS, "hits": []}
        return {"status": "success", "filters": filters, "hits": selected}

    async def _keyword_index(self, filters, revision, boundary, deadline, info):
        """常驻索引：版本与到期边界都没变就直接复用，变了才重建一次。

        Args:
            filters: 已经规范化的知识库及业务范围过滤字典。
            revision: 当前知识库索引版本。
            boundary: 有效文档中最近的到期时间边界。
            deadline: 当前阶段基于单调时钟的截止时间。
            info: 当前检索诊断信息，记录降级阶段、候选和耗时。
        """
        cache_key = repr((filters, revision, boundary))
        index = self.index_cache.get(cache_key)
        if index is not None:
            return index
        try:
            chunks = await asyncio.wait_for(
                self.store.keyword_snapshot(max_chunks=self.settings.rag_keyword_max_chunks,
                                            validate=False, **filters),
                _remaining(deadline))
        except AppError as exc:
            # 切片数超过建索引上限只影响关键词这一路：退回纯向量，
            # 不能让整条查询跟着失败。其余 AppError（库不存在、向量档不兼容）照旧上抛。
            if exc.code != "E613":
                raise
            logger.warning("RAG keyword index over limit; using vector recall")
            info["degraded_stages"].append("keyword")
            return None
        except Exception:
            logger.warning("RAG keyword snapshot failed; using vector recall", exc_info=True)
            info["degraded_stages"].append("keyword")
            return None
        # 只在版本或有效期边界变化后构建关键词索引，后续查询复用只读对象。
        index = BM25Index(chunks)
        # TTL 只是兜底：版本号覆盖了所有写路径，它防的是绕过应用的直接改库。
        self.index_cache.put(cache_key, index, self.settings.rag_cache_ttl_seconds)
        return index

    @traceable(name="rag.hybrid", run_type="chain")
    async def _hybrid(self, query, filters, limit, conversation_context="", deadline=None):
        # 诊断字段 cache_hit 表示结果缓存命中；degraded_stages 列出降级环节；queries 为各路问题，resolved_question 为消歧后的主问题。
        # dense_candidates/keyword_candidates 记录各路候选 ID；fusion_scores/rerank_scores 记录排序分数，selected 为最终证据 ID。
        # elapsed_ms 为本次检索耗时毫秒数；dense_timeout_branches 仅在向量分支超时时出现。
        """在同一预算内执行改写、并发向量及关键词召回、RRF 融合与可选重排。

        Args:
            query: 当前检索问题文本。
            filters: 已经规范化的知识库及业务范围过滤字典。
            limit: 本次查询或展示返回条数上限。
            conversation_context: 受限的最近历史，只用于理解当前问题指代。
            deadline: 当前阶段基于单调时钟的截止时间。
        """
        start = monotonic()
        info = {"cache_hit": False, "degraded_stages": []}
        budget = self.settings.rag_timeout_seconds
        recall_deadline = (deadline or start + budget) - min(_FUSION_RESERVE_SECONDS, budget * 0.25)
        # 索引版本 + 到期边界。命中缓存时也只读这一行，不碰正文。
        revision = boundary = None
        try:
            revision, boundary = await asyncio.wait_for(
                self.store.index_state(validate=False, **filters), _remaining(recall_deadline))
        except AppError:
            # Missing/disabled KB or incompatible embedding profile is not a recall outage.
            raise
        except Exception:
            # 版本都拿不到就无法判断索引是否过期，这一轮只走向量、不复用任何缓存。
            logger.warning("RAG index version unavailable; vector recall only", exc_info=True)
            info["degraded_stages"].append("keyword")
        config = {key: value for key, value in self.settings.model_dump().items()
                  if key.startswith("rag_") and key not in _CACHE_KEY_EXCLUDED}
        config["embedding_model"] = self.settings.ollama_embed_model
        config["rewrite_model"] = self.settings.ollama_model
        key = hashlib.sha256(json.dumps([query, conversation_context[-2000:], filters, limit,
                                         revision, boundary, config],
                                       sort_keys=True, default=str).encode()).hexdigest()
        cached = self.cache.get(key) if revision is not None else None
        if cached is not None:
            cached["retrieval"]["cache_hit"] = True
            cached["retrieval"]["elapsed_ms"] = round((monotonic()-start)*1000)
            return cached
        # resolved 消解了指代，召回 / 重排 / 回答三个阶段共用它；variants 只用于多路召回。
        resolved, queries = query, [query]
        if self.settings.rag_rewrite_enabled:
            try:
                resolved, queries = await asyncio.wait_for(
                    self.rewriter.rewrite(query, conversation_context[-2000:]), _remaining(recall_deadline))
            except Exception:
                logger.warning("RAG rewrite failed; using original query", exc_info=True)
                info["degraded_stages"].append("rewrite")
        info["queries"] = queries
        info["resolved_question"] = resolved
        keyword_rankings: list = []
        # Dense failures must not discard a useful exact-term match; a hanging branch
        # is cut at the recall deadline and only its own result is given up.
        async def dense(q):
            """执行单路问题向量化和向量检索；非业务故障降级为空候选。

            Args:
                q: 当前一路向量召回的问题表达。
            """
            try:
                vector = await self.embedder.embed(q)
                return await self.store.search(vector, **filters, limit=self.settings.rag_candidate_k,
                                               max_distance=self.settings.rag_max_distance,
                                               validate=False)
            except AppError:
                raise
            except Exception:
                logger.warning("RAG dense recall failed", exc_info=True)
                info["degraded_stages"].append("dense")
                return []
        # 索引准备和向量召回同批起跑。串行时关键词快照一慢就把召回预算吃光，
        # 向量分支带着 0 秒预算启动，「降级到向量」这条路径等于不存在。
        dense_branches = [asyncio.ensure_future(dense(q)) for q in queries]
        index_task = (
            asyncio.ensure_future(
                self._keyword_index(filters, revision, boundary, recall_deadline, info))
            if revision is not None else None
        )
        results, pending = await _settle(
            dense_branches + ([index_task] if index_task is not None else []), recall_deadline)
        dense_rankings = [result or [] for result in results[:len(dense_branches)]]
        index = results[-1] if index_task is not None else None
        if pending:
            timed_out = sum(1 for branch in pending if branch in dense_branches)
            logger.warning("RAG recall hit the stage budget; kept %d/%d branches",
                           len(dense_branches) - timed_out, len(dense_branches))
            if timed_out:
                info["degraded_stages"].append("dense_timeout")
                info["dense_timeout_branches"] = timed_out
            if index_task in pending:
                info["degraded_stages"].append("keyword")
        if index is not None:
            try:
                keyword_rankings = [[chunk for chunk, _ in index.search(q, self.settings.rag_candidate_k)]
                                    for q in queries]
            except Exception:
                logger.warning("RAG keyword scoring failed; using vector recall", exc_info=True)
                info["degraded_stages"].append("keyword")
        # With a snapshot, intersect candidates; without one, fresh vector SQL enforces
        # active KB, document validity and all explicit scope filters.
        live = {c.chunk_id for c in index.chunks} if index is not None else None
        rankings = [[c for c in ranking if live is None or c.chunk_id in live]
                    for ranking in dense_rankings+keyword_rankings]
        # 用各路排序名次执行 RRF 融合，避免直接比较不同算法的原始分数。
        candidates, scores = fuse(rankings, limit=self.settings.rag_fusion_k)
        info["dense_candidates"] = [[c.chunk_id for c in r] for r in dense_rankings]
        info["keyword_candidates"] = [[c.chunk_id for c in r] for r in keyword_rankings]
        info["fusion_scores"] = scores
        if candidates and self.settings.rag_rerank_enabled:
            try:
                ranked = await asyncio.wait_for(
                    self.reranker.rerank(resolved, candidates), _remaining(recall_deadline))
                info["rerank_scores"] = {c.chunk_id: score for c, score in ranked}
                threshold = self.settings.rag_rerank_min_score
                candidates = [c for c, score in ranked if threshold is None or score >= threshold]
            except Exception:
                logger.warning("RAG rerank failed; retaining RRF ordering", exc_info=True)
                info["degraded_stages"].append("rerank")
        # 最后统一裁剪证据条数与字符预算，并生成本轮连续引用编号。
        selected = _select_evidence(
            candidates, max_chars=self.settings.rag_max_context_chars, limit=limit)
        info["selected"] = [c["chunk_id"] for c in selected]
        info["elapsed_ms"] = round((monotonic()-start)*1000)
        if not selected:
            degraded = set(info["degraded_stages"])
            status = "unavailable" if degraded & {"dense", "dense_timeout", "keyword"} else "empty"
            return {"error": "E603" if status == "unavailable" else "E602", "status": status,
                    "message": UNAVAILABLE if status == "unavailable" else NO_DOCUMENTS,
                    "resolved_question": resolved, "hits": [], "retrieval": info}
        result = {"status": "success", "filters": filters, "resolved_question": resolved,
                  "hits": selected, "retrieval": info}
        # Short-lived degraded evidence reduces load while bounding recovery delay.
        # A fresh corpus/scope revision is always required even on cache hits.
        if revision is not None:
            ttl = self.settings.rag_cache_ttl_seconds
            if info["degraded_stages"]:
                ttl = min(ttl, self.settings.rag_degraded_cache_ttl_seconds)
            self.cache.put(key, result, ttl)
        return result
