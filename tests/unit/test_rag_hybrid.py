"""Hybrid retrieval regressions: lexical rescue, scope revision, degradation, reranking."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from web3_agent.config.settings import Settings
from web3_agent.rag.models import Citation
from web3_agent.rag.keyword_index import BM25Index
from web3_agent.rag.fusion import fuse
from web3_agent.rag.retriever import KnowledgeRetriever
from web3_agent.rag.cache import RetrievalCache


def chunk(key, text):
    return Citation(id='S1', chunk_id=key, title='FAQ', section='', source_url='',
                    version='v1', doc_category='faq', updated_at='2026-01-01', content=text, distance=2)


def retriever(chunks, **options):
    config = Settings(_env_file=None, rag_enabled=True, rag_hybrid_enabled=True, **options)
    store = SimpleNamespace(has_documents=AsyncMock(return_value=True),
        keyword_snapshot=AsyncMock(return_value=('revision1', chunks)), search=AsyncMock(return_value=[]))
    return KnowledgeRetriever(config, SimpleNamespace(embed=AsyncMock(return_value=[1.])), store)


def test_lexical_casefold_and_chinese_matches():
    index = BM25Index([chunk('dex', '什么是DEX？去中心化交易所。'), chunk('gas', 'Gas 是手续费。')])
    assert index.search('什么是dex？')[0][0].chunk_id == 'dex'
    assert index.search('去中心化交易所')[0][0].chunk_id == 'dex'
    assert index.search('DEX') == index.search('dex')


def test_rrf_deduplicates_and_promotes_agreement():
    a, b, c = [chunk(k, k) for k in 'abc']
    result, _ = fuse([[a,b,b], [c,b]])
    assert [v.chunk_id for v in result] == ['b','a','c']


def test_bm25_rescues_vector_miss_and_cache_respects_scope_revision():
    async def run():
        item = chunk('dex', 'DEX is a decentralized exchange.')
        r = retriever([item])
        first = await r.search('什么是dex？')
        assert first['hits'][0]['chunk_id'] == 'dex'
        assert first['retrieval']['cache_hit'] is False
        second = await r.search('什么是dex？')
        assert second['retrieval']['cache_hit'] is True
        assert r.embedder.embed.await_count == 1
        # A disabled/expired document is absent from the next live snapshot.
        r.store.keyword_snapshot.return_value = ('revision2', [])
        assert (await r.search('什么是dex？'))['status'] == 'empty'
        r.store.keyword_snapshot.return_value = ('revision3', [item])
        await r.search('什么是dex？', kb_id='other')
        assert r.embedder.embed.await_count == 2
        assert r.store.keyword_snapshot.call_args.kwargs['kb_id'] == 'other'
    asyncio.run(run())


def test_dense_and_rewrite_failure_preserve_keyword_evidence_without_cache():
    async def run():
        r = retriever([chunk('dex', 'DEX definition')], rag_rewrite_enabled=True)
        r.rewriter.rewrite = AsyncMock(side_effect=TimeoutError)
        r.embedder.embed.side_effect = TimeoutError
        result = await r.search('dex')
        assert result['status'] == 'success'
        assert set(result['retrieval']['degraded_stages']) == {'rewrite','dense'}
        await r.search('dex')
        assert r.rewriter.rewrite.await_count == 2
    asyncio.run(run())


def test_rerank_threshold_can_reject_every_candidate():
    async def run():
        item = chunk('dex', 'DEX definition')
        r = retriever([item], rag_rerank_enabled=True, rag_rerank_min_score=.7)
        r.reranker.rerank = AsyncMock(return_value=[(item,.2)])
        assert (await r.search('dex'))['status'] == 'empty'
    asyncio.run(run())


def test_rerank_failure_uses_rrf_and_top_k_budget():
    async def run():
        r = retriever([chunk('a', 'DEX one'), chunk('b','DEX two')], rag_rerank_enabled=True)
        r.reranker.rerank = AsyncMock(side_effect=TimeoutError)
        result = await r.search('dex', top_k=1)
        assert len(result['hits']) == 1
        assert result['retrieval']['degraded_stages'] == ['rerank']
    asyncio.run(run())


def test_cache_is_bounded_and_does_not_leak_mutations():
    cache = RetrievalCache(max_entries=1)
    cache.put('a', {'hits':[]}, 60)
    value = cache.get('a')
    value['hits'].append('bad')
    assert cache.get('a') == {'hits':[]}
    cache.put('b', {}, 60)
    assert cache.get('a') is None
    cache.put('c', {}, 0)
    assert cache.get('c') is None


def test_rewrite_retrieves_original_and_expansion_with_history():
    async def run():
        r = retriever([chunk('dex', 'DEX definition')], rag_rewrite_enabled=True)
        r.rewriter.rewrite = AsyncMock(return_value=['它是什么？', 'DEX definition'])
        result = await r.search('它是什么？', conversation_context='user: DEX')
        assert result['status'] == 'success'
        assert r.embedder.embed.await_count == 2
        r.rewriter.rewrite.assert_awaited_once_with('它是什么？', 'user: DEX')
        assert result['retrieval']['queries'] == ['它是什么？', 'DEX definition']
        assert len(result['hits']) == 1
    asyncio.run(run())


def test_reranker_http_contract_and_invalid_indices(monkeypatch):
    import httpx
    from web3_agent.rag.reranker import HttpReranker
    async def run():
        response_rows = [{'index': 0, 'score': .1}, {'index': 1, 'score': .9}]
        def handle(request):
            import json
            assert json.loads(request.content)['texts'] == ['first','second']
            return httpx.Response(200, json=response_rows)
        original = httpx.AsyncClient
        monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handle)))
        ranker = HttpReranker(Settings(_env_file=None, rag_rerank_url='http://reranker/rerank'))
        result = await ranker.rerank('question',[chunk('a','first'),chunk('b','second')])
        assert result[0][0].chunk_id == 'b'
        response_rows[1]['index'] = 0
        import pytest
        with pytest.raises(ValueError, match='invalid'):
            await ranker.rerank('question',[chunk('a','first'),chunk('b','second')])
    asyncio.run(run())


def test_chat_supplies_bounded_history_and_only_explicit_filters():
    from web3_agent.agent.graph.read_graph import initial_state
    from web3_agent.config.constants import Intent
    history = [('user', 'DEX ' + 'x'*3000), ('assistant', '解释DEX')]
    state = initial_state(history,'它的风险？',intent=Intent.QUERY_KNOWLEDGE,
                          slots={'chain_id':'通用','doc_category':'DEX'})
    assert len(state['rag_args']['conversation_context']) <= 2000
    # 会话链不参与过滤；用户明说的分类保留。
    assert 'chain_id' not in state['rag_args'] and 'doc_category' not in state['rag_args']
