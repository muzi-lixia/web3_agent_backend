"""Hybrid retrieval regressions: lexical rescue, scope revision, degradation, reranking."""
import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from web3_agent.config.settings import Settings
from web3_agent.rag.models import Citation
from web3_agent.rag.keyword_index import BM25Index
from web3_agent.rag.fusion import DENSE_WEIGHT, KEYWORD_WEIGHT, fuse
from web3_agent.rag.retriever import KnowledgeRetriever
from web3_agent.rag.cache import RetrievalCache


def chunk(key, text):
    return Citation(id='S1', chunk_id=key, title='FAQ', section='', source_url='',
                    version='v1', doc_category='faq', updated_at='2026-01-01', content=text, distance=2)


def retriever(chunks, **options):
    config = Settings(_env_file=None, rag_enabled=True, rag_hybrid_enabled=True, **options)
    store = SimpleNamespace(has_documents=AsyncMock(return_value=True),
        index_state=AsyncMock(return_value=(1, None)),
        keyword_snapshot=AsyncMock(return_value=chunks), search=AsyncMock(return_value=[]))
    client = SimpleNamespace(chat=AsyncMock(return_value='{"query": ""}'))
    return KnowledgeRetriever(config, SimpleNamespace(embed=AsyncMock(return_value=[1.])), store, client)


def test_cache_key_tracks_every_retrieval_setting():
    """缓存键取全部 rag_* 配置：新加的重排地址/向量版本也要让旧结果失效。"""
    async def run():
        item=chunk('dex','DEX definition')
        r=retriever([item])
        r.store.search.return_value=[item]
        await r.search('dex')
        assert (await r.search('dex'))['retrieval']['cache_hit'] is True
        r.settings.rag_rerank_url='http://rerank:8080/rerank'
        assert (await r.search('dex'))['retrieval']['cache_hit'] is False
    asyncio.run(run())


def test_lexical_casefold_and_chinese_matches():
    index = BM25Index([chunk('dex', '什么是DEX？去中心化交易所。'), chunk('gas', 'Gas 是手续费。')])
    assert index.search('什么是dex？')[0][0].chunk_id == 'dex'
    assert index.search('去中心化交易所')[0][0].chunk_id == 'dex'
    assert index.search('DEX') == index.search('dex')


def test_rrf_deduplicates_and_promotes_agreement():
    a, b, c = [chunk(k, k) for k in 'abc']
    result, _ = fuse([[a,b,b], [c,b]])
    assert [v.chunk_id for v in result] == ['b','a','c']


def test_keyword_route_weight_lowers_its_first_place_only():
    """关键词路第一名不该和向量路第一名同分：词面重合的块会把语义命中顶下去。"""
    a, d = chunk('a', 'a'), chunk('d', 'd')
    _, equal = fuse([[a], [d]])
    _, weighted = fuse([[a], [d]], weights=[DENSE_WEIGHT, KEYWORD_WEIGHT])
    assert equal['a'] == equal['d']
    assert weighted['a'] == equal['a'] > weighted['d']


def test_fusion_rejects_misaligned_weights():
    with pytest.raises(ValueError):
        fuse([[chunk('a', 'a')]], weights=[DENSE_WEIGHT, KEYWORD_WEIGHT])


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
        # 真实路径里 has_documents 会先拦住这种情况；这里只验证版本变化让缓存失效。
        r.store.index_state.return_value = (2, None)
        r.store.keyword_snapshot.return_value = []
        assert (await r.search('什么是dex？'))['status'] == 'empty'
        r.store.index_state.return_value = (3, None)
        r.store.keyword_snapshot.return_value = [item]
        await r.search('什么是dex？', kb_id='other')
        assert r.embedder.embed.await_count == 3
        assert r.store.index_state.call_args.kwargs['kb_id'] == 'other'
    asyncio.run(run())


def test_dense_and_rewrite_failure_preserve_keyword_evidence_with_short_cache():
    async def run():
        r = retriever([chunk('dex', 'DEX definition')], rag_rewrite_enabled=True)
        r.rewriter.rewrite = AsyncMock(side_effect=TimeoutError)
        r.embedder.embed.side_effect = TimeoutError
        result = await r.search('dex')
        assert result['status'] == 'success'
        assert set(result['retrieval']['degraded_stages']) == {'rewrite','dense'}
        await r.search('dex')
        assert r.rewriter.rewrite.await_count == 1
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


def test_index_cache_shares_the_object_instead_of_copying_it():
    """索引只读，共用同一个对象；递归复制等于每个请求铺一遍整个知识库。"""
    cache = RetrievalCache(max_entries=2, clone=False)
    index = BM25Index([chunk('a', 'DEX')])
    cache.put('index', index, 60)
    assert cache.get('index') is index


def test_index_is_built_once_and_reused_until_version_changes():
    async def run():
        item = chunk('dex', 'DEX definition')
        r = retriever([item])
        await r.search('dex')
        await r.search('what is dex')  # 不同 query：结果缓存不命中，但仍复用索引
        assert r.store.index_state.await_count == 2
        assert r.store.keyword_snapshot.await_count == 1
        r.store.index_state.return_value = (2, None)
        await r.search('dex definition')
        assert r.store.keyword_snapshot.await_count == 2
    asyncio.run(run())


def test_expiry_boundary_rebuilds_index_without_a_publish_event():
    """valid_until 到点不该等发布事件：边界一挪，索引键就变了。"""
    from datetime import datetime, timedelta, timezone
    async def run():
        r = retriever([chunk('dex', 'DEX definition')])
        r.store.index_state.return_value = (1, datetime.now(timezone.utc)+timedelta(days=1))
        await r.search('dex')
        assert r.store.keyword_snapshot.await_count == 1
        r.store.index_state.return_value = (1, None)
        await r.search('dex one')
        assert r.store.keyword_snapshot.await_count == 2
    asyncio.run(run())


def test_query_terms_are_tokenized_once_per_search(monkeypatch):
    import web3_agent.rag.keyword_index as keyword_index
    calls = []
    original = keyword_index.tokenize
    monkeypatch.setattr(keyword_index, 'tokenize',
                        lambda text: (calls.append(text), original(text))[1])
    index = BM25Index([chunk('a', 'DEX one'), chunk('b', 'DEX two')])
    calls.clear()
    index.search('DEX')
    assert calls == ['DEX']


def test_index_scans_only_chunks_that_contain_the_query_terms():
    index = BM25Index([chunk('a', 'DEX one'), chunk('b', 'gas fee'), chunk('c', 'DEX two')])
    assert list(index.postings['dex'][0]) == [0, 2]
    assert list(index.postings['fee'][0]) == [1]
    assert index.search('nothing here') == []
    assert [item.chunk_id for item, _ in index.search('dex')] == ['a', 'c']


def test_rewrite_retrieves_original_and_expansion_with_history():
    async def run():
        r = retriever([chunk('dex', 'DEX definition')], rag_rewrite_enabled=True)
        r.rewriter.rewrite = AsyncMock(return_value=('DEX definition', ['它是什么？', 'DEX definition']))
        result = await r.search('它是什么？', conversation_context='user: DEX')
        assert result['status'] == 'success'
        assert result['resolved_question'] == 'DEX definition'
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


def test_reranker_online_contracts(monkeypatch):
    import httpx
    from web3_agent.rag.reranker import HttpReranker
    original = httpx.AsyncClient
    def patch(handler):
        monkeypatch.setattr(httpx, 'AsyncClient',
                            lambda **kw: original(transport=httpx.MockTransport(handler)))
    async def run():
        import json
        seen = {}
        def cohere(request):
            seen['body'] = json.loads(request.content)
            seen['auth'] = request.headers.get('authorization')
            return httpx.Response(200, json={'results': [
                {'index': 0, 'relevance_score': .2}, {'index': 1, 'relevance_score': .8}]})
        patch(cohere)
        ranker = HttpReranker(Settings(_env_file=None, rag_rerank_url='http://rerank/v1/rerank',
                                       rag_rerank_provider='cohere', rag_rerank_api_key='k',
                                       rag_rerank_model='BAAI/bge-reranker-v2-m3'))
        result = await ranker.rerank('question', [chunk('a', 'first'), chunk('b', 'second')])
        assert seen['body']['documents'] == ['first', 'second']
        assert seen['body']['model'] == 'BAAI/bge-reranker-v2-m3'
        assert seen['auth'] == 'Bearer k'
        assert [c.chunk_id for c, _ in result] == ['b', 'a']
        def dashscope(request):
            seen['body'] = json.loads(request.content)
            return httpx.Response(200, json={'output': {'results': [
                {'index': 1, 'relevance_score': .9}, {'index': 0, 'relevance_score': .1}]}})
        patch(dashscope)
        ranker = HttpReranker(Settings(_env_file=None, rag_rerank_url='http://dash/rerank',
                                       rag_rerank_provider='dashscope', rag_rerank_api_key='k',
                                       rag_rerank_model='gte-rerank-v2'))
        result = await ranker.rerank('question', [chunk('a', 'first'), chunk('b', 'second')])
        assert seen['body']['input']['documents'] == ['first', 'second']
        assert seen['body']['parameters']['top_n'] == 2
        assert [c.chunk_id for c, _ in result] == ['b', 'a']
    asyncio.run(run())


def test_chat_supplies_bounded_history_and_no_extra_filters():
    from web3_agent.agent.graph.read_graph import initial_state
    from web3_agent.config.constants import Intent
    history = [('user', 'DEX ' + 'x'*3000), ('assistant', '解释DEX')]
    state = initial_state(history,'它的风险？',intent=Intent.QUERY_KNOWLEDGE)
    assert len(state['rag_args']['conversation_context']) <= 2000
    # 会话链不参与过滤，检索参数只有问题与知识库 id。
    assert 'chain_id' not in state['rag_args'] and 'doc_category' not in state['rag_args']


@pytest.mark.parametrize('failure_stage',['snapshot','scoring'])
def test_keyword_failure_falls_back_to_fresh_scoped_dense_query(monkeypatch,failure_stage):
    async def run():
        item=chunk('dex','DEX definition')
        r=retriever([item])
        r.store.search.return_value=[item]
        if failure_stage=='snapshot':
            r.store.keyword_snapshot.side_effect=RuntimeError('keyword unavailable')
        else:
            monkeypatch.setattr(BM25Index,'search',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('bad index')))
        result=await r.search('dex',kb_id='chosen',chain_id=56,protocol_id='dex_x')
        assert result['status']=='success' and result['hits'][0]['chunk_id']=='dex'
        assert result['retrieval']['degraded_stages']==['keyword']
        assert r.store.search.call_args.kwargs['kb_id']=='chosen'
        assert r.store.search.call_args.kwargs['chain_id']==56
        assert r.store.search.call_args.kwargs['protocol_id']=='dex_x'
        await r.search('dex',kb_id='chosen',chain_id=56,protocol_id='dex_x')
        assert r.embedder.embed.await_count==1
    asyncio.run(run())


def test_both_recall_routes_failed_is_unavailable_not_empty():
    async def run():
        r=retriever([])
        r.store.keyword_snapshot.side_effect=RuntimeError
        r.embedder.embed.side_effect=TimeoutError
        assert (await r.search('dex'))['status']=='unavailable'
    asyncio.run(run())


def test_hanging_dense_branch_keeps_keyword_hits_within_stage_budget():
    """慢故障：向量分支一直等下去，不能连带作废已经算出来的关键词结果。"""
    from time import monotonic
    async def run():
        cancelled=asyncio.Event()
        r=retriever([chunk('dex','DEX definition')],rag_timeout_seconds=.6)
        async def hang(_query):
            try:
                await asyncio.sleep(5)
            finally:
                cancelled.set()
        r.embedder.embed.side_effect=hang
        started=monotonic()
        result=await r.search('dex')
        assert monotonic()-started<.6
        assert result['status']=='success' and result['hits'][0]['chunk_id']=='dex'
        assert result['retrieval']['degraded_stages']==['dense_timeout']
        assert result['retrieval']['dense_timeout_branches']==1
        assert cancelled.is_set()
    asyncio.run(run())


def test_completed_dense_branch_survives_sibling_timeout():
    async def run():
        r=retriever([chunk('dex','DEX definition')],rag_rewrite_enabled=True,rag_timeout_seconds=.6)
        r.rewriter.rewrite=AsyncMock(return_value=('beta',['alpha','beta']))
        async def embed(query):
            if query=='beta':
                await asyncio.sleep(5)
            return [1.]
        r.embedder.embed.side_effect=embed
        r.store.search.return_value=[chunk('dex','DEX definition')]
        result=await r.search('alpha')
        assert result['status']=='success'
        assert [hit['chunk_id'] for hit in result['hits']]==['dex']
        assert result['retrieval']['degraded_stages']==['dense_timeout']
    asyncio.run(run())


def test_dense_timeout_without_lexical_evidence_is_unavailable():
    async def run():
        r=retriever([chunk('dex','DEX definition')],rag_timeout_seconds=.6)
        async def hang(_query):
            await asyncio.sleep(5)
        r.embedder.embed.side_effect=hang
        result=await r.search('zzz')
        assert result['status']=='unavailable' and result['error']=='E603'
    asyncio.run(run())


def test_disabled_knowledge_base_never_uses_dense_fallback():
    from web3_agent.common.exceptions import AppError, error_tuple
    async def run():
        r=retriever([])
        r.store.index_state.side_effect=AppError(*error_tuple('E611'))
        with pytest.raises(AppError) as error:
            await r.search('dex')
        assert error.value.code == 'E611'
        r.embedder.embed.assert_not_awaited()
        r.store.search.assert_not_awaited()
    asyncio.run(run())


def test_index_version_and_ttl_both_invalidate_cache(monkeypatch):
    """缓存依据是版本号，不是每次重算正文；版本或 TTL 变了才重新召回。"""
    import web3_agent.rag.cache as cache_module
    async def run():
        now=[100.]
        monkeypatch.setattr(cache_module,'monotonic',lambda:now[0])
        item=chunk('dex','DEX definition')
        r=retriever([item])
        assert (await r.search('dex'))['retrieval']['cache_hit'] is False
        assert (await r.search('dex'))['retrieval']['cache_hit'] is True
        assert r.embedder.embed.await_count==1
        # 发布/停用推进版本号：结果缓存与索引一起作废。
        r.store.index_state.return_value=(2,None)
        assert (await r.search('dex'))['retrieval']['cache_hit'] is False
        assert r.embedder.embed.await_count==2
        assert r.store.keyword_snapshot.await_count==2
        # 版本没变，只有 TTL 到期。
        now[0]+=601
        assert (await r.search('dex'))['retrieval']['cache_hit'] is False
        assert r.embedder.embed.await_count==3
    asyncio.run(run())


def test_index_version_failure_disables_cache_and_lexical_recall():
    """版本读不到就无法判断索引是否过期：只走向量，也不复用任何缓存。"""
    async def run():
        r=retriever([])
        r.store.index_state.side_effect=RuntimeError
        r.store.search.return_value=[chunk('dex','DEX definition')]
        await r.search('dex')
        await r.search('dex')
        assert r.embedder.embed.await_count==2
        r.store.keyword_snapshot.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize(('stage','code'),[('index','E611'),('dense','E611')])
def test_app_errors_are_not_hidden_as_recall_degradation(stage,code):
    from web3_agent.common.exceptions import AppError,error_tuple
    async def run():
        r=retriever([chunk('dex','DEX definition')])
        target={'index':r.store.index_state,'snapshot':r.store.keyword_snapshot,'dense':r.store.search}[stage]
        target.side_effect=AppError(*error_tuple(code))
        with pytest.raises(AppError) as error:
            await r.search('dex')
        assert error.value.code==code
        if stage=='index':
            r.embedder.embed.assert_not_awaited()
    asyncio.run(run())


def test_keyword_corpus_limit_raises_actionable_configuration_error(monkeypatch):
    import web3_agent.rag.vector_store as vector_store
    from contextlib import asynccontextmanager
    from web3_agent.common.exceptions import AppError
    async def run():
        @asynccontextmanager
        async def session():
            yield SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda:[None,None])))
        monkeypatch.setattr(vector_store,'session_scope',session)
        store=vector_store.PgKnowledgeStore()
        store.validate_kb=AsyncMock()
        with pytest.raises(AppError) as error:
            await store.keyword_snapshot(max_chunks=1)
        assert error.value.code=='E613'
    asyncio.run(run())


def test_keyword_corpus_limit_degrades_to_vector_recall():
    """切片数超过建索引上限只丢关键词这一路，整条查询照常返回向量结果。"""
    from web3_agent.common.exceptions import AppError,error_tuple
    async def run():
        item=chunk('dex','DEX definition')
        r=retriever([item])
        r.store.search.return_value=[item]
        r.store.keyword_snapshot.side_effect=AppError(*error_tuple('E613'))
        result=await r.search('dex')
        assert result['status']=='success'
        assert 'keyword' in result['retrieval']['degraded_stages']
        r.embedder.embed.assert_awaited()
    asyncio.run(run())


@pytest.mark.parametrize(('code','stage'), [('E611','dense')])
def test_actionable_errors_reach_http_response(code,stage):
    import httpx
    from fastapi import FastAPI
    from web3_agent.middlewares.error_handler_middleware import register_exception_handlers
    from web3_agent.common.exceptions import AppError,error_tuple
    async def run():
        r=retriever([chunk('dex','DEX definition')])
        target=r.store.search if stage=='dense' else r.store.keyword_snapshot
        target.side_effect=AppError(*error_tuple(code))
        app=FastAPI()
        register_exception_handlers(app)
        @app.get('/query')
        async def query():
            return await r.search('dex')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.get('/query')
        assert response.status_code==404
        assert response.json()['errorCode']==code
    asyncio.run(run())
