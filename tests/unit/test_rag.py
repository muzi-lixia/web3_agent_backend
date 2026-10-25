"""P0 RAG：入库、作用域、证据引用、写路径隔离与原工具兼容。"""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.dialects import postgresql
from sqlalchemy import select
from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.agent.intent_router import route_intent
from web3_agent.agent.agent_loop import _lacks_evidence
from web3_agent.common.types import LlmOutput, ToolCallRecord
from web3_agent.config.constants import Intent, Route, CallSource
from web3_agent.config.settings import Settings
from web3_agent.rag.models import Citation, DocumentSpec
from web3_agent.rag.chunker import split_document
from web3_agent.rag.config import normalize_filters
from web3_agent.rag.tool import build_rag_general_knowledge
from web3_agent.rag.retriever import KnowledgeRetriever, NO_DOCUMENTS, UNAVAILABLE, _select_evidence
from web3_agent.rag.ingestion_job import prepare, ingest
from web3_agent.rag.vector_store import active_scope
from web3_agent.persistence.pg.orm_models import RagDocument

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "rag"


def settings(**kwargs):
    return Settings(_env_file=None, rag_enabled=True, allowed_chain_ids=[1,56], **kwargs)



def citation(**kwargs):
    values=dict(id="S1",chunk_id="a"*64,title="测试规则",section="退出",source_url="https://example.com/rules",
        doc_category="stake",version="v1",chain_id=1,protocol_id="100",updated_at="2026-01-01T00:00:00Z",content="解锁期为7天。\n仅在锁定期内提前退出，罚金为2%。",distance=.1)
    values.update(kwargs)
    return Citation(**values)


def test_optional_filters_allow_global_and_independent_scope():
    assert normalize_filters() == dict(chain_id=None, protocol_id=None, doc_category=None,kb_id="default")
    assert normalize_filters("56", "dex_x", "swap") == dict(chain_id=56, protocol_id="dex_x", doc_category="swap",kb_id="default")


def test_chunking_retains_headings_overlap_and_long_content():
    text="# A\n"+"词语 "*1000+"\n# B\n最后一段"
    chunks=split_document(text,chunk_tokens=300)
    assert len(chunks)>1 and chunks[-1].section=="B"
    assert chunks[-1].content=="最后一段"
    assert all(chunk.content for chunk in chunks)
    assert split_document(" ")==[]
    with pytest.raises(ValueError):
        split_document(text,overlap_ratio=.9)


def test_manifest_validation_and_dry_run_do_not_embed():
    spec=DocumentSpec.model_validate(json.loads((FIXTURES/"manifest.json").read_text())[0])
    chunks,checksum,signature=prepare(spec,FIXTURES,settings())
    assert len(checksum)==64 and len(signature)==64 and chunks and not spec.reviewed
    with pytest.raises(ValueError):
        DocumentSpec.model_validate({**spec.model_dump(),"protocol_id":""})
    with pytest.raises(ValueError,match="时区"):
        DocumentSpec.model_validate({**spec.model_dump(),"valid_from":datetime(2026,1,1)})
    assert prepare(spec.model_copy(update={"chain_id":None,"protocol_id":None}),FIXTURES,settings())[0]


def test_publish_requires_review_before_model_or_database():
    spec=DocumentSpec.model_validate(json.loads((FIXTURES/"manifest.json").read_text())[0])
    embedder=SimpleNamespace(embed_many=AsyncMock())
    store=SimpleNamespace(publish=AsyncMock())
    with pytest.raises(ValueError,match="reviewed"):
        asyncio.run(ingest(spec,FIXTURES,settings(),embedder,store))
    embedder.embed_many.assert_not_awaited()
    store.publish.assert_not_awaited()


def test_ingestion_embeds_before_atomic_publication():
    spec=DocumentSpec.model_validate(json.loads((FIXTURES/"manifest.json").read_text())[0]).model_copy(update={"reviewed":True})
    async def embed(texts):
        return [[1.0]+[0.0]*1023 for _ in texts]
    store=SimpleNamespace(validate_kb=AsyncMock(),publish=AsyncMock(return_value={"chunks":3}),
                          plan_index=AsyncMock(return_value=("doc1",False)))
    result=asyncio.run(ingest(spec,FIXTURES,settings(),SimpleNamespace(embed_many=embed),store))
    assert result=={"chunks":3}
    assert len(store.publish.call_args.args[1])==len(store.publish.call_args.args[2])
    # 索引指纹必须跟着发布一起落库，否则下次无法判断切片是不是过期了。
    assert store.publish.call_args.kwargs["signature"]


def test_unchanged_document_skips_embedding_entirely():
    """同一份内容、同一套索引配置：一个向量都不该算。"""
    spec=DocumentSpec.model_validate(json.loads((FIXTURES/"manifest.json").read_text())[0]).model_copy(update={"reviewed":True})
    embedder=SimpleNamespace(embed_many=AsyncMock())
    store=SimpleNamespace(validate_kb=AsyncMock(),publish=AsyncMock(),
                          plan_index=AsyncMock(return_value=("doc1",True)))
    result=asyncio.run(ingest(spec,FIXTURES,settings(),embedder,store))
    assert result=={"document_id":"doc1","chunks":0,"unchanged":True}
    embedder.embed_many.assert_not_awaited()
    store.publish.assert_not_awaited()


def test_pg_query_filters_scope_version_and_expiry_before_limit():
    query=select(RagDocument).where(*active_scope(1,"100",datetime.now(timezone.utc)))
    compiled=query.compile(dialect=postgresql.dialect())
    assert "chain_id" in str(compiled) and "protocol_id" in str(compiled)
    assert "status" in str(compiled) and "valid_until" in str(compiled) and "valid_from" in str(compiled)
    assert 1 in compiled.params.values() and "100" in compiled.params.values()


@pytest.mark.parametrize("filters", [{"chain_id":0},{"chain_id":True},{"protocol_id":"bad / tag"},{"doc_category":""}])
def test_invalid_scope_never_queries_or_embeds(filters):
    embedder=SimpleNamespace(embed=AsyncMock())
    store=SimpleNamespace(has_documents=AsyncMock(),search=AsyncMock())
    result=asyncio.run(KnowledgeRetriever(settings(),embedder,store).search("规则",**filters))
    assert result["status"]=="invalid"
    embedder.embed.assert_not_awaited()
    store.search.assert_not_awaited()


@pytest.mark.parametrize("filters", [{},{"chain_id":56},{"protocol_id":"dex_x"},{"doc_category":"wallet"},
    {"chain_id":56,"protocol_id":"dex_x","doc_category":"swap"}])
def test_each_filter_combination_is_passed_without_implicit_wallet_chain(filters):
    store=SimpleNamespace(has_documents=AsyncMock(return_value=True),search=AsyncMock(return_value=[citation()]))
    embedder=SimpleNamespace(embed=AsyncMock(return_value=[0.0]*1024))
    result=asyncio.run(KnowledgeRetriever(settings(),embedder,store).search("什么是Gas",**filters))
    assert result["status"]=="success"
    expected=normalize_filters(**filters)
    assert store.has_documents.call_args.kwargs==expected
    assert all(store.search.call_args.kwargs[k]==v for k,v in expected.items())


def test_empty_corpus_skips_embedding_and_has_explicit_status():
    embedder=SimpleNamespace(embed=AsyncMock())
    store=SimpleNamespace(has_documents=AsyncMock(return_value=False),search=AsyncMock())
    result=asyncio.run(KnowledgeRetriever(settings(),embedder,store).search("解锁",chain_id=1,protocol_id="100"))
    assert result["status"]=="empty" and result["hits"]==[]
    embedder.embed.assert_not_awaited()


def test_retrieval_enforces_scope_limit_distance_and_returns_sources():
    store=SimpleNamespace(has_documents=AsyncMock(return_value=True),search=AsyncMock(return_value=[citation()]))
    embedder=SimpleNamespace(embed=AsyncMock(return_value=[0.0]*1024))
    result=asyncio.run(KnowledgeRetriever(settings(),embedder,store).search("多久解锁",chain_id=1,protocol_id="dex_x",doc_category="swap",top_k=999))
    assert result["hits"][0]["source_url"]=="https://example.com/rules"
    kwargs=store.search.call_args.kwargs
    assert kwargs["chain_id"]==1 and kwargs["protocol_id"]=="dex_x" and kwargs["doc_category"]=="swap" and kwargs["limit"]==3
    assert kwargs["max_distance"]==.42


def test_non_hybrid_hits_are_numbered_like_the_hybrid_path():
    store=SimpleNamespace(has_documents=AsyncMock(return_value=True),
        search=AsyncMock(return_value=[citation(chunk_id="a",content="第一段"),
                                       citation(chunk_id="b",content="第二段")]))
    embedder=SimpleNamespace(embed=AsyncMock(return_value=[0.0]*1024))
    result=asyncio.run(KnowledgeRetriever(settings(),embedder,store).search("解锁",chain_id=1))
    assert [hit["id"] for hit in result["hits"]]==["S1","S2"]


def test_evidence_selection_skips_oversized_chunks_and_stops_at_limit():
    chunks=[citation(chunk_id="a",content="x"*50),citation(chunk_id="b",content="yy"),
            citation(chunk_id="c",content="zz"),citation(chunk_id="d",content="ww")]
    selected=_select_evidence(chunks,max_chars=4,limit=2)
    assert [item["chunk_id"] for item in selected]==["b","c"]
    assert [item["id"] for item in selected]==["S1","S2"]


def test_store_failure_degrades_without_rules_or_sensitive_exception():
    store=SimpleNamespace(has_documents=AsyncMock(side_effect=RuntimeError("secret DB URL")))
    result=asyncio.run(KnowledgeRetriever(settings(),None,store).search("解锁",chain_id=1,protocol_id="100"))
    assert result["message"]==UNAVAILABLE and "secret" not in json.dumps(result)


def test_tool_exposes_optional_filters_and_allows_global_terms():
    retriever=SimpleNamespace(search=AsyncMock(return_value={"hits":[]}))
    tool=build_rag_general_knowledge(retriever)
    assert {"chain_id","protocol_id","doc_category"} <= tool.args.keys()
    asyncio.run(tool.ainvoke({"query_str":"什么是Gas"}))
    assert retriever.search.call_args.kwargs["chain_id"] is None
    asyncio.run(tool.ainvoke({"query_str":"滑点规则","chain_id":56,"protocol_id":"dex_x","doc_category":"swap"}))
    assert retriever.search.call_args.kwargs["chain_id"] == 56


def test_document_success_is_not_live_chain_evidence():
    records=[ToolCallRecord(name="rag_general_knowledge",ok=True)]
    assert _lacks_evidence(Intent.QUERY_STAKE,records)
    assert not _lacks_evidence(Intent.QUERY_KNOWLEDGE,records)


def test_knowledge_routes_read_and_stake_commands_still_route_write():
    knowledge=LlmOutput(intent=Intent.QUERY_KNOWLEDGE,confidence=.99,slots={"protocol_id":"100"})
    assert route_intent(knowledge,user_message="什么是Gas").route is Route.READ
    command=LlmOutput(intent=Intent.STAKE,confidence=.99,slots={"stakeContractAliasOrAddress":"100","tokenSymbol":"ETH","humanAmount":"0.1"})
    assert route_intent(command,user_message="质押 0.1 ETH 到 A 池").route is Route.WRITE


def test_graph_retrieves_with_the_original_query():
    seen=[]
    class Model:
        count=0
        async def ainvoke(self,messages):
            self.count+=1
            return AIMessage(content="解锁期为7天。[S1]")
    class Runner:
        async def run(self,name,args,source):
            seen.append(args)
            return json.dumps({"status":"success","hits":[citation().model_dump()]},ensure_ascii=False),ToolCallRecord(name=name,args=args,ok=True)
    state=initial_state([],"池100多久解锁",intent=Intent.QUERY_KNOWLEDGE)
    model=Model()
    result=asyncio.run(build_read_graph(None,Runner(),5,rag_answer_model=model).ainvoke(state))
    assert seen==[{"query_str":"池100多久解锁","kb_id":"default"}]
    assert model.count==1
    assert result["rag_citations"][0]["protocol_id"]=="100"
    assert result["rag_status"]=="success"


def test_general_knowledge_does_not_require_protocol_or_pool():
    output=LlmOutput(intent=Intent.QUERY_KNOWLEDGE,confidence=.99,slots={})
    assert route_intent(output,user_message="奖励怎么算").route is Route.READ


def test_agent_write_command_never_enters_rag_graph(monkeypatch):
    from tests.unit.test_agent_loop_evidence import _loop, _without_db, _Graph
    graph=_Graph("should not run")
    loop=_without_db(_loop(graph,Intent.STAKE),monkeypatch)
    loop._settings=settings()
    loop._llm.extract_intent=AsyncMock(return_value=LlmOutput(intent=Intent.STAKE,confidence=.99,slots={
        "stakeContractAliasOrAddress":"100","tokenSymbol":"ETH","humanAmount":"0.1"}))
    outcome=asyncio.run(loop.handle(siwe_user_id="u",wallet_address="0x"+"1"*40,chain_id=1,user_message="帮我质押0.1ETH到池100"))
    from web3_agent.config.constants import MSG_WRITE_NOT_ENABLED
    assert graph.calls==0 and outcome.content==MSG_WRITE_NOT_ENABLED and outcome.citations==[]


def test_agent_never_keeps_fabricated_rules_when_retrieval_failed(monkeypatch):
    from tests.unit.test_agent_loop_evidence import _loop, _without_db, _Graph
    graph=_Graph("提前赎回不收罚金，解锁期1天",[ToolCallRecord(name="rag_general_knowledge",ok=False,error="E602")])
    loop=_without_db(_loop(graph,Intent.QUERY_KNOWLEDGE),monkeypatch)
    loop._settings=settings()
    loop._llm.extract_intent=AsyncMock(return_value=LlmOutput(intent=Intent.QUERY_KNOWLEDGE,confidence=.99,slots={"protocol_id":"100"}))
    outcome=asyncio.run(loop.handle(siwe_user_id="u",wallet_address="0x"+"1"*40,chain_id=1,user_message="池100有罚金吗"))
    assert outcome.content==NO_DOCUMENTS and outcome.degraded and outcome.citations==[]


def test_agent_returns_rag_model_answer_with_sources_without_literal_validation(monkeypatch):
    from tests.unit.test_agent_loop_evidence import _loop, _without_db
    class Graph:
        async def ainvoke(self,state,config=None):
            return {"messages":[AIMessage(content="罚金为0%。[S1]")],"steps":2,
                "tool_calls":[ToolCallRecord(name="rag_general_knowledge",ok=True).model_dump()],
                "rag_status":"success","rag_citations":[citation().model_dump()]}
    loop=_without_db(_loop(Graph(),Intent.QUERY_KNOWLEDGE),monkeypatch)
    loop._settings=settings()
    loop._llm.extract_intent=AsyncMock(return_value=LlmOutput(intent=Intent.QUERY_KNOWLEDGE,confidence=.99,slots={"protocol_id":"100"}))
    outcome=asyncio.run(loop.handle(siwe_user_id="u",wallet_address="0x"+"1"*40,chain_id=1,user_message="池100有罚金吗"))
    assert outcome.content.startswith("罚金为0%。[S1]")
    assert not outcome.degraded and outcome.citations[0]["protocol_id"] == "100"


@pytest.mark.parametrize("enabled", [False, True])
def test_rag_toggle_preserves_three_original_tools(monkeypatch, enabled):
    from web3_agent import main
    from web3_agent.agent.tool_registry import ToolRegistry
    registered=[]
    class Registry(ToolRegistry):
        def register(self, tool, group, mutating):
            registered.append(tool.name)
            return super().register(tool, group, mutating)
    monkeypatch.setattr(main,"ToolRegistry",Registry)
    container=main.build_container(Settings(_env_file=None,rag_enabled=enabled,model_check_on_startup=False))
    assert registered[:3]==["get_asset_balance","get_transaction_records","get_stake_info"]
    # post_transfer 只进状态机组，不随 rag 开关变化。
    assert registered[3]=="post_transfer"
    assert registered[4:]==(["rag_general_knowledge"] if enabled else [])
    asyncio.run(container.adapter.aclose())


def test_missing_pool_is_a_router_clarification():
    output=LlmOutput(intent=Intent.QUERY_KNOWLEDGE,confidence=.99,slots={})
    decision=route_intent(output,user_message="这个池子多久解锁")
    assert decision.route is Route.READ and not decision.missing_slots


def test_question_tail_protocol_id_reaches_intent_model(monkeypatch):
    from tests.unit.test_agent_loop_evidence import _loop, _without_db, _Graph
    loop=_without_db(_loop(_Graph("unused"),Intent.QUERY_KNOWLEDGE),monkeypatch)
    loop._settings=settings()
    extract=AsyncMock(return_value=LlmOutput(intent=Intent.QUERY_KNOWLEDGE,confidence=.99,slots={}))
    loop._llm.extract_intent=extract
    text="请参考协议官方规则解释提前退出的各种限制条件，尤其是罚金和冷却期，池子ID是100"
    asyncio.run(loop.handle(siwe_user_id="u",wallet_address="0x"+"1"*40,chain_id=1,user_message=text))
    assert extract.call_args.args[1]==text


def test_empty_retrieval_ends_graph_without_model_inventing_rules():
    class Model:
        count=0
        async def ainvoke(self,messages):
            self.count+=1
            if self.count>1:
                raise AssertionError("空检索后不应再请求模型补全")
            return AIMessage(content="",tool_calls=[{"name":"rag_general_knowledge","args":{},"id":"x"}])
    model=Model()
    runner=SimpleNamespace(run=AsyncMock(return_value=(json.dumps({"error":"E602","status":"empty","hits":[]}),ToolCallRecord(name="rag_general_knowledge",ok=False,error="E602"))))
    state=initial_state([],"池100多久解锁",intent=Intent.QUERY_KNOWLEDGE)
    result=asyncio.run(build_read_graph(model,runner,5).ainvoke(state))
    assert model.count==0 and result["rag_status"]=="empty" and result["rag_citations"]==[]


def test_sql_global_scope_does_not_implicitly_filter_chain_or_protocol():
    from web3_agent.persistence.pg.orm_models import RagDocument
    global_query=str(select(RagDocument.id).where(*active_scope()).compile(dialect=postgresql.dialect()))
    assert "chain_id =" not in global_query and "protocol_id =" not in global_query
    scoped=select(RagDocument.id).where(*active_scope(56,"dex_x",doc_category="swap")).compile(dialect=postgresql.dialect())
    assert {56,"dex_x","swap"} <= set(scoped.params.values())
    assert "valid_until" in str(scoped)


def test_general_document_metadata_has_no_pool_requirement():
    values=dict(path="gas.md",title="Gas",source_url="https://example.com/gas",doc_category="term",
                version="v1",valid_from="2026-01-01T00:00:00Z")
    doc=DocumentSpec(**values)
    assert doc.chain_id is None and doc.protocol_id is None
    with pytest.raises(ValueError):
        DocumentSpec(**{**values,"doc_category":"bad category"})


def test_local_document_without_url_retains_citation_title_and_version():
    spec=DocumentSpec(path="local.md",title="内部知识",doc_category="faq",version="v1",valid_from="2026-01-01T00:00:00Z")
    assert spec.source_url == ""
    hits=[citation(source_url="").model_dump()]
    assert hits[0]["title"] == "测试规则" and hits[0]["version"] == "v1"


def test_chunk_provenance_is_exact_in_normalized_text_and_overlap():
    text="# 第一章\n"+"知识内容 "*700+"\n## 第二章\n钱包说明"
    chunks=split_document(text,chunk_tokens=400)
    assert all(text[c.char_start:c.char_end]==c.content for c in chunks)
    assert all(0<c.token_count<=400 for c in chunks)
    assert chunks[1].char_start<chunks[0].char_end
    assert chunks[-1].section=="第二章"


def test_query_always_has_a_library_and_profiles_filter_sql():
    from web3_agent.rag.models import EmbeddingProfile
    assert normalize_filters(kb_id="wallet-guides")["kb_id"]=="wallet-guides"
    with pytest.raises(ValueError): normalize_filters(kb_id="")
    compiled=select(RagDocument.id).where(*active_scope(kb_id="wallet-guides",profile=EmbeddingProfile())).compile(dialect=postgresql.dialect())
    assert "wallet-guides" in compiled.params.values() and "bge-m3" in compiled.params.values()
    assert "embedding_dimension" in str(compiled) and "embedding_version" in str(compiled)


def test_graph_cannot_switch_library_in_model_tool_arguments():
    seen=[]
    class Model:
        count=0
        async def ainvoke(self,messages):
            self.count+=1
            return AIMessage(content="",tool_calls=[{"name":"rag_general_knowledge","args":{"kb_id":"another-library"},"id":"kb"}]) if self.count==1 else AIMessage(content="解锁期为7天。[S1]")
    async def run(name,args,source):
        seen.append(args)
        return json.dumps({"status":"success","hits":[citation(kb_id="chosen-library").model_dump()]}),ToolCallRecord(name=name,args=args,ok=True)
    state=initial_state([],"规则",intent=Intent.QUERY_KNOWLEDGE,kb_id="chosen-library")
    model=Model()
    asyncio.run(build_read_graph(None,SimpleNamespace(run=run),5,rag_answer_model=model).ainvoke(state))
    assert seen[0]["kb_id"]=="chosen-library"
    assert len(seen)==1 and model.count==1


@pytest.mark.parametrize("state,model,expected",[(None,"bge-m3","E611"),("disabled","bge-m3","E611"),("active","other-model","E612"),("active","bge-m3",None)])
def test_real_knowledge_base_repository_guards_status_and_model(state,model,expected):
    from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
    from web3_agent.common.exceptions import AppError
    row=None if state is None else SimpleNamespace(status=state,embedding_provider="ollama",embedding_model=model,embedding_dimension=1024,embedding_version="v1")
    db=SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda:row)))
    if expected:
        with pytest.raises(AppError) as exc:
            asyncio.run(KnowledgeBaseRepository().require_active("default",db))
        assert exc.value.code==expected
    else:
        assert asyncio.run(KnowledgeBaseRepository().require_active("default",db)) is row


def test_index_signature_tracks_chunking_and_embedding_config():
    """内容没变但切块参数变了，签名必须变，否则旧切片会被当成 unchanged 留下。"""
    from web3_agent.rag.ingestion_job import index_signature
    checksum = "c" * 64
    base = index_signature(checksum, settings())
    assert base == index_signature(checksum, settings())
    assert index_signature(checksum, settings(rag_chunk_tokens=500)) != base
    assert index_signature(checksum, settings(rag_chunk_overlap=0.2)) != base
    assert index_signature(checksum, settings(rag_embedding_version="v2")) != base
    assert index_signature("d" * 64, settings()) != base
