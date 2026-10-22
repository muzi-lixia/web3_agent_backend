"""显式 TEST_RAG_DATABASE_URL 才运行；临时 schema 隔离，不探测业务数据库。"""
import asyncio
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from web3_agent.persistence.pg import base as pg
from web3_agent.persistence.pg.orm_models import RagDocument as Document, RagChunk as Chunk, RagIngestionJob as Job, RagKnowledgeBase as KB
from web3_agent.rag.models import DocumentSpec
from web3_agent.rag.chunker import Chunk as TextChunk
from web3_agent.rag.vector_store import PgKnowledgeStore

pytestmark=pytest.mark.skipif(not os.environ.get("TEST_RAG_DATABASE_URL"),reason="需要显式配置专用 TEST_RAG_DATABASE_URL")


def run_isolated(monkeypatch,flow):
    async def run():
        schema="test_stake_rag_"+uuid.uuid4().hex
        admin=create_async_engine(os.environ["TEST_RAG_DATABASE_URL"])
        engine=None
        try:
            async with admin.begin() as conn:
                await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            engine=create_async_engine(os.environ["TEST_RAG_DATABASE_URL"],connect_args={"server_settings":{"search_path":f"{schema},public"}})
            monkeypatch.setattr(pg,"_sessionmaker",async_sessionmaker(engine,expire_on_commit=False))
            async with engine.begin() as conn:
                # 强制在测试 schema 建表，避免 checkfirst 命中 public 的同名表。
                await conn.run_sync(lambda sync: KB.__table__.create(sync,checkfirst=False))
                await conn.run_sync(lambda sync: Document.__table__.create(sync,checkfirst=False))
                await conn.run_sync(lambda sync: Chunk.__table__.create(sync,checkfirst=False))
                await conn.run_sync(lambda sync: Job.__table__.create(sync,checkfirst=False))
            from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
            await KnowledgeBaseRepository().ensure_default()
            await flow(PgKnowledgeStore())
        finally:
            if engine:
                await engine.dispose()
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await admin.dispose()
    asyncio.run(run())


def spec(pool="100",chain=1,version="v1",**kwargs):
    values=dict(path="test.md",title="测试",source_url="https://example.com/rules",chain_id=chain,protocol_id=pool,
        doc_category="stake",version=version,reviewed=True,valid_from=datetime.now(timezone.utc)-timedelta(days=1))
    values.update(kwargs)
    return DocumentSpec(**values)


def publish(store,document,body="解锁期为7天",checksum="a"):
    return store.publish(document,[TextChunk("退出",body)],[[1.0]+[0.0]*1023],checksum)


def test_scope_and_cosine_threshold(monkeypatch):
    async def flow(store):
        await publish(store,spec())
        await publish(store,spec(pool="200"),body="其他池")
        await publish(store,spec(chain=56),body="其他链")
        hits=await store.search([1.0]+[0.0]*1023,chain_id=1,protocol_id="100",limit=3,max_distance=.42)
        assert len(hits)==1 and hits[0].content=="解锁期为7天"
        assert await store.search([0.0,1.0]+[0.0]*1022,chain_id=1,protocol_id="100",limit=3,max_distance=.42)==[]
    run_isolated(monkeypatch,flow)


def test_atomic_new_version_and_idempotency(monkeypatch):
    async def flow(store):
        first=await publish(store,spec())
        assert (await publish(store,spec()))["unchanged"]
        await publish(store,spec(version="v2"),body="解锁期为14天",checksum="b")
        hits=await store.search([1.0]+[0.0]*1023,chain_id=1,protocol_id="100",limit=3,max_distance=.42)
        assert len(hits)==1 and hits[0].version=="v2" and "14" in hits[0].content
        async with pg.session_scope() as db:
            assert (await db.get(Document,first["document_id"])).status=="superseded"
    run_isolated(monkeypatch,flow)


def test_changed_content_requires_new_version(monkeypatch):
    async def flow(store):
        await publish(store,spec())
        with pytest.raises(ValueError,match="version"):
            await publish(store,spec(),body="错误替换",checksum="different")
        hits=await store.search([1.0]+[0.0]*1023,chain_id=1,protocol_id="100",limit=3,max_distance=.42)
        assert hits[0].content=="解锁期为7天"
    run_isolated(monkeypatch,flow)


def test_disabled_and_expired_documents_are_not_retrieved(monkeypatch):
    async def flow(store):
        result=await publish(store,spec())
        assert await store.disable(result["document_id"])
        assert not await store.has_documents(1,"100")
        with pytest.raises(ValueError,match="过期"):
            await publish(store,spec(pool="200",valid_until=datetime.now(timezone.utc)-timedelta(hours=1)))
        assert not await store.has_documents(1,"200")
    run_isolated(monkeypatch,flow)


def test_global_terms_and_categories_are_independent_scopes(monkeypatch):
    async def flow(store):
        await publish(store,spec(chain=None,pool=None,doc_category="term"),body="Gas术语说明")
        await publish(store,spec(chain=56,pool="dex_x",doc_category="swap"),body="兑换规则")
        vector=[1.0]+[0.0]*1023
        assert len(await store.search(vector,limit=5))==2
        hits=await store.search(vector,doc_category="term")
        assert len(hits)==1 and hits[0].chain_id is None and hits[0].protocol_id is None
        hits=await store.search(vector,chain_id=56,protocol_id="dex_x",doc_category="swap")
        assert len(hits)==1 and hits[0].content=="兑换规则"
        assert await store.search(vector,chain_id=1,protocol_id="dex_x")==[]
    run_isolated(monkeypatch,flow)


def test_global_document_version_replacement(monkeypatch):
    async def flow(store):
        await publish(store,spec(chain=None,pool=None,doc_category="wallet"))
        await publish(store,spec(chain=None,pool=None,doc_category="wallet",version="v2"),body="钱包教程新版",checksum="b")
        hits=await store.search([1.0]+[0.0]*1023,doc_category="wallet")
        assert len(hits)==1 and hits[0].version=="v2"
    run_isolated(monkeypatch,flow)


def test_upload_idempotency_and_concurrent_claims(monkeypatch):
    from web3_agent.rag.jobs import JobRepository
    from web3_agent.common.exceptions import AppError
    async def flow(store):
        jobs=JobRepository()
        args=dict(spec=spec(),filename="test.md",data=b"hello",created_by="admin",idempotency_key="request1")
        first,duplicate=await jobs.enqueue(**args)
        assert not duplicate
        second,duplicate=await jobs.enqueue(**args)
        assert duplicate and second["job_id"]==first["job_id"]
        with pytest.raises(AppError):
            await jobs.enqueue(**{**args,"data":b"different"})
        claims=await asyncio.gather(jobs.claim(timeout_seconds=600,max_attempts=3),jobs.claim(timeout_seconds=600,max_attempts=3))
        assert sum(job is not None for job in claims)==1
        assert (await jobs.get(first["job_id"]))["attempts"]==1
    run_isolated(monkeypatch,flow)


def test_vectors_and_job_success_commit_together_and_stale_claim_cannot_publish(monkeypatch):
    from web3_agent.rag.jobs import JobRepository, LostClaim
    async def flow(store):
        jobs=JobRepository()
        upload,_=await jobs.enqueue(spec=spec(),filename="test.md",data=b"hello",created_by="admin")
        job=await jobs.claim(timeout_seconds=600,max_attempts=3)
        with pytest.raises(LostClaim):
            await store.publish(spec(),[TextChunk("标题","内容")],[[1.0]+[0.0]*1023],"a",
                job_id=job["id"],claim_token="stale-token")
        assert not await store.has_documents(1,"100")
        result=await store.publish(spec(),[TextChunk("标题","内容")],[[1.0]+[0.0]*1023],"a",
            job_id=job["id"],claim_token=job["token"])
        state=await jobs.get(upload["job_id"])
        assert state["status"]=="succeeded" and state["document_id"]==result["document_id"]
        assert state["total_chunks"]==state["embedded_chunks"]==1
        assert await store.has_documents(1,"100")
    run_isolated(monkeypatch,flow)


def test_interrupted_job_recovery_and_bounded_retry(monkeypatch):
    from sqlalchemy import update
    from web3_agent.rag.jobs import JobRepository
    from web3_agent.common.exceptions import AppError
    async def flow(store):
        jobs=JobRepository()
        upload,_=await jobs.enqueue(spec=spec(),filename="test.md",data=b"hello",created_by="admin")
        first=await jobs.claim(timeout_seconds=600,max_attempts=3)
        async with pg.session_scope() as db:
            await db.execute(update(Job).where(Job.id==upload["job_id"]).values(lease_until=datetime.now(timezone.utc)-timedelta(seconds=1)))
        reclaimed=await jobs.claim(timeout_seconds=600,max_attempts=3)
        assert first["id"]==reclaimed["id"] and first["token"]!=reclaimed["token"]
        await jobs.fail(reclaimed["id"],reclaimed["token"],"TEST","失败")
        assert (await jobs.retry(reclaimed["id"],3))["status"]=="queued"
        third=await jobs.claim(timeout_seconds=600,max_attempts=3)
        await jobs.fail(third["id"],third["token"],"TEST","失败")
        with pytest.raises(AppError): await jobs.retry(third["id"],3)
    run_isolated(monkeypatch,flow)


def test_local_documents_without_urls_do_not_replace_other_titles(monkeypatch):
    async def flow(store):
        await publish(store,spec(source_url="",title="本地A"),body="A旧版")
        await publish(store,spec(source_url="",title="本地B"),body="B内容")
        await publish(store,spec(source_url="",title="本地A",version="v2"),body="A新版",checksum="b")
        hits=await store.search([1.0]+[0.0]*1023,chain_id=1,protocol_id="100",limit=5)
        assert {(hit.title,hit.version) for hit in hits} == {("本地A","v2"),("本地B","v1")}
        assert all(hit.source_url == "" for hit in hits)
    run_isolated(monkeypatch,flow)


def test_library_isolation_disabled_state_and_embedding_profile(monkeypatch):
    from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
    from web3_agent.rag.models import EmbeddingProfile
    from web3_agent.common.exceptions import AppError
    async def flow(store):
        bases=KnowledgeBaseRepository()
        other=await bases.create("其他知识库","隔离测试","admin")
        await publish(store,spec(),body="默认库")
        await publish(store,spec(kb_id=other["id"]),body="其他库")
        vector=[1.0]+[0.0]*1023
        assert [h.content for h in await store.search(vector)]==["默认库"]
        assert [h.content for h in await store.search(vector,kb_id=other["id"])]==["其他库"]
        with pytest.raises(AppError) as exc:
            await PgKnowledgeStore(EmbeddingProfile(embedding_model="another-model")).has_documents()
        assert exc.value.code=="E612"
        await bases.update(other["id"],{"status":"disabled"})
        with pytest.raises(AppError) as exc:
            await store.search(vector,kb_id=other["id"])
        assert exc.value.code=="E611"
        with pytest.raises(AppError):
            await publish(store,spec(kb_id=other["id"],version="v2"))
        assert len(await store.search(vector))==1
    run_isolated(monkeypatch,flow)


def test_same_document_identity_is_unique_per_library_and_provenance_persisted(monkeypatch):
    from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
    async def flow(store):
        other=await KnowledgeBaseRepository().create("第二库","","admin")
        chunk=TextChunk("章节","PDF原文",token_count=5,char_start=0,char_end=5,page_number=2,page_end=3,metadata={"risk_level":"high"})
        a=await store.publish(spec(),[chunk],[[1.0]+[0.0]*1023],"a")
        b=await store.publish(spec(kb_id=other["id"]),[chunk],[[1.0]+[0.0]*1023],"a")
        assert a["document_id"]!=b["document_id"]
        hits=await store.search([1.0]+[0.0]*1023,kb_id=other["id"])
        assert len(hits)==1 and hits[0].page_number==2 and hits[0].page_end==3
        assert hits[0].metadata["risk_level"]=="high" and hits[0].token_count==5
    run_isolated(monkeypatch,flow)
