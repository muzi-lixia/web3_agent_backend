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
from web3_agent.persistence.pg.orm_models import RagDocument as Document, RagChunk as Chunk, RagIngestionJob as Job, RagKnowledgeBase as KB, RagIndexVersion
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
                await conn.run_sync(lambda sync: RagIndexVersion.__table__.create(sync,checkfirst=False))
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


def test_changed_index_signature_rebuilds_chunks_in_place(monkeypatch):
    """内容没变、切块配置变了：切片必须换掉，而不是判成 unchanged 留着旧切片。"""
    async def flow(store):
        first = await publish(store, spec())
        assert first["chunks"] == 1
        # 同一份内容、同一版本，只有索引指纹变了。
        second = await store.publish(spec(), [TextChunk("退出", "解锁期为7天"), TextChunk("补充", "罚金说明")],
                                     [[1.0] + [0.0] * 1023, [0.0, 1.0] + [0.0] * 1022], "a", signature="new-signature")
        assert second["document_id"] == first["document_id"]
        assert second.get("rebuilt") and second["chunks"] == 2
        async with pg.session_scope() as db:
            chunks = (await db.execute(select(Chunk).where(Chunk.document_id == first["document_id"]))).scalars().all()
            document = await db.get(Document, first["document_id"])
        assert len(chunks) == 2
        assert document.index_signature == "new-signature"
    run_isolated(monkeypatch, flow)


def test_plan_index_skips_reembedding_only_when_content_and_signature_match(monkeypatch):
    async def flow(store):
        assert await store.plan_index(spec(), "a", "sig-1") == (store._identity(spec())[2], False)
        await publish(store, spec())
        document_id, unchanged = await store.plan_index(spec(), "a", "")
        assert document_id and unchanged
        # 指纹不同 → 需要重建；内容不同 → 也要重新走一遍，由 publish 报版本错误。
        assert not (await store.plan_index(spec(), "a", "sig-2"))[1]
        assert not (await store.plan_index(spec(), "other", ""))[1]
    run_isolated(monkeypatch, flow)


def test_index_version_tracks_publish_and_disable(monkeypatch):
    async def flow(store):
        assert await store.index_state() == (0, None)
        result = await publish(store, spec())
        assert (await store.index_state())[0] == 1
        # 同版本同内容重复发布是幂等的，不该让所有进程白重建一次索引。
        assert (await publish(store, spec()))["unchanged"]
        assert (await store.index_state())[0] == 1
        assert await store.disable(result["document_id"])
        assert (await store.index_state())[0] == 2
        assert not await store.disable(result["document_id"])
        assert (await store.index_state())[0] == 2
        until = datetime.now(timezone.utc) + timedelta(days=1)
        await publish(store, spec(pool="200", valid_until=until), body="到期文档")
        revision, boundary = await store.index_state(chain_id=1, protocol_id="200")
        assert revision == 3 and boundary == until
        assert await store.index_state(chain_id=1, protocol_id="100") == (3, None)
    run_isolated(monkeypatch, flow)


def test_snapshot_sql_never_selects_embedding_columns(monkeypatch):
    """BM25 一个向量数值都不用；把 1024 维列读出来只是白付传输与反序列化。"""
    from sqlalchemy import event
    async def flow(store):
        await publish(store, spec())
        statements = []
        engine = pg._sessionmaker.kw["bind"].sync_engine
        listener = lambda conn, cursor, statement, parameters, context, many: statements.append(statement)
        event.listen(engine, "before_cursor_execute", listener)
        try:
            chunks = await store.keyword_snapshot()
        finally:
            event.remove(engine, "before_cursor_execute", listener)
        assert [c.content for c in chunks] == ["解锁期为7天"]
        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        # 只关心向量列本身；embedding_model / embedding_version 是文档过滤条件，必须出现。
        assert selects and not any("rag_chunk.embedding" in s for s in selects)
    run_isolated(monkeypatch, flow)


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


def test_expired_payload_is_erased_and_retry_is_refused(monkeypatch):
    """原始文件留 3 天：到点抹掉正文，任务行留着；抹掉之后不许再重试。"""
    from web3_agent.rag import jobs as jobs_module
    from web3_agent.rag.jobs import PAYLOAD_RETENTION_DAYS, JobRepository
    from web3_agent.common.exceptions import AppError

    async def flow(store):
        job_id = str(uuid.uuid4())
        old = datetime.now(timezone.utc) - timedelta(days=PAYLOAD_RETENTION_DAYS + 1)
        async with pg.session_scope() as db:
            db.add(Job(id=job_id, kb_id="default", created_by="u1", fingerprint="f",
                       filename="a.md", payload=b"raw bytes", document_metadata={},
                       status="failed", stage="failed", error_code="INGESTION_FAILED",
                       attempts=1, total_chunks=0, embedded_chunks=0,
                       created_at=old, updated_at=old))
        monkeypatch.setattr(jobs_module, "_last_purge", 0.0)
        assert await JobRepository().claim(timeout_seconds=1, max_attempts=3) is None
        async with pg.session_scope() as db:
            assert (await db.execute(select(Job.payload).where(Job.id == job_id))).scalar_one() is None
        with pytest.raises(AppError) as error:
            await JobRepository().retry(job_id, 3)
        assert error.value.code == "E609"

    run_isolated(monkeypatch, flow)


def test_payload_inside_retention_window_is_kept(monkeypatch):
    """没到期的还能重试，不能顺手清掉。"""
    from web3_agent.rag import jobs as jobs_module
    from web3_agent.rag.jobs import JobRepository

    async def flow(store):
        job_id = str(uuid.uuid4())
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        async with pg.session_scope() as db:
            db.add(Job(id=job_id, kb_id="default", created_by="u1", fingerprint="f",
                       filename="a.md", payload=b"raw bytes", document_metadata={},
                       status="failed", stage="failed", error_code="INGESTION_FAILED",
                       attempts=1, total_chunks=0, embedded_chunks=0,
                       created_at=recent, updated_at=recent))
        monkeypatch.setattr(jobs_module, "_last_purge", 0.0)
        await JobRepository().claim(timeout_seconds=1, max_attempts=3)
        async with pg.session_scope() as db:
            assert (await db.execute(select(Job.payload).where(Job.id == job_id))).scalar_one() == b"raw bytes"

    run_isolated(monkeypatch, flow)
