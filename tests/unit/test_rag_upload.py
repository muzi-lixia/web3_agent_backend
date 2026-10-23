"""HTTP → durable-job boundary → offline pipeline → retrieval, without production services."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from web3_agent.api.deps import require_identity
from web3_agent.api.v1.rag_router import router, public_router
from web3_agent.common.exceptions import AppError, error_tuple
from web3_agent.config.settings import Settings
from web3_agent.middlewares.error_handler_middleware import register_exception_handlers
from web3_agent.rag.document_loader import parse_document, DocumentParseError
from web3_agent.rag.worker import IngestionWorker
from web3_agent.rag.jobs import LostClaim, job_view
from web3_agent.rag.models import Citation
from web3_agent.rag.retriever import KnowledgeRetriever

ADDRESS = "0x"+"1"*40
METADATA = dict(title="Gas术语",source_url="https://example.com/gas",doc_category="term",
                version="v1",reviewed=True,valid_from="2026-01-01T00:00:00Z")


def settings(**kwargs):
    return Settings(_env_file=None,rag_enabled=True,rag_upload_enabled=True,rag_admin_addresses=[ADDRESS],**kwargs)


class KnowledgeBases:
    def __init__(self):
        self.rows={"default":dict(id="default",name="默认知识库",description="",status="active")}
    async def require_active(self,kb_id):
        if kb_id not in self.rows or self.rows[kb_id]["status"]!="active":
            raise AppError(*error_tuple("E611"))
    async def list(self,include_disabled=False):
        return [dict(v) for v in self.rows.values() if include_disabled or v["status"]=="active"]
    async def create(self,name,description,created_by):
        row=dict(id=str(uuid4()),name=name,description=description,status="active")
        self.rows[row["id"]]=row
        return dict(row)
    async def update(self,kb_id,values):
        self.rows[kb_id].update(values)
        return dict(self.rows[kb_id])


class Jobs:
    def __init__(self): self.rows={}; self.bases=KnowledgeBases()
    async def enqueue(self, *, spec, filename, data, **kwargs):
        await self.bases.require_active(spec.kb_id)
        job_id=str(uuid4())
        self.rows[job_id]=dict(job_id=job_id,id=job_id,kb_id=spec.kb_id,filename=filename,data=data,
            metadata=spec.model_dump(mode="json",exclude={"path"}), status="queued",stage="queued",
            token="claim",attempts=0,total_chunks=0,embedded_chunks=0,document_id=None,error=None)
        return self.view(job_id), False
    def view(self,job_id): return {k:v for k,v in self.rows[job_id].items() if k not in {"id","data","token"}}
    async def get(self,job_id): return self.view(job_id)
    async def claim(self,**kwargs):
        for row in self.rows.values():
            if row["status"]=="queued":
                row.update(status="processing",stage="loading",attempts=row["attempts"]+1)
                return deepcopy(row)
    async def progress(self,job_id,token,stage,**kwargs): self.rows[job_id].update(stage=stage,**kwargs)
    async def fail(self,job_id,token,code,message): self.rows[job_id].update(status="failed",error=dict(code=code,message=message))
    async def retry(self,job_id,max_attempts):
        self.rows[job_id].update(status="queued",stage="queued",error=None)
        return self.view(job_id)


class Store:
    def __init__(self,jobs): self.jobs,self.hits=jobs,[]
    async def validate_kb(self,kb_id):
        await self.jobs.bases.require_active(kb_id)
    async def publish(self,spec,chunks,vectors,checksum,**kwargs):
        await self.validate_kb(spec.kb_id)
        assert all(len(v)==1024 for v in vectors)
        self.hits.extend([Citation(id=f"S{i}",chunk_id=str(uuid4()),title=spec.title,kb_id=spec.kb_id,token_count=c.token_count,
            page_number=c.page_number,page_end=c.page_end,char_start=c.char_start,char_end=c.char_end,section=c.section,
            source_url=str(spec.source_url),version=spec.version,doc_category=spec.doc_category,
            chain_id=spec.chain_id,protocol_id=spec.protocol_id,updated_at="2026-01-01T00:00:00Z",
            content=c.content,distance=0) for i,c in enumerate(chunks,1)])
        self.jobs.rows[kwargs["job_id"]].update(status="succeeded",stage="succeeded",document_id=str(uuid4()))
    async def has_documents(self,**filters):
        await self.validate_kb(filters.get("kb_id","default"))
        return any(h.kb_id==filters.get("kb_id","default") for h in self.hits)
    async def search(self,vector,**kwargs): return [h for h in self.hits if h.kb_id==kwargs.get("kb_id","default")]


def app_fixture(config=None):
    config=config or settings()
    jobs=Jobs();store=Store(jobs)
    embedder=SimpleNamespace(embed_many=AsyncMock(side_effect=lambda texts:[[1.0]+[0.0]*1023 for _ in texts]),
                             embed=AsyncMock(return_value=[1.0]+[0.0]*1023))
    worker=IngestionWorker(config,embedder,store,jobs)
    app=FastAPI();register_exception_handlers(app);app.include_router(router,prefix="/api/v1");app.include_router(public_router,prefix="/api/v1")
    app.state.container=SimpleNamespace(settings=config,knowledge_bases=jobs.bases,rag_jobs=jobs,rag_store=store,
                                       rag_retriever=KnowledgeRetriever(config,embedder,store))
    app.dependency_overrides[require_identity]=lambda:SimpleNamespace(user_id="admin",address=ADDRESS)
    return app,worker,jobs,store,embedder


async def upload(client,content=b"# Gas\nGas is a unit of computation.",metadata=None,filename="gas.md"):
    import json
    return await client.post("/api/v1/rag/documents",files={"file":(filename,content,"text/markdown")},
                             data={"metadata":json.dumps(metadata or METADATA)})


@pytest.mark.parametrize("with_source", [False, True])
def test_frontend_upload_to_retrieval_without_second_processing_request(with_source):
    async def flow():
        app,worker,jobs,store,embedder=app_fixture()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            metadata = dict(METADATA)
            if not with_source:
                metadata.pop("source_url")
            response=await upload(client,metadata=metadata)
            assert response.status_code==202
            body=response.json();assert body["data"]["status"]=="queued"
            assert not store.hits and not embedder.embed_many.await_count
            assert await worker.run_once()
            state=(await client.get(body["status_url"])).json()["data"]
            assert state["status"]=="succeeded" and state["document_id"]
            found=await client.post("/api/v1/rag/search",json={"query_str":"What is Gas?"})
            hits=found.json()["data"]["hits"]
            assert found.status_code==200 and "unit of computation" in hits[0]["content"]
            assert bool(hits[0]["source_url"]) == with_source
            assert hits[0]["doc_category"]=="term" and hits[0]["chain_id"] is None
            assert "data" not in state and "payload" not in state
    asyncio.run(flow())


@pytest.mark.parametrize("metadata",[{**METADATA,"reviewed":False},{**METADATA,"path":"/etc/passwd"},
    {**METADATA,"doc_category":"bad category"},{**METADATA,"valid_from":"2035-01-01T00:00:00Z"},
    {**METADATA,"valid_until":"2026-01-02T00:00:00Z"}])
def test_upload_rejects_invalid_metadata_before_queue(metadata):
    async def flow():
        app,_,jobs,_,_=app_fixture()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            result=await upload(client,metadata=metadata)
            assert result.status_code==400 and result.json()["errorCode"]=="E607"
            assert not jobs.rows
    asyncio.run(flow())


def test_upload_auth_disabled_and_file_limits():
    async def flow():
        app,_,jobs,_,_=app_fixture(settings(rag_upload_max_bytes=1024))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            assert (await upload(client,content=b"x"*1025)).status_code==413
            assert (await upload(client,filename="a.exe")).status_code==400
            app.state.container.settings.rag_admin_addresses=[]
            assert (await upload(client)).status_code==403
            app.state.container.settings.rag_admin_addresses=[ADDRESS]
            app.state.container.settings.rag_upload_enabled=False
            assert (await upload(client)).status_code==503
            app.dependency_overrides.clear()
            assert (await upload(client)).status_code==401
            assert not jobs.rows
    asyncio.run(flow())


def test_chunked_request_body_limit_is_enforced_without_content_length():
    async def flow():
        app,_,jobs,_,_=app_fixture(settings(rag_upload_max_bytes=1024))
        async def body():
            yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
            for _ in range(70): yield b"x"*1024
            yield b"\r\n--boundary--\r\n"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            r=await client.post("/api/v1/rag/documents",content=body(),headers={"content-type":"multipart/form-data; boundary=boundary"})
            assert r.status_code==413 and r.json()["errorCode"]=="E604" and not jobs.rows
    asyncio.run(flow())


def test_parse_failure_and_retry_never_publish_partial_data():
    async def flow():
        app,worker,jobs,store,embedder=app_fixture()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            r=await upload(client,content=b"\xff\xfe")
            await worker.run_once()
            state=await jobs.get(r.json()["data"]["job_id"])
            assert state["status"]=="failed" and state["error"]["code"]=="PARSE_FAILED"
            assert not store.hits and not embedder.embed_many.await_count
            retry=await client.post(r.json()["status_url"]+"/retry")
            assert retry.status_code==202 and retry.json()["data"]["status"]=="queued"
    asyncio.run(flow())


def test_model_failure_is_safe_and_never_calls_publish():
    async def flow():
        app,worker,jobs,store,embedder=app_fixture()
        embedder.embed_many.side_effect=RuntimeError("secret model URL")
        store.publish=AsyncMock()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            r=await upload(client)
            await worker.run_once()
            state=await jobs.get(r.json()["data"]["job_id"])
            assert state["status"]=="failed" and "secret" not in str(state["error"])
            store.publish.assert_not_awaited()
    asyncio.run(flow())


def pdf_bytes(text="Gas is computation",encrypted=False):
    writer=PdfWriter();page=writer.add_blank_page(width=300,height=300)
    font=DictionaryObject({NameObject("/Type"):NameObject("/Font"),NameObject("/Subtype"):NameObject("/Type1"),NameObject("/BaseFont"):NameObject("/Helvetica")})
    page[NameObject("/Resources")]=DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):font})})
    stream=DecodedStreamObject();stream.set_data(f"BT /F1 12 Tf 10 10 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")]=writer._add_object(stream)
    if encrypted: writer.encrypt("password")
    target=BytesIO();writer.write(target);return target.getvalue()


def test_real_pdf_parser_text_and_encrypted_input():
    assert "Gas is computation" in parse_document(pdf_bytes(),"x.pdf")
    with pytest.raises(DocumentParseError,match="加密"):
        parse_document(pdf_bytes(encrypted=True),"x.pdf")
    with pytest.raises(DocumentParseError,match="文件头"):
        parse_document(b"hello","x.pdf")
    assert parse_document(b"\xef\xbb\xbf# Gas\r\ntext\x00", "gas.md")=="# Gas\ntext"


def test_upload_to_two_libraries_is_scoped_and_disabled_library_rejects_new_files():
    async def flow():
        app,worker,jobs,store,_=app_fixture()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            created=await client.post("/api/v1/rag/knowledge-bases",json={"name":"钱包资料","description":"指南"})
            assert created.status_code==201
            kb_id=created.json()["data"]["id"]
            r=await upload(client,content=b"Default library fact.")
            assert r.json()["data"]["kb_id"]=="default"
            await worker.run_once()
            r=await upload(client,content=b"Wallet library fact.",metadata={**METADATA,"kb_id":kb_id})
            await worker.run_once()
            assert r.json()["data"]["kb_id"]==kb_id
            result=(await client.post("/api/v1/rag/search",json={"query_str":"fact","kb_id":kb_id})).json()["data"]
            assert len(result["hits"])==1 and result["hits"][0]["content"]=="Wallet library fact."
            result=(await client.post("/api/v1/rag/search",json={"query_str":"fact"})).json()["data"]
            assert len(result["hits"])==1 and result["hits"][0]["content"]=="Default library fact."
            assert (await client.patch(f"/api/v1/rag/knowledge-bases/{kb_id}",json={"status":"disabled"})).status_code==200
            result=await upload(client,metadata={**METADATA,"kb_id":kb_id})
            assert result.status_code==404 and result.json()["errorCode"]=="E611"
            visible=(await client.get("/api/v1/knowledge-bases")).json()["data"]["items"]
            assert all(k["id"]!=kb_id for k in visible)
    asyncio.run(flow())


def test_invalid_library_and_embedding_configuration_cannot_reach_embedding():
    async def flow():
        app,worker,jobs,_,embedder=app_fixture()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            result=await upload(client,metadata={**METADATA,"kb_id":"missing"})
            assert result.status_code==404 and not jobs.rows
            accepted=await upload(client)
            worker.store.validate_kb=AsyncMock(side_effect=AppError(*error_tuple("E612")))
            await worker.run_once()
            row=await jobs.get(accepted.json()["data"]["job_id"])
            assert row["status"]=="failed" and row["error"]["code"]=="E612"
            embedder.embed_many.assert_not_awaited()
    asyncio.run(flow())


def test_pdf_pages_and_offsets_survive_pipeline():
    from pypdf import PdfReader
    from web3_agent.rag.document_loader import parse_document_with_spans
    from web3_agent.rag.ingestion_job import prepare_content
    from web3_agent.rag.models import DocumentSpec
    writer=PdfWriter()
    for text in ("First page explains Gas.","Second page explains wallets."):
        writer.add_page(PdfReader(BytesIO(pdf_bytes(text))).pages[0])
    target=BytesIO();writer.write(target)
    parsed=parse_document_with_spans(target.getvalue(),"pages.pdf")
    assert [p[2] for p in parsed.page_spans]==[1,2]
    spec=DocumentSpec(path="pages.pdf",**METADATA)
    chunks,_=prepare_content(spec,parsed,settings())
    assert chunks[0].page_number==1 and chunks[-1].page_end==2
    assert all(parsed.text[c.char_start:c.char_end]==c.content for c in chunks)
    assert all(c.token_count>0 and c.token_count_method=="approx_v1" for c in chunks)


@pytest.mark.parametrize("filename", ["guide.md", "guide.txt"])
def test_normal_english_document_is_not_rejected_as_mnemonic(filename):
    text = "this document explains how users can safely manage their digital assets across different networks"
    assert parse_document(text.encode(), filename) == text


def test_transaction_hash_in_document_is_not_rejected_as_private_key():
    text = "Transaction hash: 0x" + "ab" * 32
    assert parse_document(text.encode(), "transaction.md") == text
