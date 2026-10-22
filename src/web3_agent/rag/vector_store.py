"""PG/pgvector 仓储：链/协议/分类/版本过滤在 SQL 中执行，发布原子替换。"""
import hashlib
import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update, or_, text
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import RagDocument as Document, RagChunk as Chunk
from web3_agent.rag.models import Citation, DEFAULT_KB_ID, EmbeddingProfile
from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
from web3_agent.persistence.pg.orm_models import RagIngestionJob, RagKnowledgeBase as KB
from web3_agent.rag.jobs import LostClaim


def active_scope(chain_id=None, protocol_id=None, now=None, doc_category=None, kb_id=DEFAULT_KB_ID, profile=None):
    filters = [Document.kb_id == kb_id, Document.status == "active", Document.valid_from <= (now or datetime.now(timezone.utc)),
               or_(Document.valid_until.is_(None), Document.valid_until > (now or datetime.now(timezone.utc)))]
    # Explicit filters are exact: never substitute another chain/protocol/category.
    for field, value in ((Document.chain_id, chain_id), (Document.protocol_id, protocol_id),
                         (Document.doc_category, doc_category)):
        if value is not None:
            filters.append(field == value)
    if profile is not None:
        filters.extend(getattr(Document,key) == value for key,value in profile.model_dump().items())
    return tuple(filters)

class PgKnowledgeStore:
    def __init__(self, profile=None):
        self.profile = profile or EmbeddingProfile()
        self.knowledge_bases = KnowledgeBaseRepository(self.profile)

    async def validate_kb(self,kb_id):
        await self.knowledge_bases.require_active(kb_id)

    async def has_documents(self, chain_id=None, protocol_id=None, doc_category=None, kb_id=DEFAULT_KB_ID):
        await self.validate_kb(kb_id)
        async with session_scope() as db:
            query = select(Document.id).join(KB,KB.id==Document.kb_id).where(KB.status=="active",*active_scope(chain_id, protocol_id, doc_category=doc_category, kb_id=kb_id,profile=self.profile)).limit(1)
            return (await db.execute(query)).first() is not None

    async def search(self, vector, *, chain_id=None, protocol_id=None, doc_category=None, limit=3, max_distance=.42, kb_id=DEFAULT_KB_ID):
        await self.validate_kb(kb_id)
        distance = Chunk.embedding.cosine_distance(vector)
        query = (select(Chunk, Document, distance.label("distance"))
            .join(Document, Document.id == Chunk.document_id).join(KB,KB.id==Document.kb_id)
            .where(KB.status=="active", *active_scope(chain_id, protocol_id, doc_category=doc_category,kb_id=kb_id,profile=self.profile), distance <= max_distance)
            .order_by(distance, Chunk.id).limit(limit))
        async with session_scope() as db:
            rows = (await db.execute(query)).all()
            return [self._citation(chunk, document, float(score), index)
                    for index, (chunk, document, score) in enumerate(rows, 1)]

    @staticmethod
    def _citation(chunk, document, score=None, index=0):
        return Citation(
                id=f"S{index}", chunk_id=chunk.id, title=document.title, kb_id=document.kb_id,
                token_count=chunk.token_count, token_count_method=chunk.token_count_method,
                char_start=chunk.char_start,char_end=chunk.char_end,page_number=chunk.page_number,
                page_end=chunk.page_end,metadata=chunk.extra_metadata,
                doc_category=document.doc_category, valid_from=document.valid_from.isoformat(),
                section=chunk.section, source_url=document.source_url, version=document.version,
                chain_id=document.chain_id, protocol_id=document.protocol_id,
                updated_at=document.updated_at.isoformat(), content=chunk.content, distance=float(score) if score is not None else None,
        )

    async def keyword_snapshot(self, *, max_chunks=10000, **filters):
        """Read active scope every request, including on cache hits; no stale disabled docs.

        A bounded first implementation; over-cap scopes fail lexical recall explicitly
        instead of silently searching an arbitrary subset. No schema migration needed.
        """
        await self.validate_kb(filters.get("kb_id", DEFAULT_KB_ID))
        query = (select(Chunk, Document).join(Document, Document.id == Chunk.document_id)
                 .join(KB, KB.id == Document.kb_id)
                 .where(KB.status == "active", *active_scope(**filters, profile=self.profile))
                 .order_by(Chunk.id).limit(max_chunks+1))
        async with session_scope() as db:
            rows = (await db.execute(query)).all()
        if len(rows) > max_chunks:
            raise ValueError("keyword corpus exceeds configured bound")
        chunks = [self._citation(chunk, doc) for chunk, doc in rows]
        revision = hashlib.sha256(repr([(c.chunk_id, c.updated_at, c.version, c.title,
                                        c.content, c.section) for c in chunks]).encode()).hexdigest()
        return revision, chunks


    async def publish(self, spec, chunks, vectors, checksum, *, job_id=None, claim_token=None):
        if not spec.reviewed:
            raise ValueError("文档未经管理员审核，不能发布")
        if spec.valid_until is not None and spec.valid_until <= datetime.now(timezone.utc):
            raise ValueError("已过期文档不能发布")
        if any(len(vector) != 1024 or not all(math.isfinite(v) for v in vector)
               or not any(v != 0 for v in vector) for vector in vectors):
            raise ValueError("知识库向量必须为 1024 维")
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("切片与向量不能为空且数量必须一致")
        # Existing URL documents retain their identity. Local documents are grouped by title.
        source = str(spec.source_url)
        source_identity = source or f"local:{spec.title}"
        source_key = f"{spec.chain_id}|{spec.protocol_id}|{spec.doc_category}|{source_identity}"
        if spec.kb_id != DEFAULT_KB_ID:
            source_key = f"{spec.kb_id}|{source_key}"
        source_scope = [Document.kb_id == spec.kb_id, Document.chain_id == spec.chain_id, Document.protocol_id == spec.protocol_id,
                        Document.source_url == source, Document.doc_category == spec.doc_category,
                        Document.status == "active"]
        if not source:
            source_scope.append(Document.title == spec.title)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_key + "|" + spec.version))
        async with session_scope() as db:
            await self.knowledge_bases.require_active(spec.kb_id,db,lock=True)
            job = None
            if job_id is not None:
                job = (await db.execute(select(RagIngestionJob).where(
                    RagIngestionJob.id == job_id).with_for_update())).scalar_one_or_none()
                if job is None or job.kb_id != spec.kb_id or job.status != "processing" or job.claim_token != claim_token or job.lease_until <= datetime.now(timezone.utc):
                    raise LostClaim()
            def complete_job(count):
                if job is not None:
                    job.status = job.stage = "succeeded"
                    job.document_id = document_id
                    job.total_chunks = job.embedded_chunks = count
                    job.claim_token = job.lease_until = None
                    job.error_code = job.error_message = None
                    job.updated_at = datetime.now(timezone.utc)
            # 仅入库时持短事务锁；embedding 在事务外完成。
            await db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": source_key})
            existing = await db.get(Document, document_id)
            if existing:
                if existing.checksum != checksum:
                    raise ValueError("同一文档版本内容已改变，请提升 version 后发布")
                if any(getattr(existing,key) != value for key,value in self.profile.model_dump().items()):
                    raise ValueError("已有版本使用不同嵌入模型，请重新索引为新文档版本")
                if existing.status != "active":
                    raise ValueError("该版本已停用，请使用新版本号，避免覆盖当前版本")
                complete_job(len(chunks))
                return {"document_id": document_id, "chunks": 0, "unchanged": True}
            current = (await db.execute(select(Document).where(
                *source_scope,
            ))).scalars().all()
            if any(spec.valid_from < document.valid_from for document in current):
                raise ValueError("不能用更早生效的文档覆盖当前版本")
            # 到未来才生效的版本不能提前下线当前版本。
            if spec.valid_from > datetime.now(timezone.utc):
                raise ValueError("P0 仅支持已生效文档，未来版本请在生效后发布")
            await db.execute(update(Document).where(
                *source_scope,
            ).values(status="superseded"))
            db.add(Document(
                id=document_id, kb_id=spec.kb_id, **self.profile.model_dump(),
                chain_id=spec.chain_id, protocol_id=spec.protocol_id,
                title=spec.title, source_url=str(spec.source_url), doc_category=spec.doc_category,
                version=spec.version, checksum=checksum, status="active",
                valid_from=spec.valid_from, valid_until=spec.valid_until,
            ))
            await db.flush()
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                chunk_id = hashlib.sha256(f"{document_id}|{index}|{chunk.content}".encode()).hexdigest()
                db.add(Chunk(id=chunk_id, document_id=document_id, ordinal=index,
                             section=chunk.section, content=chunk.content, embedding=vector,
                             token_count=chunk.token_count,token_count_method=chunk.token_count_method,
                             char_start=chunk.char_start,char_end=chunk.char_end,page_number=chunk.page_number,
                             page_end=chunk.page_end,extra_metadata=chunk.metadata))
            complete_job(len(chunks))
        return {"document_id": document_id, "chunks": len(chunks), "unchanged": False}

    async def disable(self, document_id):
        # 保留来源与历史版本，但检索立即排除该版本。
        async with session_scope() as db:
            result = await db.execute(update(Document).where(Document.id == document_id).values(status="disabled"))
            return (result.rowcount or 0) > 0
