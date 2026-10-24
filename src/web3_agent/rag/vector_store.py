"""PG/pgvector 仓储：链/协议/分类/版本过滤在 SQL 中执行，发布原子替换。"""
import hashlib
import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Load
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import RagDocument as Document, RagChunk as Chunk
from web3_agent.persistence.pg.orm_models import RagIndexVersion
from web3_agent.rag.models import Citation, DEFAULT_KB_ID, EmbeddingProfile
from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
from web3_agent.persistence.pg.orm_models import RagIngestionJob, RagKnowledgeBase as KB
from web3_agent.rag.jobs import LostClaim
from web3_agent.common.exceptions import AppError, error_tuple


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
    # 建索引只用到这些列。select(Chunk) 会把 1024 维向量一起读出来 ——
    # BM25 一个数值都不用，白付整表的传输与反序列化。
    _CHUNK_COLUMNS = (Chunk.id, Chunk.section, Chunk.content, Chunk.token_count,
                      Chunk.token_count_method, Chunk.char_start, Chunk.char_end,
                      Chunk.page_number, Chunk.page_end, Chunk.extra_metadata)
    _DOCUMENT_COLUMNS = (Document.id, Document.kb_id, Document.title, Document.source_url,
                         Document.version, Document.doc_category, Document.valid_from,
                         Document.chain_id, Document.protocol_id, Document.updated_at)

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

    async def search(self, vector, *, chain_id=None, protocol_id=None, doc_category=None, limit=3, max_distance=.42, kb_id=DEFAULT_KB_ID, validate=True):
        if validate:
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

    async def index_state(self, *, validate=True, **filters):
        """关键词索引的版本与到期边界；一条单行查询，不读任何正文。

        revision 由发布/停用事件推进（见 _bump_index）。boundary 是当前有效集里
        最早的 valid_until：时间越过它，同一句 SQL 就会返回下一个边界，索引键随之
        改变，索引在下次查询时自然重建 —— 不需要定时任务，也不会用到过期文档。
        """
        kb_id = filters.get("kb_id", DEFAULT_KB_ID)
        if validate:
            await self.validate_kb(kb_id)
        now = datetime.now(timezone.utc)
        revision = (select(RagIndexVersion.revision)
                    .where(RagIndexVersion.kb_id == kb_id).scalar_subquery())
        boundary = (select(func.min(Document.valid_until))
                    .where(*active_scope(**filters, now=now, profile=self.profile))
                    .scalar_subquery())
        async with session_scope() as db:
            row = (await db.execute(select(revision, boundary))).one()
        return int(row[0] or 0), row[1]

    async def keyword_snapshot(self, *, max_chunks=10000, validate=True, **filters):
        """读取一次有效切片用于建索引；调用方只在版本变化时才会走到这里。"""
        if validate:
            await self.validate_kb(filters.get("kb_id", DEFAULT_KB_ID))
        query = (select(Chunk, Document)
                 .options(Load(Chunk).load_only(*self._CHUNK_COLUMNS),
                          Load(Document).load_only(*self._DOCUMENT_COLUMNS))
                 .join(Document, Document.id == Chunk.document_id)
                 .join(KB, KB.id == Document.kb_id)
                 .where(KB.status == "active", *active_scope(**filters, profile=self.profile))
                 .order_by(Chunk.id).limit(max_chunks+1))
        async with session_scope() as db:
            rows = (await db.execute(query)).all()
        if len(rows) > max_chunks:
            raise AppError(*error_tuple("E613"))
        return [self._citation(chunk, doc) for chunk, doc in rows]

    async def _bump_index(self, db, kb_id):
        """发布或停用后推进版本，下一次查询就会重建索引。"""
        await db.execute(insert(RagIndexVersion).values(kb_id=kb_id, revision=1)
                         .on_conflict_do_update(index_elements=[RagIndexVersion.kb_id],
                                                set_={"revision": RagIndexVersion.revision + 1,
                                                      "updated_at": func.now()}))


    @staticmethod
    def _chunk_row(document_id, index, chunk, vector):
        return Chunk(id=hashlib.sha256(f"{document_id}|{index}|{chunk.content}".encode()).hexdigest(),
                     document_id=document_id, ordinal=index, section=chunk.section,
                     content=chunk.content, embedding=vector, token_count=chunk.token_count,
                     token_count_method=chunk.token_count_method, char_start=chunk.char_start,
                     char_end=chunk.char_end, page_number=chunk.page_number,
                     page_end=chunk.page_end, extra_metadata=chunk.metadata)

    @staticmethod
    def _identity(spec):
        """文档身份：URL 文档按来源，本地文档按标题；同名同范围才算同一份。"""
        source = str(spec.source_url)
        source_identity = source or f"local:{spec.title}"
        source_key = f"{spec.chain_id}|{spec.protocol_id}|{spec.doc_category}|{source_identity}"
        if spec.kb_id != DEFAULT_KB_ID:
            source_key = f"{spec.kb_id}|{source_key}"
        scope = [Document.kb_id == spec.kb_id, Document.chain_id == spec.chain_id,
                 Document.protocol_id == spec.protocol_id, Document.source_url == source,
                 Document.doc_category == spec.doc_category, Document.status == "active"]
        if not source:
            scope.append(Document.title == spec.title)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_key + "|" + spec.version))
        return source_key, tuple(scope), document_id

    async def plan_index(self, spec, checksum, signature):
        """发布前预判：同一份内容、同一套索引配置就不必再向量化一遍。

        返回 (document_id, unchanged)。这里只是省成本，真正的一致性由 publish
        在事务内带着行锁再判一次 —— 两次判断之间别人发布了同版本也不算错。
        """
        _, _, document_id = self._identity(spec)
        async with session_scope() as db:
            existing = await db.get(Document, document_id)
        unchanged = (
            existing is not None
            and existing.status == "active"
            and existing.checksum == checksum
            and (existing.index_signature or "") == signature
        )
        return document_id, unchanged

    async def publish(self, spec, chunks, vectors, checksum, *, signature="", job_id=None, claim_token=None):
        if not spec.reviewed:
            raise ValueError("文档未经管理员审核，不能发布")
        if spec.valid_until is not None and spec.valid_until <= datetime.now(timezone.utc):
            raise ValueError("已过期文档不能发布")
        if any(len(vector) != 1024 or not all(math.isfinite(v) for v in vector)
               or not any(v != 0 for v in vector) for vector in vectors):
            raise ValueError("知识库向量必须为 1024 维")
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("切片与向量不能为空且数量必须一致")
        source_key, source_scope, document_id = self._identity(spec)
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
                if (existing.index_signature or "") == signature:
                    complete_job(len(chunks))
                    return {"document_id": document_id, "chunks": 0, "unchanged": True}
                # 内容没变、切块或嵌入配置变了：原地重建切片，不新增文档版本。
                # 否则新切片会被丢弃，检索继续用旧参数切出来的内容。
                await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    db.add(self._chunk_row(document_id, index, chunk, vector))
                existing.index_signature = signature
                existing.updated_at = datetime.now(timezone.utc)
                await self._bump_index(db, spec.kb_id)
                complete_job(len(chunks))
                return {"document_id": document_id, "chunks": len(chunks),
                        "unchanged": False, "rebuilt": True}
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
                version=spec.version, checksum=checksum, index_signature=signature, status="active",
                valid_from=spec.valid_from, valid_until=spec.valid_until,
            ))
            await db.flush()
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                db.add(self._chunk_row(document_id, index, chunk, vector))
            await self._bump_index(db, spec.kb_id)
            complete_job(len(chunks))
        return {"document_id": document_id, "chunks": len(chunks), "unchanged": False}

    async def disable(self, document_id):
        # 保留来源与历史版本，但检索立即排除该版本。
        async with session_scope() as db:
            document = await db.get(Document, document_id)
            if document is None or document.status == "disabled":
                return False
            document.status = "disabled"
            await self._bump_index(db, document.kb_id)
            return True
