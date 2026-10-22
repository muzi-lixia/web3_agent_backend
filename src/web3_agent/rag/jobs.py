"""PostgreSQL durable queue with bounded leases and claim-token fencing."""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update, or_, and_, func
from sqlalchemy.orm import undefer
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import RagIngestionJob as Job, RagDocument, RagChunk
from web3_agent.common.exceptions import AppError, error_tuple
from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository


def now():
    return datetime.now(timezone.utc)


def job_view(row):
    return dict(job_id=row.id, kb_id=row.kb_id, filename=row.filename, metadata=row.document_metadata,
                status=row.status, stage=row.stage, attempts=row.attempts,
                total_chunks=row.total_chunks, embedded_chunks=row.embedded_chunks,
                document_id=row.document_id, error=(dict(code=row.error_code, message=row.error_message)
                if row.error_code else None), created_at=row.created_at, updated_at=row.updated_at)


class LostClaim(Exception):
    pass


class JobRepository:
    def __init__(self,profile=None):
        self.knowledge_bases=KnowledgeBaseRepository(profile)

    async def enqueue(self, *, spec, filename, data, created_by, idempotency_key=None):
        metadata = spec.model_dump(mode="json", exclude={"path"})
        fingerprint = hashlib.sha256(data + spec.model_dump_json(exclude={"path"}).encode()).hexdigest()
        async with session_scope() as db:
            await self.knowledge_bases.require_active(spec.kb_id,db,lock=True)
            # Serialize retries of the same upload before checking its fingerprint.
            if idempotency_key:
                from sqlalchemy import text
                await db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                                 {"key": f"rag-upload:{created_by}:{idempotency_key}"})
                existing = (await db.execute(select(Job).where(Job.created_by == created_by,
                    Job.idempotency_key == idempotency_key))).scalar_one_or_none()
                if existing:
                    if existing.fingerprint != fingerprint:
                        raise AppError(*error_tuple("E609"))
                    return job_view(existing), True
            row = Job(id=str(uuid.uuid4()), kb_id=spec.kb_id, created_by=created_by, idempotency_key=idempotency_key,
                      fingerprint=fingerprint, filename=filename, payload=data, document_metadata=metadata,
                      status="queued", stage="queued", attempts=0, total_chunks=0, embedded_chunks=0,
                      created_at=now(), updated_at=now())
            db.add(row)
            await db.flush()
            return job_view(row), False

    async def get(self, job_id):
        async with session_scope() as db:
            row = await db.get(Job, job_id)
            if row is None:
                raise AppError(*error_tuple("E608"))
            return job_view(row)

    async def list(self, *, limit=20, offset=0, kb_id=None):
        async with session_scope() as db:
            rows = (await db.execute(select(Job).where(*([Job.kb_id==kb_id] if kb_id else [])).order_by(Job.created_at.desc(), Job.id).limit(limit+1).offset(offset))).scalars().all()
            return {"items": [job_view(row) for row in rows[:limit]], "has_more": len(rows)>limit}

    async def claim(self, *, timeout_seconds, max_attempts):
        async with session_scope() as db:
            expired = and_(Job.status == "processing", Job.lease_until < now())
            await db.execute(update(Job).where(expired, Job.attempts >= max_attempts).values(
                status="failed", error_code="WORKER_INTERRUPTED", error_message="处理多次中断，已达到重试上限",
                claim_token=None, lease_until=None, updated_at=now()))
            row = (await db.execute(select(Job).options(undefer(Job.payload)).where(
                or_(Job.status == "queued", expired), Job.attempts < max_attempts
            ).order_by(Job.created_at, Job.id).with_for_update(skip_locked=True).limit(1))).scalar_one_or_none()
            if row is None:
                return None
            row.status, row.stage = "processing", "loading"
            row.attempts += 1
            row.claim_token = str(uuid.uuid4())
            row.lease_until = now() + timedelta(seconds=timeout_seconds + 60)
            row.total_chunks = row.embedded_chunks = 0
            row.error_code = row.error_message = None
            row.updated_at = now()
            await db.flush()
            return dict(id=row.id, token=row.claim_token, filename=row.filename, data=row.payload,
                        metadata={**row.document_metadata,"kb_id":row.kb_id})

    async def progress(self, job_id, token, stage, **counts):
        async with session_scope() as db:
            result = await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing", Job.lease_until > now()).values(stage=stage, updated_at=now(), **counts))
            if not result.rowcount:
                raise LostClaim()

    async def fail(self, job_id, token, code, message):
        async with session_scope() as db:
            await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing").values(status="failed", error_code=code,
                error_message=message, claim_token=None, lease_until=None, updated_at=now()))

    async def retry(self, job_id, max_attempts):
        async with session_scope() as db:
            row = (await db.execute(select(Job).where(Job.id == job_id).with_for_update())).scalar_one_or_none()
            if row is None:
                raise AppError(*error_tuple("E608"))
            if row.status != "failed" or row.attempts >= max_attempts:
                raise AppError(*error_tuple("E609"))
            row.status = row.stage = "queued"
            row.error_code = row.error_message = None
            row.updated_at = now()
            await db.flush()
            return job_view(row)

    async def documents(self, *, limit=20, offset=0, kb_id=None):
        async with session_scope() as db:
            counts = select(RagChunk.document_id, func.count().label("chunks")).group_by(RagChunk.document_id).subquery()
            rows = (await db.execute(select(RagDocument, counts.c.chunks).outerjoin(counts,
                counts.c.document_id == RagDocument.id).where(*([RagDocument.kb_id==kb_id] if kb_id else [])).order_by(RagDocument.updated_at.desc(), RagDocument.id)
                .limit(limit+1).offset(offset))).all()
            return {"items": [dict(document_id=d.id,kb_id=d.kb_id, title=d.title, source_url=d.source_url,
                embedding_provider=d.embedding_provider,embedding_model=d.embedding_model,
                embedding_dimension=d.embedding_dimension,embedding_version=d.embedding_version,
                chain_id=d.chain_id, protocol_id=d.protocol_id, doc_category=d.doc_category,
                version=d.version, status=d.status, valid_from=d.valid_from, valid_until=d.valid_until,
                updated_at=d.updated_at, chunks=count or 0) for d,count in rows[:limit]], "has_more": len(rows)>limit}
