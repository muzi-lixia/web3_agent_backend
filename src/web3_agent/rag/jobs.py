"""PostgreSQL durable queue with bounded leases and claim-token fencing."""
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update, or_, and_, func
from sqlalchemy.orm import undefer
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import RagIngestionJob as Job, RagDocument, RagChunk
from web3_agent.common.exceptions import AppError, error_tuple
from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository

logger = logging.getLogger(__name__)

# 原始文件只留 3 天：够排查、够重试，之后正文不再占库。任务行本身一直保留。
PAYLOAD_RETENTION_DAYS = 3
_PURGE_INTERVAL_SECONDS = 3600.0
_last_purge = 0.0


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

    async def _purge_payloads(self, db):
        """抹掉过了保留期的原始文件。只看已终结的任务，排队中的一律不动。

        worker 每次轮询都会进 claim，所以清理跟着 worker 走，不需要额外定时器；
        一小时最多真扫一次。清理是附加动作，失败只记日志 —— 不能让「清旧文件」
        把「领新任务」一起拖下水，所以套一层 savepoint。
        """
        global _last_purge
        if time.monotonic() - _last_purge < _PURGE_INTERVAL_SECONDS:
            return
        _last_purge = time.monotonic()
        try:
            async with db.begin_nested():
                await db.execute(update(Job).where(
                    Job.payload.is_not(None), Job.status.in_(("succeeded", "failed")),
                    Job.updated_at < now() - timedelta(days=PAYLOAD_RETENTION_DAYS),
                ).values(payload=None))
        except Exception:  # noqa: BLE001
            logger.exception("rag payload purge failed")

    async def claim(self, *, timeout_seconds, max_attempts):
        async with session_scope() as db:
            await self._purge_payloads(db)
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

    async def succeed(self, job_id, token, document_id, chunks):
        """内容与索引都没变：任务算成功，但没有重算任何向量。"""
        async with session_scope() as db:
            result = await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing", Job.lease_until > now()).values(
                status="succeeded", stage="succeeded", document_id=document_id,
                total_chunks=chunks, embedded_chunks=chunks, claim_token=None, lease_until=None,
                error_code=None, error_message=None, updated_at=now()))
            if not result.rowcount:
                raise LostClaim()

    async def fail(self, job_id, token, code, message):
        async with session_scope() as db:
            await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing").values(status="failed", error_code=code,
                error_message=message, claim_token=None, lease_until=None, updated_at=now()))

    async def retry(self, job_id, max_attempts):
        async with session_scope() as db:
            row = (await db.execute(select(Job).options(undefer(Job.payload))
                                    .where(Job.id == job_id).with_for_update())).scalar_one_or_none()
            if row is None:
                raise AppError(*error_tuple("E608"))
            # 原始文件过了保留期就重试不了：只能重新上传一份。
            if row.status != "failed" or row.attempts >= max_attempts or row.payload is None:
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
