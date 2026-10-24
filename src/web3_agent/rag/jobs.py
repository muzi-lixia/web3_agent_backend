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
# PAYLOAD_RETENTION_DAYS：终结入库任务的原始文件保留天数。
PAYLOAD_RETENTION_DAYS = 3
# _PURGE_INTERVAL_SECONDS：原始上传文件清理扫描的最小间隔秒数。
_PURGE_INTERVAL_SECONDS = 3600.0
# _last_purge：本进程最近一次触发原始文件清理的单调时钟时间。
_last_purge = 0.0


def now():
    """返回入库队列时间判断使用的当前 UTC 时间。"""
    return datetime.now(timezone.utc)


def job_view(row):
    # 视图字段 job_id/kb_id/filename 分别为任务、知识库与文件标识；metadata 是审核后的文档元数据。
    # status/stage 表示总体状态和细分阶段；attempts 为已领取次数；total_chunks/embedded_chunks 表示切片总数和完成数。
    # document_id 为成功发布的文档；error 为可空的 {code, message}；created_at/updated_at 为创建和最近变更时间。
    # 该视图不返回原始文件 payload、claim_token 或租约到期时间。
    """将任务 ORM 行转换为前端状态视图，不暴露原始文件字节和租约 token。

    Args:
        row: 单条 ORM 数据记录。
    """
    return dict(job_id=row.id, kb_id=row.kb_id, filename=row.filename, metadata=row.document_metadata,
                status=row.status, stage=row.stage, attempts=row.attempts,
                total_chunks=row.total_chunks, embedded_chunks=row.embedded_chunks,
                document_id=row.document_id, error=(dict(code=row.error_code, message=row.error_message)
                if row.error_code else None), created_at=row.created_at, updated_at=row.updated_at)


# 数据职责：任务租约已丢失的内部信号，旧 worker 应停止写回。
class LostClaim(Exception):
    pass


# 数据职责：RAG 入库任务持久仓储，负责幂等入队、租约领取及执行状态更新。
class JobRepository:
    def __init__(self,profile=None):
        # knowledge_bases：知识库容器及嵌入配置管理仓储。
        """初始化 JobRepository 所需的依赖和运行状态。

        Args:
            profile: 知识库必须兼容的嵌入配置。
        """
        self.knowledge_bases=KnowledgeBaseRepository(profile)

    async def enqueue(self, *, spec, filename, data, created_by, idempotency_key=None):
        """校验知识库和上传幂等内容，把原始文件及元数据持久化入队。

        Args:
            spec: 经过校验的文档入库元数据。
            filename: 安全文件名，用于选择格式解析器。
            data: 待解析或持久化的原始上传文件字节。
            created_by: 创建知识库或上传任务的用户标识。
            idempotency_key: 客户端重试使用的同一请求标识。
        """
        metadata = spec.model_dump(mode="json", exclude={"path"})
        fingerprint = hashlib.sha256(data + spec.model_dump_json(exclude={"path"}).encode()).hexdigest()
        async with session_scope() as db:
            # 写入前锁定并校验知识库，防止停用或不兼容配置与发布并发穿透。
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
        """按任务 ID 返回状态视图，不存在时抛 E608。

        Args:
            job_id: 入库任务唯一标识。
        """
        async with session_scope() as db:
            # 仅读取任务状态元数据，延迟加载的文件字节不会随状态查询返回。
            row = await db.get(Job, job_id)
            if row is None:
                raise AppError(*error_tuple("E608"))
            return job_view(row)

    async def list(self, *, limit=20, offset=0, kb_id=None):
        """按当前仓储职责分页或排序列出记录，并应用可选过滤。

        Args:
            limit: 本次查询或展示返回条数上限。
            offset: 分页跳过的记录条数。
            kb_id: 目标知识库标识，默认 default。
        """
        async with session_scope() as db:
            rows = (await db.execute(select(Job).where(*([Job.kb_id==kb_id] if kb_id else [])).order_by(Job.created_at.desc(), Job.id).limit(limit+1).offset(offset))).scalars().all()
            return {"items": [job_view(row) for row in rows[:limit]], "has_more": len(rows)>limit}

    async def _purge_payloads(self, db):
        """抹掉过了保留期的原始文件。只看已终结的任务，排队中的一律不动。

        worker 每次轮询都会进 claim，所以清理跟着 worker 走，不需要额外定时器；
        一小时最多真扫一次。清理是附加动作，失败只记日志 —— 不能让「清旧文件」
        把「领新任务」一起拖下水，所以套一层 savepoint。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
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
        # 领取结果字段 id 为任务 ID，token 为本次租约标识，filename 为安全文件名，data 为原始字节，metadata 为审核元数据。
        # 领取后调用进度或发布方法都必须携带 token，防止已过期执行者继续覆盖任务。
        """清理终结任务原始文件并用 SKIP LOCKED 领取待执行或租约过期的入库任务。

        Args:
            timeout_seconds: 单个入库任务的处理时间预算，领取租约时另加收尾宽限，单位为秒。
            max_attempts: 最大执行或重试次数，具体以当前流程为准。

        Returns:
            dict | None: 领取任务的执行快照，包含 id、token、filename、data、metadata；没有任务时为 None。
        """
        async with session_scope() as db:
            await self._purge_payloads(db)
            # 仅重新领取已经失去有效租约的执行中任务，活跃执行者继续独占任务。
            expired = and_(Job.status == "processing", Job.lease_until < now())
            await db.execute(update(Job).where(expired, Job.attempts >= max_attempts).values(
                status="failed", error_code="WORKER_INTERRUPTED", error_message="处理多次中断，已达到重试上限",
                claim_token=None, lease_until=None, updated_at=now()))
            row = (await db.execute(select(Job).options(undefer(Job.payload)).where(
                or_(Job.status == "queued", expired), Job.attempts < max_attempts
            ).order_by(Job.created_at, Job.id).with_for_update(skip_locked=True).limit(1))).scalar_one_or_none()
            if row is None:
                return None
            # 任务领取即增加尝试次数并分配新租约，旧执行者后续写回将失效。
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
        """在仍持有有效租约时更新入库阶段和进度，失去资格则抛 LostClaim。

        Args:
            job_id: 入库任务唯一标识。
            token: 领取任务时获得的租约标识。
            stage: 执行阶段名称，用于进度或错误定位。
            **counts: 要更新的切片进度计数，如 total_chunks、embedded_chunks。

        Raises:
            LostClaim: 当前执行者的租约已失效或不再属于该任务。
        """
        async with session_scope() as db:
            result = await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing", Job.lease_until > now()).values(stage=stage, updated_at=now(), **counts))
            if not result.rowcount:
                raise LostClaim()

    async def succeed(self, job_id, token, document_id, chunks):
        """内容与索引都没变：任务算成功，但没有重算任何向量。

        Args:
            job_id: 入库任务唯一标识。
            token: 领取任务时获得的租约标识。
            document_id: 目标文档唯一标识，关联该文档的所有切片。
            chunks: 本批文档切片，或任务的切片统计数量。
        """
        async with session_scope() as db:
            result = await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing", Job.lease_until > now()).values(
                status="succeeded", stage="succeeded", document_id=document_id,
                total_chunks=chunks, embedded_chunks=chunks, claim_token=None, lease_until=None,
                error_code=None, error_message=None, updated_at=now()))
            if not result.rowcount:
                raise LostClaim()

    async def fail(self, job_id, token, code, message):
        """仅更新仍属于当前执行者的任务为失败，并清理租约。

        Args:
            job_id: 入库任务唯一标识。
            token: 领取任务时获得的租约标识。
            code: 已登记的业务错误码。
            message: 按当前函数用途传入的用户文本或已解析 SIWE 消息。
        """
        async with session_scope() as db:
            await db.execute(update(Job).where(Job.id == job_id, Job.claim_token == token,
                Job.status == "processing").values(status="failed", error_code=code,
                error_message=message, claim_token=None, lease_until=None, updated_at=now()))

    async def retry(self, job_id, max_attempts):
        """检查失败状态、尝试次数和原始文件是否保留，允许时重新入队。

        Args:
            job_id: 入库任务唯一标识。
            max_attempts: 最大执行或重试次数，具体以当前流程为准。
        """
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
        """分页列出文档及各自切片数量，用于知识库管理界面。

        Args:
            limit: 本次查询或展示返回条数上限。
            offset: 分页跳过的记录条数。
            kb_id: 目标知识库标识，默认 default。
        """
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
