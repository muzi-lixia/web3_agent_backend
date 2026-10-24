"""Shared knowledge-base containers; embedding configuration is immutable in this version."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from web3_agent.common.exceptions import AppError, error_tuple
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.orm_models import RagKnowledgeBase as KB
from web3_agent.rag.models import DEFAULT_KB_ID, EmbeddingProfile


def kb_view(row):
    return {key: getattr(row,key) for key in ("id","name","description","status","embedding_provider",
        "embedding_model","embedding_dimension","embedding_version","created_at","updated_at")}


class KnowledgeBaseRepository:
    def __init__(self, profile=None):
        self.profile = profile or EmbeddingProfile()

    async def check_schema(self):
        from sqlalchemy.exc import SQLAlchemyError
        from web3_agent.persistence.pg.orm_models import RagDocument, RagChunk, RagIngestionJob, RagIndexVersion
        try:
            async with session_scope() as db:
                for model in (KB,RagDocument,RagChunk,RagIngestionJob,RagIndexVersion):
                    await db.execute(select(model).limit(0))
        except SQLAlchemyError as exc:
            raise RuntimeError("RAG 表结构尚未就绪，请先审阅并执行 docs/sql/rag-knowledge-base-upgrade.sql；create_all 不会升级已有表") from exc

    async def ensure_default(self):
        async with session_scope() as db:
            await db.execute(insert(KB).values(id=DEFAULT_KB_ID,name="默认知识库",description="默认共享文档知识库",
                status="active",created_by="system",**self.profile.model_dump()).on_conflict_do_nothing(index_elements=[KB.id]))

    async def require_active(self, kb_id, db=None, *, lock=False):
        if db is None:
            async with session_scope() as session:
                return await self.require_active(kb_id,session,lock=lock)
        query=select(KB).where(KB.id==kb_id)
        if lock:
            query=query.with_for_update()
        row=(await db.execute(query)).scalar_one_or_none()
        if row is None or row.status != "active":
            raise AppError(*error_tuple("E611"))
        if any(getattr(row,key) != value for key,value in self.profile.model_dump().items()):
            raise AppError(*error_tuple("E612"))
        return row

    async def list(self, *, include_disabled=False):
        async with session_scope() as db:
            query=select(KB).order_by(KB.created_at,KB.id)
            if not include_disabled:
                query=query.where(KB.status=="active")
            return [kb_view(row) for row in (await db.execute(query)).scalars()]

    async def create(self, name, description, created_by):
        from sqlalchemy.exc import IntegrityError
        try:
            async with session_scope() as db:
                row=KB(id=str(uuid.uuid4()),name=name,description=description,status="active",
                       created_by=created_by,**self.profile.model_dump())
                db.add(row)
                await db.flush()
                return kb_view(row)
        except IntegrityError as exc:
            raise AppError(*error_tuple("E609")) from exc

    async def update(self, kb_id, values):
        if not values or set(values)-{"name","description","status"}:
            raise AppError(*error_tuple("E101"))
        from sqlalchemy.exc import IntegrityError
        try:
            async with session_scope() as db:
                row=(await db.execute(select(KB).where(KB.id==kb_id).with_for_update())).scalar_one_or_none()
                if row is None:
                    raise AppError(*error_tuple("E611"))
                for key,value in values.items():
                    setattr(row,key,value)
                row.updated_at=datetime.now(timezone.utc)
                await db.flush()
                return kb_view(row)
        except IntegrityError as exc:
            raise AppError(*error_tuple("E609")) from exc
