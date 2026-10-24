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
    # 返回字段 id/name/description/status 描述知识库身份和可用状态。
    # embedding_provider/model/dimension/version 描述必须保持兼容的向量配置；created_at/updated_at 为记录时间。
    """把知识库 ORM 行转换为管理与选择界面所需的公开元数据。

    Args:
        row: 单条 ORM 数据记录。
    """
    return {key: getattr(row,key) for key in ("id","name","description","status","embedding_provider",
        "embedding_model","embedding_dimension","embedding_version","created_at","updated_at")}


# 数据职责：共享知识库仓储，负责状态检查、配置兼容和管理修改。
class KnowledgeBaseRepository:
    def __init__(self, profile=None):
        # profile：当前服务要求的嵌入配置，用来检查知识库兼容性。
        """初始化 KnowledgeBaseRepository 所需的依赖和运行状态。

        Args:
            profile: 本服务要求的知识库嵌入配置。
        """
        self.profile = profile or EmbeddingProfile()

    async def check_schema(self):
        """用只读查询检查所有 RAG 表结构是否可用，失败时提示手动升级。"""
        from sqlalchemy.exc import SQLAlchemyError
        from web3_agent.persistence.pg.orm_models import RagDocument, RagChunk, RagIngestionJob, RagIndexVersion
        try:
            async with session_scope() as db:
                for model in (KB,RagDocument,RagChunk,RagIngestionJob,RagIndexVersion):
                    await db.execute(select(model).limit(0))
        except SQLAlchemyError as exc:
            raise RuntimeError("RAG 表结构尚未就绪，请先审阅并执行 docs/sql/rag-knowledge-base-upgrade.sql；create_all 不会升级已有表") from exc

    async def ensure_default(self):
        """用幂等插入确保默认共享知识库存在。"""
        async with session_scope() as db:
            await db.execute(insert(KB).values(id=DEFAULT_KB_ID,name="默认知识库",description="默认共享文档知识库",
                status="active",created_by="system",**self.profile.model_dump()).on_conflict_do_nothing(index_elements=[KB.id]))

    async def require_active(self, kb_id, db=None, *, lock=False):
        """校验知识库存在、启用且嵌入配置兼容；写入时可要求行锁。

        Args:
            kb_id: 目标知识库标识，默认 default。
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            lock: 是否在知识库校验时获取数据库行锁。

        Raises:
            AppError: 知识库不存在、已停用或嵌入配置不兼容。
        """
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
        """按当前仓储职责分页或排序列出记录，并应用可选过滤。

        Args:
            include_disabled: 是否同时列出已停用知识库。
        """
        async with session_scope() as db:
            query=select(KB).order_by(KB.created_at,KB.id)
            if not include_disabled:
                query=query.where(KB.status=="active")
            return [kb_view(row) for row in (await db.execute(query)).scalars()]

    async def create(self, name, description, created_by):
        """按当前嵌入配置新建共享知识库，唯一性冲突转成业务错误。

        Args:
            name: 知识库展示名称。
            description: 知识库说明文本。
            created_by: 创建知识库或上传任务的用户标识。
        """
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
        """只允许修改知识库名称、描述和状态，在行锁下更新。

        Args:
            kb_id: 目标知识库标识，默认 default。
            values: 客户端显式提供的更新字段字典。
        """
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
