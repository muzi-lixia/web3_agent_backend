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
    """构造有效文档的 SQL 条件，统一应用知识库、时效和显式范围过滤。

    Args:
        chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
        protocol_id: 协议精确过滤，None 表示不限制。
        now: 注入的当前时间；为空时读取当前 UTC 时间。
        doc_category: 文档分类精确过滤，None 表示不限制。
        kb_id: 目标知识库标识，默认 default。
        profile: 本仓储要求的知识库嵌入配置。
    """
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

# 数据职责：PG/pgvector 文档仓储，统一处理范围过滤、版本发布和索引更新。
class PgKnowledgeStore:
    # 建索引只用到这些列。select(Chunk) 会把 1024 维向量一起读出来 ——
    # BM25 一个数值都不用，白付整表的传输与反序列化。
    # _CHUNK_COLUMNS：关键词索引快照需要的切片列，不读取 embedding 向量。
    _CHUNK_COLUMNS = (Chunk.id, Chunk.section, Chunk.content, Chunk.token_count,
                      Chunk.token_count_method, Chunk.char_start, Chunk.char_end,
                      Chunk.page_number, Chunk.page_end, Chunk.extra_metadata)
    # _DOCUMENT_COLUMNS：关键词索引快照需要的文档来源和范围列。
    _DOCUMENT_COLUMNS = (Document.id, Document.kb_id, Document.title, Document.source_url,
                         Document.version, Document.doc_category, Document.valid_from,
                         Document.chain_id, Document.protocol_id, Document.updated_at)

    def __init__(self, profile=None):
        # profile：该仓储读取和发布向量时要求的嵌入配置。
        """初始化 PgKnowledgeStore 所需的依赖和运行状态。

        Args:
            profile: 本仓储要求的知识库嵌入配置。
        """
        self.profile = profile or EmbeddingProfile()
        # knowledge_bases：知识库容器及嵌入配置管理仓储。
        self.knowledge_bases = KnowledgeBaseRepository(self.profile)

    async def validate_kb(self,kb_id):
        """要求目标知识库启用且与当前向量配置兼容。

        Args:
            kb_id: 目标知识库标识，默认 default。
        """
        await self.knowledge_bases.require_active(kb_id)

    async def has_documents(self, chain_id=None, protocol_id=None, doc_category=None, kb_id=DEFAULT_KB_ID):
        """检查当前过滤范围内是否至少有一篇有效文档，避免无意义向量化。

        Args:
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            protocol_id: 协议精确过滤，None 表示不限制。
            doc_category: 文档分类精确过滤，None 表示不限制。
            kb_id: 目标知识库标识，默认 default。
        """
        await self.validate_kb(kb_id)
        async with session_scope() as db:
            query = select(Document.id).join(KB,KB.id==Document.kb_id).where(KB.status=="active",*active_scope(chain_id, protocol_id, doc_category=doc_category, kb_id=kb_id,profile=self.profile)).limit(1)
            return (await db.execute(query)).first() is not None

    async def search(self, vector, *, chain_id=None, protocol_id=None, doc_category=None, limit=3, max_distance=.42, kb_id=DEFAULT_KB_ID, validate=True):
        """在 SQL 中限定有效文档作用域，按余弦距离查询最近切片并返回来源证据。

        Args:
            vector: 当前查询向量或切片向量。
            chain_id: 目标或过滤用的 EVM 网络编号，None 表示未限定。
            protocol_id: 协议精确过滤，None 表示不限制。
            doc_category: 文档分类精确过滤，None 表示不限制。
            limit: 本次查询或展示返回条数上限。
            max_distance: 允许的最大余弦距离，越小越严格。
            kb_id: 目标知识库标识，默认 default。
            validate: 是否在读取前额外检查知识库状态及嵌入配置。

        Returns:
            list[Citation]: 按余弦距离排序的有效文档切片证据。
        """
        if validate:
            await self.validate_kb(kb_id)
        # 使用余弦距离排序；距离与 BM25 分数不同，混合召回不能直接相加。
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
        """把文档、切片和可选距离组装为可展示且可溯源的引用对象。

        Args:
            chunk: 单个文档切片对象。
            document: 切片所属的文档 ORM 记录。
            score: 当前切片的距离值或相关性分数，具体由调用方约定。
            index: 当前切片序号或引用编号。
        """
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

        Args:
            validate: 是否在读取前额外检查知识库状态及嵌入配置。
            **filters: 已经规范化的知识库及业务范围过滤字典。

        Returns:
            (索引递增版本, 最近文档到期时间)，二者都参与缓存失效判断。
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
        """读取一次有效切片用于建索引；调用方只在版本变化时才会走到这里。

        Args:
            max_chunks: 允许建立关键词索引的最大切片数量。
            validate: 是否在读取前额外检查知识库状态及嵌入配置。
            **filters: 已经规范化的知识库及业务范围过滤字典。
        """
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
        """发布或停用后推进版本，下一次查询就会重建索引。

        Args:
            db: 调用方提供的异步数据库会话；事务提交由外层负责。
            kb_id: 目标知识库标识，默认 default。
        """
        await db.execute(insert(RagIndexVersion).values(kb_id=kb_id, revision=1)
                         .on_conflict_do_update(index_elements=[RagIndexVersion.kb_id],
                                                set_={"revision": RagIndexVersion.revision + 1,
                                                      "updated_at": func.now()}))


    @staticmethod
    def _chunk_row(document_id, index, chunk, vector):
        """为切片生成稳定 ID 并构造包含向量和来源偏移的 ORM 行。

        Args:
            document_id: 目标文档唯一标识，关联该文档的所有切片。
            index: 当前切片序号或引用编号。
            chunk: 单个文档切片对象。
            vector: 当前查询向量或切片向量。
        """
        return Chunk(id=hashlib.sha256(f"{document_id}|{index}|{chunk.content}".encode()).hexdigest(),
                     document_id=document_id, ordinal=index, section=chunk.section,
                     content=chunk.content, embedding=vector, token_count=chunk.token_count,
                     token_count_method=chunk.token_count_method, char_start=chunk.char_start,
                     char_end=chunk.char_end, page_number=chunk.page_number,
                     page_end=chunk.page_end, extra_metadata=chunk.metadata)

    @staticmethod
    def _identity(spec):
        """文档身份：URL 文档按来源，本地文档按标题；同名同范围才算同一份。

        Args:
            spec: 经过校验的文档入库元数据。

        Returns:
            (同来源串行化锁键, 同来源 SQL 筛选条件, 含版本的稳定文档 ID)。
        """
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

        Args:
            spec: 经过校验的文档入库元数据。
            checksum: 文档内容与元数据的摘要。
            signature: 内容、切块参数和嵌入配置的联合指纹。
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
        # 返回 document_id、chunks 和 unchanged：分别表示文档标识、本次实际写入切片数、是否复用原索引。
        # 同内容因切块配置变化而重建时另给 rebuilt=true；文档和任务成功状态在同一事务提交。
        """校验文档和向量，核对任务租约后原子替换版本、切片与成功状态。

        Args:
            spec: 经过校验的文档入库元数据。
            chunks: 本批文档切片，或任务的切片统计数量。
            vectors: 与切片一一对应的向量列表。
            checksum: 文档内容与元数据的摘要。
            signature: 内容、切块参数和嵌入配置的联合指纹。
            job_id: 入库任务唯一标识。
            claim_token: 本次执行者独占的租约标识。
        """
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
            # 写入前锁定并校验知识库，防止停用或不兼容配置与发布并发穿透。
            await self.knowledge_bases.require_active(spec.kb_id,db,lock=True)
            job = None
            if job_id is not None:
                job = (await db.execute(select(RagIngestionJob).where(
                    RagIngestionJob.id == job_id).with_for_update())).scalar_one_or_none()
                if job is None or job.kb_id != spec.kb_id or job.status != "processing" or job.claim_token != claim_token or job.lease_until <= datetime.now(timezone.utc):
                    raise LostClaim()
            def complete_job(count):
                """在文档发布的当前事务内同步完成入库任务并清理租约。

                Args:
                    count: 已经处理完的切片数量。
                """
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
                # 与文档变更一同推进索引版本，提交后各进程缓存会在后续查询中自然失效。
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
            # 与文档变更一同推进索引版本，提交后各进程缓存会在后续查询中自然失效。
            await self._bump_index(db, spec.kb_id)
            complete_job(len(chunks))
        return {"document_id": document_id, "chunks": len(chunks), "unchanged": False}

    async def disable(self, document_id):
        # 保留来源与历史版本，但检索立即排除该版本。
        """停用文档并递增索引版本，后续检索不再使用该版本。

        Args:
            document_id: 目标文档唯一标识，关联该文档的所有切片。
        """
        async with session_scope() as db:
            document = await db.get(Document, document_id)
            if document is None or document.status == "disabled":
                return False
            document.status = "disabled"
            await self._bump_index(db, document.kb_id)
            return True
