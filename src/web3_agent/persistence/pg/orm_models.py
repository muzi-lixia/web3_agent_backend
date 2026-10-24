"""ORM 表定义。只做存储，不含业务方法。

会话、记忆、审计、代币注册表、配额、nonce、RAG 与幂等表。
领域模型在 common/types.py，与这里的 ORM 严格分离，由 repositories 做转换。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    LargeBinary,
    ForeignKey,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
    Index
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from web3_agent.persistence.pg.base import Base

EMBEDDING_DIM = 1024  # bge-m3


class ConversationMessage(Base):
    __tablename__ = "conversation_message"
    __table_args__ = (
        Index("ix_conv_msg_user_id_desc", "siwe_user_id", text("id DESC")),
        # 摘要最多一条：删旧插新的并发靠这条索引定胜负，不靠应用层先查后写。
        Index(
            "uq_conv_summary_user",
            "siwe_user_id",
            unique=True,
            postgresql_where=text("role = 'summary'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    siwe_user_id: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool | summary
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # 一轮对话的标识。同一轮先写 user、后写 assistant，中间隔着模型调用，
    # 两台设备并发时 id 顺序会变成 u1,u2,a1,a2 —— 靠 id 拼历史等于把 a1 读成回答 u2。
    # 加载时按 turn_id 分组、组内 user 在前，重建出 u1,a1,u2,a2。
    # 摘要行（role='summary'）没有 turn_id：它不属于任何一轮，也不进对话窗口。
    turn_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationGeneration(Base):
    """会话生命周期版本。清空或彻底删除数据时 +1，旧任务据此放弃写回。

    后台任务（摘要、记忆抽取）和跨十几秒的对话轮次都在删除边界之外运行：
    它们读到的水位线、历史、档案可能已经被清掉，写回时若不做版本比对，
    用户刚删掉的内容就会在几秒后重新出现。

    这一行只存用户 id 和一个计数器，不含任何对话内容，因此 purge 也保留它：
    删掉再重建会让版本回到 0，正好与清空前启动的任务撞上。
    """

    __tablename__ = "conversation_generation"

    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    generation: Mapped[int] = mapped_column(BigInteger, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UserProfile(Base):
    __tablename__ = "user_profile"

    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    answer_style: Mapped[str | None] = mapped_column(String(16), nullable=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    preferred_chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frequent_tokens: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    risk_preference: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UserMemory(Base):
    """长期记忆第二段：自由文本背景（V2）。

    与 user_profile 的分工：档案只放枚举/链号/代币符号，装不下的背景放这里。
    同一 subject 只允许一条 active —— 用部分唯一索引保证，不靠相似度去重。
    旧行改成 superseded 保留，可回溯用户什么时候改的主意。

    不存向量：active 集被上一条唯一索引压到「主题数」量级，注入时整批取出来
    按时间倒序就够，向量召回要为这个规模付一次 query embedding。
    """

    __tablename__ = "user_memory"
    __table_args__ = (
        Index(
            "uq_user_memory_active",
            "siwe_user_id",
            "subject",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    siwe_user_id: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    memory_type: Mapped[str] = mapped_column(String(16))  # preference | constraint | profile
    subject: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="active")
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserMemoryState(Base):
    """抽取水位线。没有它就无法判断「最近这批消息抽过没有」，会重复抽同一段。

    只记位置上不记轮数：轮次要靠 message 计数推，消息一删计数就错位。

    claim_token / claim_expires_at 是抽取任务的租约：同一用户同一时刻只允许
    一个任务在跑。多个后台任务同时抽同一段区间，除了白花一次模型调用，还会
    互相覆盖 —— 慢的那个后写回，把水位线拉回去、并用自己的旧结论把新结论顶掉。
    """

    __tablename__ = "user_memory_state"

    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_extracted_message_id: Mapped[int] = mapped_column(BigInteger, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    claim_token: Mapped[str | None] = mapped_column(String(32), nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    siwe_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    siwe_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    langsmith_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TokenRegistry(Base):
    __tablename__ = "token_registry"

    chain_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_address: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(64))
    raw_chain_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decimals: Mapped[int] = mapped_column(Integer)
    is_whitelisted: Mapped[bool] = mapped_column(Boolean, default=True)


class QuotaCounter(Base):
    """配额与限流计数，1 分钟分桶，单语句 upsert 累加。

    subject 形如 `{userId}|{metric}`，metric ∈ steps / tokens / rpcCalls / txCount / qps。
    不单开 metric 列，是为了让主键查询一列搞定（PRD 附录B 的 subject/window/count 语义）。
    """

    __tablename__ = "quota_counter"

    subject: Mapped[str] = mapped_column(String(128), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, index=True
    )
    count: Mapped[int] = mapped_column(BigInteger, default=0)


class SiweNonce(Base):
    """SIWE 一次性 nonce。

    consumed_at 承担"单次消费"语义：核销靠 UPDATE ... WHERE consumed_at IS NULL，
    并发下只有一个请求能拿到那条记录，不需要额外加锁。
    address 非空表示 nonce 已绑定地址，验签时 recovered 必须与之一致。
    """

    __tablename__ = "siwe_nonce"

    nonce: Mapped[str] = mapped_column(String(64), primary_key=True)
    address: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RagKnowledgeBase(Base):
    __tablename__ = "rag_knowledge_base"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    embedding_provider: Mapped[str] = mapped_column(String(32))
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    embedding_version: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RagIndexVersion(Base):
    """关键词索引版本，每个知识库一行。发布或停用文档时 +1。

    索引是进程内对象，"该不该重建"此前靠每次查询把有效集读一遍、拿正文算指纹：
    命中缓存也逃不掉一次全量读取。改成事件推进的版本号之后，查询只读这一行；
    正文只在版本变化（或下文的有效期边界到达）时才读一次。
    """

    __tablename__ = "rag_index_version"

    kb_id: Mapped[str] = mapped_column(
        ForeignKey("rag_knowledge_base.id", ondelete="RESTRICT"), primary_key=True
    )
    revision: Mapped[int] = mapped_column(BigInteger, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagDocument(Base):
    __tablename__ = "rag_document"
    kb_id: Mapped[str] = mapped_column(ForeignKey("rag_knowledge_base.id", ondelete="RESTRICT"), index=True)
    embedding_provider: Mapped[str] = mapped_column(String(32))
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    embedding_version: Mapped[str] = mapped_column(String(64))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    protocol_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    source_url: Mapped[str] = mapped_column(Text)
    doc_category: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(64))
    checksum: Mapped[str] = mapped_column(String(64))
    # 索引构建指纹：内容 + 切块器版本 + 切块参数 + 嵌入配置。checksum 只描述内容，
    # 改切块参数时它不变，靠这一列判断切片是否过期。
    index_signature: Mapped[str] = mapped_column(String(64), default="", server_default="")
    status: Mapped[str] = mapped_column(String(16), default="active")
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagChunk(Base):
    __tablename__ = "rag_chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_rag_chunk_order"),
        # 相似度检索走的是 ORDER BY cosine_distance LIMIT k，没有这条索引就是全表扫。
        # 见 scripts/rag_vector_index.py：既有库要 CONCURRENTLY 单独建，create_all 不会补。
        Index(
            "ix_rag_chunk_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("rag_document.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    token_count: Mapped[int] = mapped_column(Integer)
    token_count_method: Mapped[str] = mapped_column(String(32))
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class RagIngestionJob(Base):
    """Durable upload queue; bytes and metadata survive process restarts."""
    __tablename__ = "rag_ingestion_job"
    kb_id: Mapped[str] = mapped_column(ForeignKey("rag_knowledge_base.id", ondelete="RESTRICT"), index=True)
    __table_args__ = (UniqueConstraint("created_by", "idempotency_key", name="uq_rag_upload_request"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    filename: Mapped[str] = mapped_column(String(255))
    payload: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, nullable=True)
    document_metadata: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(16), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    claim_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    embedded_chunks: Mapped[int] = mapped_column(Integer, default=0)
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatIdempotency(Base):
    """客户端重试的去重记录：同一用户同一个 key 只跑一轮，重试直接取回上次的响应。

    response 存的是已经组装好的响应体（不含引用正文之外的额外内容），
    保留一天后由下一次带 key 的请求顺手删掉。
    """

    __tablename__ = "chat_idempotency"

    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="running")
    response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

