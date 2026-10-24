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

# EMBEDDING_DIM：数据库向量列固定维度，必须与嵌入模型输出一致。
EMBEDDING_DIM = 1024  # bge-m3


# 数据职责：用户消息、助手回答及唯一摘要的存储表，按 SIWE 用户归属。
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

    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str] = mapped_column(String(64))
    # role：消息角色：user 用户、assistant 助手、tool 工具、summary 摘要。
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool | summary
    # content：普通消息为对话正文；role 为 summary 时保存摘要字段的 JSON。
    content: Mapped[str] = mapped_column(Text)
    # tool_calls：本轮工具调用轨迹，包含名称、参数及执行结果；未调用时可为空。
    tool_calls: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # 一轮对话的标识。同一轮先写 user、后写 assistant，中间隔着模型调用，
    # 两台设备并发时 id 顺序会变成 u1,u2,a1,a2 —— 靠 id 拼历史等于把 a1 读成回答 u2。
    # 加载时按 turn_id 分组、组内 user 在前，重建出 u1,a1,u2,a2。
    # 摘要行（role='summary'）没有 turn_id：它不属于任何一轮，也不进对话窗口。
    # turn_id：同一轮用户和助手消息共用的标识，用于重排并发交错历史；摘要行为空。
    turn_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
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

    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # generation：用户会话生命周期版本；清空时递增，旧任务写回前必须比对。
    generation: Mapped[int] = mapped_column(BigInteger, default=0)
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# 数据职责：用户显式结构化偏好的存储表，每个用户最多一条。
class UserProfile(Base):
    __tablename__ = "user_profile"

    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # answer_style：用户明确指定的回答风格：concise 简洁、detailed 详细。
    answer_style: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # language：用户明确指定的回答语言，例如 zh 中文、en 英文。
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # preferred_chain_id：用户明确指定的默认网络；不是最近登录或查询的网络。
    preferred_chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # frequent_tokens：常用代币记录，元素包含 symbol 和 chainId，区分不同网络上的同名代币。
    frequent_tokens: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # risk_preference：用户明确表达的风险偏好：conservative、balanced 或 aggressive。
    risk_preference: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
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

    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str] = mapped_column(String(64))
    # text：已通过清洗的用户长期背景正文，以事实陈述保存。
    text: Mapped[str] = mapped_column(Text)
    # memory_type：长期记忆类型：preference 偏好、constraint 约束、profile 背景。
    memory_type: Mapped[str] = mapped_column(String(16))  # preference | constraint | profile
    # subject：长期记忆的主题标识，同一用户同主题只允许一条 active 记录。
    subject: Mapped[str] = mapped_column(String(64))
    # confidence：这条记忆抽取时的模型置信度。
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # status：记忆状态：active 当前有效、superseded 已被同主题新记忆替代。
    status: Mapped[str] = mapped_column(String(16), default="active")
    # superseded_at：记忆被同主题新记录替代的时间；仍有效时为空。
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserMemoryState(Base):
    """抽取水位线。没有它就无法判断「最近这批消息抽过没有」，会重复抽同一段。

    只记位置上不记轮数：轮次要靠 message 计数推，消息一删计数就错位。

    claim_token / claim_expires_at 是抽取任务的租约：同一用户同一时刻只允许
    一个任务在跑。多个后台任务同时抽同一段区间，除了白花一次模型调用，还会
    互相覆盖 —— 慢的那个后写回，把水位线拉回去、并用自己的旧结论把新结论顶掉。
    """

    __tablename__ = "user_memory_state"

    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # last_extracted_message_id：已完成长期记忆抽取的最大消息 ID，作为下一批的起点水位线。
    last_extracted_message_id: Mapped[int] = mapped_column(BigInteger, default=0)
    # fail_count：记忆抽取累计失败计数，用于限制持续失败后的再次触发。
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    # claim_token：本次后台任务的唯一租约标识；写回时校验，防止旧执行者覆盖新任务。
    claim_token: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # claim_expires_at：记忆抽取租约到期时间，到期后其他任务可以接管。
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# 数据职责：请求和业务事件的审计记录表，供追踪问题与删除动作溯源。
class AuditLog(Base):
    __tablename__ = "audit_log"

    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # trace_id：贯穿 HTTP 请求与审计的追踪标识。
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # siwe_address：已通过 SIWE 验证的钱包地址。
    siwe_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # event：审计事件名称，例如登录成功、意图路由或数据清除。
    event: Mapped[str] = mapped_column(String(64), index=True)
    # payload：审计事件附加 JSON，记录本事件所需字段，不存放完整请求对象。
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # langsmith_run_id：关联 LangSmith 执行记录的 ID，用于反查模型调用轨迹。
    langsmith_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 数据职责：各网络 ERC20 白名单元数据，网络与合约地址共同确定记录。
class TokenRegistry(Base):
    __tablename__ = "token_registry"

    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # token_address：ERC20 合约地址；合成原生币元数据时使用空字符串。
    token_address: Mapped[str] = mapped_column(String(64), primary_key=True)
    # symbol：代币符号，例如 ETH、USDC。
    symbol: Mapped[str] = mapped_column(String(32))
    # name：注册代币的展示名称。
    name: Mapped[str] = mapped_column(String(64))
    # raw_chain_name：数据源提供的原始网络名称，保留用于展示或排查。
    raw_chain_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # decimals：代币精度：原始整数金额需要除以 10 的该次方才是人类可读金额。
    decimals: Mapped[int] = mapped_column(Integer)
    # is_whitelisted：代币是否在系统允许查询的白名单内。
    is_whitelisted: Mapped[bool] = mapped_column(Boolean, default=True)


class QuotaCounter(Base):
    """配额与限流计数，1 分钟分桶，单语句 upsert 累加。

    subject 形如 `{userId}|{metric}`，metric ∈ steps / tokens / rpcCalls / txCount / qps。
    不单开 metric 列，是为了让主键查询一列搞定（PRD 附录B 的 subject/window/count 语义）。
    """

    __tablename__ = "quota_counter"

    # subject：配额主体与维度组合键，通常为 u:<用户ID>|<维度> 或 ip:<地址>|<维度>。
    subject: Mapped[str] = mapped_column(String(128), primary_key=True)
    # window_start：当前配额分钟桶的起点时间，与主体键共同唯一确定计数器。
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, index=True
    )
    # count：该主体在当前时间桶内累计的资源用量。
    count: Mapped[int] = mapped_column(BigInteger, default=0)


class SiweNonce(Base):
    """SIWE 一次性 nonce。

    consumed_at 承担"单次消费"语义：核销靠 UPDATE ... WHERE consumed_at IS NULL，
    并发下只有一个请求能拿到那条记录，不需要额外加锁。
    address 非空表示 nonce 已绑定地址，验签时 recovered 必须与之一致。
    """

    __tablename__ = "siwe_nonce"

    # nonce：SIWE 一次性随机挑战值，验签成功后核销以防重放。
    nonce: Mapped[str] = mapped_column(String(64), primary_key=True)
    # address：钱包地址；登录接口用它绑定签名身份，查询接口用它指定账户。
    address: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # issued_at：挑战或签名消息的签发时间。
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # expires_at：凭据或租约失效时间；超过后不能继续使用。
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    # consumed_at：nonce 核销时间；为空表示尚未被成功消费。
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 数据职责：共享知识库容器与嵌入配置，文档必须与该配置兼容。
class RagKnowledgeBase(Base):
    __tablename__ = "rag_knowledge_base"
    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # name：知识库展示名称，全库唯一。
    name: Mapped[str] = mapped_column(String(100), unique=True)
    # description：知识库的说明文本，用于管理界面展示。
    description: Mapped[str] = mapped_column(Text, default="")
    # status：知识库状态：active 可使用、disabled 已停用。
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    # embedding_provider：生成向量的服务类型，本版本为 ollama。
    embedding_provider: Mapped[str] = mapped_column(String(32))
    # embedding_model：生成和查询向量所用模型；同一知识库必须保持兼容。
    embedding_model: Mapped[str] = mapped_column(String(128))
    # embedding_dimension：向量维度，当前数据库列固定为 1024。
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    # embedding_version：嵌入配置的业务版本，区分同名模型的不同索引配置。
    embedding_version: Mapped[str] = mapped_column(String(64))
    # created_by：创建知识库或上传任务的用户标识；系统初始化可使用 system。
    created_by: Mapped[str] = mapped_column(String(128))
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
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

    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("rag_knowledge_base.id", ondelete="RESTRICT"), primary_key=True
    )
    # revision：知识库索引版本号；发布或停用文档时递增以使缓存失效。
    revision: Mapped[int] = mapped_column(BigInteger, default=0)
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 数据职责：文档业务版本、时效、范围及索引配置的持久化元数据。
class RagDocument(Base):
    __tablename__ = "rag_document"
    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: Mapped[str] = mapped_column(ForeignKey("rag_knowledge_base.id", ondelete="RESTRICT"), index=True)
    # embedding_provider：生成向量的服务类型，本版本为 ollama。
    embedding_provider: Mapped[str] = mapped_column(String(32))
    # embedding_model：生成和查询向量所用模型；同一知识库必须保持兼容。
    embedding_model: Mapped[str] = mapped_column(String(128))
    # embedding_dimension：向量维度，当前数据库列固定为 1024。
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    # embedding_version：嵌入配置的业务版本，区分同名模型的不同索引配置。
    embedding_version: Mapped[str] = mapped_column(String(64))
    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # protocol_id：协议标识；为空表示文档或查询不限定协议。
    protocol_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    # title：文档展示标题，也参与向量化文本和本地文档身份计算。
    title: Mapped[str] = mapped_column(String(200))
    # source_url：文档来源链接，仅用于溯源展示，不会自动抓取该链接。
    source_url: Mapped[str] = mapped_column(Text)
    # doc_category：文档分类标签，用于精确过滤，例如概念、教程或 FAQ。
    doc_category: Mapped[str] = mapped_column(String(64), index=True)
    # version：管理员提供的文档业务版本，同版本不得用不同内容覆盖。
    version: Mapped[str] = mapped_column(String(64))
    # checksum：文档内容与相关元数据的摘要，用于检测同一业务版本的内容变化。
    checksum: Mapped[str] = mapped_column(String(64))
    # 索引构建指纹：内容 + 切块器版本 + 切块参数 + 嵌入配置。checksum 只描述内容，
    # 改切块参数时它不变，靠这一列判断切片是否过期。
    # index_signature：内容摘要、切块版本和嵌入配置的联合指纹，用于判断是否需要重建切片。
    index_signature: Mapped[str] = mapped_column(String(64), default="", server_default="")
    # status：文档状态：active 生效版本、superseded 已被新版本替代、disabled 手动停用。
    status: Mapped[str] = mapped_column(String(16), default="active")
    # valid_from：文档生效时间；检索只接受已生效的版本。
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # valid_until：文档失效时间；为空表示未设置到期时间。
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 数据职责：文档切片正文、向量与来源偏移；按文档和顺序号唯一。
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
    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # document_id：所属或本次发布生成的文档 ID。
    document_id: Mapped[str] = mapped_column(ForeignKey("rag_document.id", ondelete="CASCADE"), index=True)
    # ordinal：切片在文档中的顺序号，当前发布流程从 0 开始编号。
    ordinal: Mapped[int] = mapped_column(Integer)
    # section：切片所属章节或 FAQ 问题标题。
    section: Mapped[str] = mapped_column(Text)
    # content：持久化切片正文，作为检索证据返回。
    content: Mapped[str] = mapped_column(Text)
    # embedding：用于语义相似度检索的 1024 维浮点向量。
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # token_count：切片的近似 token 数，用于估算切块大小，并非模型精确计费值。
    token_count: Mapped[int] = mapped_column(Integer)
    # token_count_method：近似 token 计数算法标识，便于解释计数口径。
    token_count_method: Mapped[str] = mapped_column(String(32))
    # char_start：切片在归一化文档正文中的起始字符偏移，包含该位置。
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # char_end：切片在归一化文档正文中的结束字符偏移，不包含该位置。
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # page_number：切片起始 PDF 页码，从 1 开始；非 PDF 或无页码时为空。
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # page_end：切片结束 PDF 页码，跨页时用于展示完整来源范围。
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # extra_metadata：切片的额外来源信息，数据库列名为 metadata。
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class RagIngestionJob(Base):
    """Durable upload queue; bytes and metadata survive process restarts."""
    __tablename__ = "rag_ingestion_job"
    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: Mapped[str] = mapped_column(ForeignKey("rag_knowledge_base.id", ondelete="RESTRICT"), index=True)
    __table_args__ = (UniqueConstraint("created_by", "idempotency_key", name="uq_rag_upload_request"),)
    # id：当前记录的唯一标识；具体类型及主键约束见字段定义。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # created_by：创建知识库或上传任务的用户标识；系统初始化可使用 system。
    created_by: Mapped[str] = mapped_column(String(128), index=True)
    # idempotency_key：客户端重试标识；同一主体同一个键用于复用已有结果。
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # fingerprint：上传文件字节与元数据的联合摘要，用于检查幂等键对应内容是否一致。
    fingerprint: Mapped[str] = mapped_column(String(64))
    # filename：已去除目录部分的原始文件名，用于选择解析器和展示。
    filename: Mapped[str] = mapped_column(String(255))
    # payload：上传原始文件字节，延迟加载；终结任务超过保留期后可置空。
    payload: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, nullable=True)
    # document_metadata：上传时审核过的文档元数据，worker 用它重建 DocumentSpec。
    document_metadata: Mapped[dict] = mapped_column(JSONB)
    # status：任务状态：queued 待执行、processing 执行中、succeeded 成功、failed 失败。
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # stage：任务当前阶段，如 queued、parsing、chunking、embedding、indexing、succeeded。
    stage: Mapped[str] = mapped_column(String(16), default="queued")
    # attempts：任务已经被领取执行的次数，领取时递增。
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    # claim_token：本次后台任务的唯一租约标识；写回时校验，防止旧执行者覆盖新任务。
    claim_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # lease_until：入库任务租约的到期时间，过期后可被其他 worker 接管。
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    # total_chunks：本次待处理或已发布的切片总数。
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # embedded_chunks：本次已完成向量化的切片数量，用于展示进度。
    embedded_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # document_id：所属或本次发布生成的文档 ID。
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # error_code：稳定的业务错误码，供前端判断错误类别。
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # error_message：任务失败时展示的错误信息；不直接回传底层敏感异常。
    error_message: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatIdempotency(Base):
    """客户端重试的去重记录：同一用户同一个 key 只跑一轮，重试直接取回上次的响应。

    response 存的是已经组装好的响应体（不含引用正文之外的额外内容），
    保留一天后由下一次带 key 的请求顺手删掉。
    """

    __tablename__ = "chat_idempotency"

    # siwe_user_id：数据所属的 SIWE 用户标识；可空时允许记录未登录事件。
    siwe_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # idempotency_key：客户端重试标识；同一主体同一个键用于复用已有结果。
    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    # request_hash：聊天请求内容的 SHA-256 摘要，防止同一个幂等键被用于不同问题。
    request_hash: Mapped[str] = mapped_column(String(64))
    # status：聊天执行状态：running 表示处理中，done 表示可复用 response。
    status: Mapped[str] = mapped_column(String(16), default="running")
    # response：已完成请求的响应快照，重复请求可直接读取。
    response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # created_at：记录创建时间；数据库时间字段使用带时区时间。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

