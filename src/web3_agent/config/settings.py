"""环境配置。全部来自环境变量 / .env，进程中只读。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from web3_agent.config.constants import DEFAULT_MAX_AMOUNT_USD


# 数据职责：运行配置集合；从环境变量和 .env 加载，字段默认值及校验约束保持在定义处。
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # app_env：应用运行环境标签，例如 dev。
    app_env: str = "dev"
    # log_level：日志级别，例如 INFO、DEBUG。
    log_level: str = "INFO"

    # --- 部署：反代层数 ---
    # 0 表示直接对外（取 socket 对端地址）。部署在 N 层反向代理之后必须改成实际层数，
    # 否则匿名请求的限流键全部落到代理 IP 上，整站共享一个桶。
    # trusted_proxy_hops：可信反向代理层数；0 时直接使用 socket 对端 IP，用于确定限流主体。
    trusted_proxy_hops: int = 0

    # --- 存储：PostgreSQL 单一存储，无缓存中间件 ---
    # database_url：SQLAlchemy 异步 PostgreSQL 连接串。
    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/web3_agent"
    # db_pool_size：数据库连接池常驻连接数。
    db_pool_size: int = 10
    # auto_create_tables：启动时是否调用 create_all 创建缺失表；不会升级已有表结构。
    auto_create_tables: bool = True
    # checkpoint_pool_size：多步计划检查点的连接池上限；写路径不高频，不需要和业务池同规格。
    checkpoint_pool_size: int = 4

    # --- 链 ---
    # allowed_chain_ids：系统允许访问的 EVM 网络编号列表。
    allowed_chain_ids: list[int] = Field(default_factory=lambda: [1])
    # rpc_urls：网络编号到 JSON-RPC URL 的映射；未配置 URL 的链不能查询余额。
    rpc_urls: dict[int, str] = Field(default_factory=dict)
    # rpc_timeout_seconds：单次 RPC 或交易浏览器 HTTP 调用的超时秒数。
    rpc_timeout_seconds: float = 10.0
    # read_cache_ttl_seconds：链上只读结果在进程内缓存的有效秒数。
    read_cache_ttl_seconds: float = 10.0

    # --- 区块浏览器（交易记录数据源）---
    # explorer_base_url：交易记录数据源的区块浏览器 API 地址。
    explorer_base_url: str = "https://api.etherscan.io/v2/api"
    # explorer_api_key：交易浏览器 API 凭据；为空时交易记录工具返回未配置。
    explorer_api_key: str = ""

    # --- 质押合约白名单：{"别名或地址": {"chain_id": 1, "address": "0x..", "adapter": "generic"}} ---
    # stake_contracts：质押合约白名单，键为别名或地址，值包含 chain_id、address、adapter。
    stake_contracts: dict[str, dict] = Field(default_factory=dict)

    # --- Agent ---
    # read_max_react_steps：只读 ReAct 图允许的最大模型调用步数。
    read_max_react_steps: int = 5
    # 单轮的三条边界各管一层：模型步数（上面那条）、下游请求数、整轮时间。
    # 一条模型消息可以带多个工具调用，一个工具又能并发打出一串下游请求，
    # 只数步数管不住后两层。
    # turn_max_downstream_calls：单轮 RPC、浏览器、向量化及重排共用的下游请求次数上限。
    turn_max_downstream_calls: int = 24  # RPC、区块浏览器、向量化共用，各自分开计数
    # turn_timeout_seconds：从进入 Agent 到生成结果的整轮超时秒数。
    turn_timeout_seconds: float = 90
    # conversation_max_messages：注入模型的近期非摘要消息条数上限。
    conversation_max_messages: int = 20
    # 条数管不住体积：一条 50 笔交易的直出文本就有 7000 字符，
    # 两三轮下来会把 num_ctx 挤满，而 Ollama 超限是静默截断。
    # conversation_max_chars：近期历史的字符预算；最新一条消息按加载器规则保留。
    conversation_max_chars: int = 6000

    # --- 摘要（中期记忆）：窗口挤出多少条才开始压、一次最多压多少 ---
    # summary_trigger_min：移出近期窗口且尚未压缩的消息达到此数量时触发摘要。
    summary_trigger_min: int = 10
    # summary_batch_cap：一次摘要任务最多读取的历史消息条数。
    summary_batch_cap: int = 30
    # summary_field_max_chars：摘要每个条目允许的最大字符数。
    summary_field_max_chars: int = 200
    # summary_field_max_items：摘要每个分类最多保留的条目数。
    summary_field_max_items: int = 50
    # summary_render_chars：注入提示词的摘要文本字符预算。
    summary_render_chars: int = 600

    # --- 长期记忆：结构化档案 ---
    # profile_render_chars：注入提示词的结构化档案字符预算。
    profile_render_chars: int = 400
    # profile_token_max_items：档案中最多展示的常用代币条目数。
    profile_token_max_items: int = 5

    # --- 长期记忆：自由文本（V2）---
    # memory_trigger_messages：水位线之后新增消息达到此数量时触发长期记忆抽取。
    memory_trigger_messages: int = 10  # ≈5 轮
    # memory_batch_cap：一次长期记忆抽取最多处理的消息条数。
    memory_batch_cap: int = 20
    # memory_fail_max：达到该失败次数后停止再次触发记忆抽取。
    memory_fail_max: int = 3
    # 抽取租约。一次抽取只有一次模型调用，正常几十秒内结束；这个值只决定
    # 进程被杀死之后，该用户的长期记忆要等多久才会重新开始抽取。
    # memory_claim_ttl_seconds：记忆抽取租约有效秒数，过期后允许其他任务接管。
    memory_claim_ttl_seconds: int = 300
    # 软记忆每轮最多注入几条。active 集本身只有「主题数」那么多，
    # 这里按最近写入取前 N 条，不做相似度筛选。
    # memory_recall_limit：当前按最近写入顺序选取的软记忆条数上限。
    memory_recall_limit: int = 3
    # memory_constraint_max：注入的 constraint 硬约束记忆条数上限。
    memory_constraint_max: int = 5
    # memory_render_chars：注入提示词的长期记忆字符预算。
    memory_render_chars: int = 400

    # --- 通用文档 RAG（与用户记忆独立）---
    # rag_embedding_version：当前知识库嵌入配置的版本标识。
    rag_embedding_version: str = Field(default="v1", min_length=1, max_length=64)
    # rag_enabled：是否启用知识库检索及对应 Agent 工具。
    rag_enabled: bool = False
    # rag_upload_enabled：是否启用知识库管理上传接口与入库 worker。
    rag_upload_enabled: bool = False
    # rag_admin_addresses：有权管理共享知识库的钱包地址列表。
    rag_admin_addresses: list[str] = Field(default_factory=list)
    # rag_upload_max_bytes：单个上传文件允许的最大字节数，不含额外 multipart 开销。
    rag_upload_max_bytes: int = Field(default=5_000_000, ge=1024, le=20_000_000)
    # rag_max_text_chars：解析后文档正文的最大字符数，防止超大文本占用资源。
    rag_max_text_chars: int = Field(default=1_000_000, ge=1000, le=5_000_000)
    # rag_max_chunks：单篇文档最多生成的切片数。
    rag_max_chunks: int = Field(default=2000, ge=1, le=10000)
    # rag_worker_poll_seconds：没有任务可处理时 worker 的轮询间隔秒数。
    rag_worker_poll_seconds: float = Field(default=2, ge=.1, le=60)
    # rag_job_timeout_seconds：单个入库任务从解析到发布的最大处理秒数。
    rag_job_timeout_seconds: float = Field(default=600, ge=10, le=3600)
    # rag_job_max_attempts：单个入库任务允许被领取执行的最大次数。
    rag_job_max_attempts: int = Field(default=3, ge=1, le=10)
    # rag_top_k：最终最多返回给回答模型的检索证据条数。
    rag_top_k: int = Field(default=3, ge=1, le=5)
    # rag_max_distance：向量召回接受的最大余弦距离，越小越严格。
    rag_max_distance: float = Field(default=0.42, ge=0, le=2)
    # rag_chunk_tokens：每个文档切片的目标近似 token 数。
    rag_chunk_tokens: int = Field(default=400, ge=400, le=600)
    # rag_chunk_overlap：同章节相邻切片重叠比例，取值小于 0.5。
    rag_chunk_overlap: float = Field(default=0.1, ge=0, lt=0.5)
    # rag_max_context_chars：最终检索证据正文的累计字符预算。
    rag_max_context_chars: int = Field(default=2400, ge=600, le=8000)
    # rag_timeout_seconds：一次知识检索的整体超时秒数。
    rag_timeout_seconds: float = Field(default=20, gt=0, le=120)

    # Hybrid retrieval: all optional stages stay inside the RAG tool.
    # rag_hybrid_enabled：是否启用向量和 BM25 关键词混合召回。
    rag_hybrid_enabled: bool = False
    # rag_candidate_k：每一路召回最多提供给融合阶段的候选条数。
    rag_candidate_k: int = Field(default=20, ge=5, le=100)
    # rag_fusion_k：RRF 融合后最多保留的候选条数。
    rag_fusion_k: int = Field(default=15, ge=3, le=100)
    # rag_keyword_max_chunks：单个作用域可建立 BM25 索引的最大切片数。
    rag_keyword_max_chunks: int = Field(default=10000, ge=100, le=100000)
    # rag_rerank_enabled：是否启用外部 cross-encoder 重排。
    rag_rerank_enabled: bool = False
    # rag_rerank_url：重排服务 HTTP 地址，使用兼容 /rerank 的请求契约。
    rag_rerank_url: str = ""
    # rag_rerank_model_version：重排模型版本，纳入检索缓存键。
    rag_rerank_model_version: str = "v1"
    # rag_rerank_timeout_seconds：重排 HTTP 请求的超时秒数，还受检索总预算限制。
    rag_rerank_timeout_seconds: float = Field(default=5, gt=0, le=30)
    # rag_rerank_min_score：重排后最低分数门槛；为空表示不过滤低分候选。
    rag_rerank_min_score: float | None = None
    # rag_rewrite_enabled：是否用模型消解问题指代并生成同义检索表达。
    rag_rewrite_enabled: bool = False
    # rag_rewrite_timeout_seconds：查询改写阶段的超时秒数。
    rag_rewrite_timeout_seconds: float = Field(default=3, gt=0, le=15)
    # rag_degraded_cache_ttl_seconds：存在降级阶段但仍获得证据时的缓存最长有效秒数。
    rag_degraded_cache_ttl_seconds: float = Field(default=10, ge=0, le=60)
    # rag_cache_ttl_seconds：检索结果与关键词索引缓存的有效秒数，0 表示不缓存。
    rag_cache_ttl_seconds: float = Field(default=600, ge=0, le=3600)
    # rag_cache_max_entries：进程内检索结果缓存最多保留的键数量。
    rag_cache_max_entries: int = Field(default=128, ge=1, le=10000)

    # --- 写链路风控（PRD 12 章：单笔等值金额上限）---
    # write_max_amount_usd：写操作预留的单笔美元等值限额，当前只读链路不据此执行交易。
    write_max_amount_usd: float = DEFAULT_MAX_AMOUNT_USD
    # workflow_ttl_seconds：一笔写操作从回显到确认、广播、回执核对的总有效秒数。
    # 到期前会强制核对一次回执，因此不能短于「用户看一眼再签个名」的实际耗时。
    workflow_ttl_seconds: int = Field(default=180, ge=60, le=86400)
    # workflow_sweep_seconds：兜底扫描间隔秒数；设为 0 表示不启动后台推进。
    workflow_sweep_seconds: int = Field(default=60, ge=0, le=3600)
    # workflow_sweep_batch：单次扫描最多处理的任务数。
    workflow_sweep_batch: int = Field(default=50, ge=1, le=500)

    # --- 本地 Ollama ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    # ollama_model：意图识别、回答及摘要使用的聊天模型名称。
    ollama_model: str = "qwen2.5:7b"
    # ollama_model_digest：期望的聊天模型指纹；为空时不检查指纹一致性。
    ollama_model_digest: str = ""
    # ollama_num_ctx：聊天模型上下文窗口容量，单位为 token。
    ollama_num_ctx: int = 8192
    # ollama_keep_alive：Ollama 在请求结束后保留模型驻留的时长字符串。
    ollama_keep_alive: str = "30m"
    # ollama_temperature：模型采样温度；0 用于提高输出确定性。
    ollama_temperature: float = 0.0
    # ollama_timeout_seconds：聊天模型 HTTP 客户端超时秒数。
    ollama_timeout_seconds: float = 120.0
    # model_check_on_startup：启动时是否检查所配聊天模型已存在及指纹匹配。
    model_check_on_startup: bool = True

    # --- 向量化（bge-m3；维度写死在建表里）---
    # ollama_embed_model：文档向量化和检索问题向量化所用模型名称。
    ollama_embed_model: str = "bge-m3"
    # ollama_embed_dim：向量维度，必须与数据库 vector(1024) 兼容。
    ollama_embed_dim: int = 1024

    # --- LangSmith（SaaS）---
    # 只认这一个开关：打开就全量上报，关闭时连 run 都不建。
    # langsmith_tracing：是否启用 LangSmith 追踪上报。
    langsmith_tracing: bool = False
    # langsmith_api_key：LangSmith 服务访问凭据。
    langsmith_api_key: str = ""
    # langsmith_project：LangSmith 中接收执行轨迹的项目名。
    langsmith_project: str = "web3-agent-dev"
    # langsmith_endpoint：LangSmith 服务 API 地址。
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- 鉴权（V1a 只读匿名可访问；写链路需要）---
    # 32 字节下限是 HS256 的安全基线，配短了启动就失败，不留到线上才发现。
    # jwt_secret：JWT 的 HMAC 签名密钥，配置校验要求至少 32 个字符。
    jwt_secret: str = Field(min_length=32)
    # jwt_algorithm：允许的 JWT 签名算法，限制为 HS256、HS384 或 HS512。
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    # jwt_ttl_seconds：登录 JWT 的有效时长，单位为秒。
    jwt_ttl_seconds: int = 7200
    # SIWE 消息里的 domain / uri 必须与本站一致，防钓鱼页面拿本站 nonce 签出跨站消息
    # siwe_domain：SIWE 消息中必须匹配的本站域名。
    siwe_domain: str = "localhost"
    # siwe_uri：SIWE 消息中必须匹配的本站 URI。
    siwe_uri: str = "http://localhost:5173"
    # siwe_nonce_ttl_seconds：一次性登录挑战的有效秒数。
    siwe_nonce_ttl_seconds: int = 300

    # --- 限流与配额（PRD 12 章：1 分钟分桶，计数落 PG）---
    # quota_anon_rpm：匿名主体每分钟允许的聊天类请求数。
    quota_anon_rpm: int = 10
    # quota_user_rpm：登录用户每分钟允许的聊天类请求数。
    quota_user_rpm: int = 30
    # /auth/nonce 单独放宽：登录前一秒一次会让前端重连时卡住
    # quota_auth_rpm：同一 IP 每分钟允许的鉴权接口请求数。
    quota_auth_rpm: int = 20
    # quota_rpc_per_minute：用户每分钟 RPC、浏览器、向量化及重排合计的下游配额阈值。
    quota_rpc_per_minute: int = 200
    # quota_steps_per_minute：用户每分钟模型调用的配额阈值。
    quota_steps_per_minute: int = 200


@lru_cache
def get_settings() -> Settings:
    """从环境和 .env 加载配置，并在进程内缓存同一实例。"""
    return Settings()
