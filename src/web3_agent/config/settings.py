"""环境配置。全部来自环境变量 / .env，进程中只读。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from web3_agent.config.constants import DEFAULT_MAX_AMOUNT_USD


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "dev"
    log_level: str = "INFO"

    # --- 部署：反代层数 ---
    # 0 表示直接对外（取 socket 对端地址）。部署在 N 层反向代理之后必须改成实际层数，
    # 否则匿名请求的限流键全部落到代理 IP 上，整站共享一个桶。
    trusted_proxy_hops: int = 0

    # --- 存储：PostgreSQL 单一存储，无缓存中间件 ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/web3_agent"
    db_pool_size: int = 10
    auto_create_tables: bool = True

    # --- 链 ---
    allowed_chain_ids: list[int] = Field(default_factory=lambda: [1])
    rpc_urls: dict[int, str] = Field(default_factory=dict)
    rpc_timeout_seconds: float = 10.0
    read_cache_ttl_seconds: float = 10.0

    # --- 区块浏览器（交易记录数据源）---
    explorer_base_url: str = "https://api.etherscan.io/v2/api"
    explorer_api_key: str = ""

    # --- 质押合约白名单：{"别名或地址": {"chain_id": 1, "address": "0x..", "adapter": "generic"}} ---
    stake_contracts: dict[str, dict] = Field(default_factory=dict)

    # --- Agent ---
    confidence_threshold: float = 0.75
    read_max_react_steps: int = 5
    # 单轮的三条边界各管一层：模型步数（上面那条）、下游请求数、整轮时间。
    # 一条模型消息可以带多个工具调用，一个工具又能并发打出一串下游请求，
    # 只数步数管不住后两层。
    turn_max_downstream_calls: int = 24  # RPC、区块浏览器、向量化共用，各自分开计数
    turn_timeout_seconds: float = 90
    conversation_max_messages: int = 20
    # 条数管不住体积：一条 50 笔交易的直出文本就有 7000 字符，
    # 两三轮下来会把 num_ctx 挤满，而 Ollama 超限是静默截断。
    conversation_max_chars: int = 6000

    # --- 摘要（中期记忆）：窗口挤出多少条才开始压、一次最多压多少 ---
    summary_trigger_min: int = 10
    summary_batch_cap: int = 30
    summary_field_max_chars: int = 200
    summary_field_max_items: int = 50
    summary_render_chars: int = 600

    # --- 长期记忆：结构化档案 ---
    profile_render_chars: int = 400
    profile_token_max_items: int = 5

    # --- 长期记忆：自由文本（V2）---
    memory_trigger_messages: int = 10  # ≈5 轮
    memory_batch_cap: int = 20
    memory_fail_max: int = 3
    # 抽取租约。一次抽取只有一次模型调用，正常几十秒内结束；这个值只决定
    # 进程被杀死之后，该用户的长期记忆要等多久才会重新开始抽取。
    memory_claim_ttl_seconds: int = 300
    # 软记忆每轮最多注入几条。active 集本身只有「主题数」那么多，
    # 这里按最近写入取前 N 条，不做相似度筛选。
    memory_recall_limit: int = 3
    memory_constraint_max: int = 5
    memory_render_chars: int = 400

    # --- 通用文档 RAG（与用户记忆独立）---
    rag_embedding_version: str = Field(default="v1", min_length=1, max_length=64)
    rag_enabled: bool = False
    rag_upload_enabled: bool = False
    rag_admin_addresses: list[str] = Field(default_factory=list)
    rag_upload_max_bytes: int = Field(default=5_000_000, ge=1024, le=20_000_000)
    rag_max_text_chars: int = Field(default=1_000_000, ge=1000, le=5_000_000)
    rag_max_chunks: int = Field(default=2000, ge=1, le=10000)
    rag_worker_poll_seconds: float = Field(default=2, ge=.1, le=60)
    rag_job_timeout_seconds: float = Field(default=600, ge=10, le=3600)
    rag_job_max_attempts: int = Field(default=3, ge=1, le=10)
    rag_top_k: int = Field(default=3, ge=1, le=5)
    rag_max_distance: float = Field(default=0.42, ge=0, le=2)
    rag_chunk_tokens: int = Field(default=400, ge=400, le=600)
    rag_chunk_overlap: float = Field(default=0.1, ge=0, lt=0.5)
    rag_max_context_chars: int = Field(default=2400, ge=600, le=8000)
    rag_timeout_seconds: float = Field(default=20, gt=0, le=120)

    # Hybrid retrieval: all optional stages stay inside the RAG tool.
    rag_hybrid_enabled: bool = False
    rag_candidate_k: int = Field(default=20, ge=5, le=100)
    rag_fusion_k: int = Field(default=15, ge=3, le=100)
    rag_keyword_max_chunks: int = Field(default=10000, ge=100, le=100000)
    rag_rerank_enabled: bool = False
    rag_rerank_url: str = ""
    rag_rerank_model_version: str = "v1"
    rag_rerank_timeout_seconds: float = Field(default=5, gt=0, le=30)
    rag_rerank_min_score: float | None = None
    rag_rewrite_enabled: bool = False
    rag_rewrite_timeout_seconds: float = Field(default=3, gt=0, le=15)
    rag_degraded_cache_ttl_seconds: float = Field(default=10, ge=0, le=60)
    rag_cache_ttl_seconds: float = Field(default=600, ge=0, le=3600)
    rag_cache_max_entries: int = Field(default=128, ge=1, le=10000)

    # --- 写链路风控（PRD 12 章：单笔等值金额上限）---
    write_max_amount_usd: float = DEFAULT_MAX_AMOUNT_USD

    # --- 本地 Ollama ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_model_digest: str = ""
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "30m"
    ollama_temperature: float = 0.0
    ollama_timeout_seconds: float = 120.0
    model_check_on_startup: bool = True

    # --- 向量化（bge-m3；维度写死在建表里）---
    ollama_embed_model: str = "bge-m3"
    ollama_embed_dim: int = 1024

    # --- LangSmith（SaaS）---
    # 只认这一个开关：打开就全量上报，关闭时连 run 都不建。
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "web3-agent-dev"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- 鉴权（V1a 只读匿名可访问；写链路需要）---
    # 32 字节下限是 HS256 的安全基线，配短了启动就失败，不留到线上才发现。
    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_ttl_seconds: int = 7200
    # SIWE 消息里的 domain / uri 必须与本站一致，防钓鱼页面拿本站 nonce 签出跨站消息
    siwe_domain: str = "localhost"
    siwe_uri: str = "http://localhost:5173"
    siwe_nonce_ttl_seconds: int = 300

    # --- 限流与配额（PRD 12 章：1 分钟分桶，计数落 PG）---
    quota_anon_rpm: int = 10
    quota_user_rpm: int = 30
    # /auth/nonce 单独放宽：登录前一秒一次会让前端重连时卡住
    quota_auth_rpm: int = 20
    quota_rpc_per_minute: int = 200
    quota_steps_per_minute: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()
