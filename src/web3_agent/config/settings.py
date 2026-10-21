"""环境配置。全部来自环境变量 / .env，进程中只读。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from web3_agent.config.constants import DEFAULT_MAX_AMOUNT_USD


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "dev"
    log_level: str = "INFO"

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
    conversation_max_messages: int = 20

    # --- 摘要（中期记忆）：窗口挤出多少条才开始压、一次最多压多少 ---
    summary_trigger_min: int = 10
    summary_batch_cap: int = 30
    summary_field_max_chars: int = 200
    summary_field_max_items: int = 50
    summary_render_chars: int = 600
    summary_fail_max: int = 3

    # --- 长期记忆：结构化档案 ---
    profile_render_chars: int = 400
    profile_token_max_items: int = 5

    # --- 长期记忆：自由文本（V2，向量）---
    memory_trigger_messages: int = 10  # ≈5 轮
    memory_batch_cap: int = 20
    memory_fail_max: int = 3
    memory_recall_limit: int = 3
    # 余弦距离上限（= 1 - 相似度）。不是照抄 0.85 那个相似度：bge-m3 对同域短句
    # 的相似度普遍偏高，两个阈值不在同一量纲上。这个值在本机语料上标定过。
    memory_recall_max_distance: float = 0.42
    memory_constraint_max: int = 5
    memory_render_chars: int = 400

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
    jwt_secret: str = "dev-only-change-me-0123456789abcdef"
    jwt_algorithm: str = "HS256"
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
