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
    conversation_ttl_seconds: int = 7200
    conversation_max_messages: int = 20

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

    # --- LangSmith（SaaS）---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "web3-agent-dev"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_sample_rate: float = 1.0

    # --- 鉴权（V1a 只读匿名可访问；写链路需要）---
    jwt_secret: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"


@lru_cache
def get_settings() -> Settings:
    return Settings()
