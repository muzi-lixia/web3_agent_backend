"""入库契约与工具证据；所有生产元数据由管理员提供。"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

DEFAULT_KB_ID = "default"
KB_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"

class EmbeddingProfile(BaseModel):
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dimension: Literal[1024] = 1024
    embedding_version: str = "v1"


def embedding_profile(settings):
    return EmbeddingProfile(embedding_model=settings.ollama_embed_model,
        embedding_dimension=settings.ollama_embed_dim, embedding_version=settings.rag_embedding_version)


class DocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kb_id: str = Field(default=DEFAULT_KB_ID, pattern=KB_ID_PATTERN)
    path: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl | Literal[""] = ""
    chain_id: int | None = Field(default=None, gt=0, le=2147483647)
    protocol_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    doc_category: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(min_length=1, max_length=64)
    reviewed: bool = False
    valid_from: datetime
    valid_until: datetime | None = None

    @field_validator("valid_from", "valid_until")
    @classmethod
    def timezone_required(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("文档时间必须带时区")
        return value

    @model_validator(mode="after")
    def dates(self):
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("文档过期时间必须晚于生效时间")
        return self

class Citation(BaseModel):
    kb_id: str = DEFAULT_KB_ID
    token_count: int = 0
    token_count_method: str = "approx_v1"
    char_start: int | None = None
    char_end: int | None = None
    page_number: int | None = None
    page_end: int | None = None
    metadata: dict = Field(default_factory=dict)
    doc_category: str
    valid_from: str = ""
    id: str
    chunk_id: str
    title: str
    section: str
    source_url: str
    version: str
    chain_id: int | None = None
    protocol_id: str | None = None
    updated_at: str
    content: str
    distance: float | None
