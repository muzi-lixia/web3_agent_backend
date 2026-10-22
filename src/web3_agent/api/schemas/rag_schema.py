"""Upload metadata excludes server-side file paths."""
from datetime import datetime
from typing import Literal
from web3_agent.rag.models import DEFAULT_KB_ID, KB_ID_PATTERN
from pydantic import BaseModel, ConfigDict, Field, model_validator
from web3_agent.rag.models import DocumentSpec


class UploadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kb_id: str = Field(default=DEFAULT_KB_ID,pattern=KB_ID_PATTERN)
    title: str
    source_url: str = ""
    chain_id: int | None = None
    protocol_id: str | None = None
    doc_category: str
    version: str
    reviewed: bool
    valid_from: datetime
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def valid_document(self):
        spec = self.to_spec("upload.txt")
        if not spec.reviewed:
            raise ValueError("发布前需要 reviewed=true")
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if spec.valid_from > now or (spec.valid_until and spec.valid_until <= now):
            raise ValueError("仅允许已生效且未过期文档")
        return self

    def to_spec(self, filename):
        return DocumentSpec(path=filename, **self.model_dump())


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kb_id: str = Field(default=DEFAULT_KB_ID,pattern=KB_ID_PATTERN)
    query_str: str = Field(min_length=1, max_length=4000)
    chain_id: int | None = Field(default=None, gt=0, le=2147483647)
    protocol_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    doc_category: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    top_k: int = Field(default=3, ge=1, le=5)


class KnowledgeBaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)
    name: str = Field(min_length=1,max_length=100)
    description: str = Field(default="",max_length=2000)


class KnowledgeBaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)
    name: str | None = Field(default=None,min_length=1,max_length=100)
    description: str | None = Field(default=None,max_length=2000)
    status: Literal["active","disabled"] | None = None

    @model_validator(mode="after")
    def no_null_updates(self):
        if not self.model_fields_set or any(getattr(self,key) is None for key in self.model_fields_set):
            raise ValueError("修改字段不能为空")
        return self
