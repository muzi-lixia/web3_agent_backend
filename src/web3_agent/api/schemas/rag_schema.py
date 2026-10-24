"""Upload metadata excludes server-side file paths."""
from datetime import datetime
from typing import Literal
from web3_agent.rag.models import DEFAULT_KB_ID, KB_ID_PATTERN
from pydantic import BaseModel, ConfigDict, Field, model_validator
from web3_agent.rag.models import DocumentSpec


# 数据职责：管理员上传文档的元数据；禁止接收未定义额外字段。
class UploadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: str = Field(default=DEFAULT_KB_ID,pattern=KB_ID_PATTERN)
    # title：文档展示标题，也参与向量化文本和本地文档身份计算。
    title: str
    # source_url：文档来源链接，仅用于溯源展示，不会自动抓取该链接。
    source_url: str = ""
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int | None = None
    # protocol_id：协议标识；为空表示文档或查询不限定协议。
    protocol_id: str | None = None
    # doc_category：文档分类标签，用于精确过滤，例如概念、教程或 FAQ。
    doc_category: str
    # version：文档业务版本或 SIWE 协议版本；具体由所属模型决定。
    version: str
    # reviewed：管理员是否已审核该文档；发布流程要求为真。
    reviewed: bool
    # valid_from：文档生效时间；检索只接受已生效的版本。
    valid_from: datetime
    # valid_until：文档失效时间；为空表示未设置到期时间。
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def valid_document(self):
        """校验上传元数据已审核、已生效且尚未过期。"""
        spec = self.to_spec("upload.txt")
        if not spec.reviewed:
            raise ValueError("发布前需要 reviewed=true")
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if spec.valid_from > now or (spec.valid_until and spec.valid_until <= now):
            raise ValueError("仅允许已生效且未过期文档")
        return self

    def to_spec(self, filename):
        """把 HTTP 上传元数据与安全文件名合成内部 DocumentSpec。

        Args:
            filename: 安全文件名，用于选择格式解析器。
        """
        return DocumentSpec(path=filename, **self.model_dump())


# 数据职责：检索调试请求，包含问题、知识库和可选精确范围。
class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: str = Field(default=DEFAULT_KB_ID,pattern=KB_ID_PATTERN)
    # query_str：待检索的问题文本，允许 1～4000 个字符。
    query_str: str = Field(min_length=1, max_length=4000)
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int | None = Field(default=None, gt=0, le=2147483647)
    # protocol_id：协议标识；为空表示文档或查询不限定协议。
    protocol_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    # doc_category：文档分类标签，用于精确过滤，例如概念、教程或 FAQ。
    doc_category: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    # top_k：请求返回的证据条数，仍受检索配置上限限制。
    top_k: int = Field(default=3, ge=1, le=5)


# 数据职责：创建共享知识库所需的展示信息。
class KnowledgeBaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)
    # name：新知识库展示名称，去除首尾空白后长度为 1～100。
    name: str = Field(min_length=1,max_length=100)
    # description：知识库的说明文本，用于管理界面展示。
    description: str = Field(default="",max_length=2000)


# 数据职责：知识库部分更新契约；只处理客户端显式提交的字段。
class KnowledgeBaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)
    # name：要修改的知识库展示名称；未提交时保持原值。
    name: str | None = Field(default=None,min_length=1,max_length=100)
    # description：知识库的说明文本，用于管理界面展示。
    description: str | None = Field(default=None,max_length=2000)
    # status：记录生命周期状态；允许的状态由对应仓储及业务流程决定。
    status: Literal["active","disabled"] | None = None

    @model_validator(mode="after")
    def no_null_updates(self):
        """要求更新请求至少包含一个字段，并拒绝显式 null 更新。"""
        if not self.model_fields_set or any(getattr(self,key) is None for key in self.model_fields_set):
            raise ValueError("修改字段不能为空")
        return self
