"""入库契约与工具证据；所有生产元数据由管理员提供。"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

# DEFAULT_KB_ID：系统默认共享知识库的稳定标识。
DEFAULT_KB_ID = "default"
# KB_ID_PATTERN：知识库标识允许的字符和长度规则。
KB_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"

# 数据职责：文档索引和查询共同使用的嵌入配置契约。
class EmbeddingProfile(BaseModel):
    # embedding_provider：生成向量的服务类型，本版本为 ollama。
    embedding_provider: str = "ollama"
    # embedding_model：生成和查询向量所用模型；同一知识库必须保持兼容。
    embedding_model: str = "bge-m3"
    # embedding_dimension：向量维度，当前数据库列固定为 1024。
    embedding_dimension: Literal[1024] = 1024
    # embedding_version：嵌入配置的业务版本，区分同名模型的不同索引配置。
    embedding_version: str = "v1"


def embedding_profile(settings):
    """从运行配置生成知识库嵌入配置快照。

    Args:
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    return EmbeddingProfile(embedding_model=settings.ollama_embed_model,
        embedding_dimension=settings.ollama_embed_dim, embedding_version=settings.rag_embedding_version)


# 数据职责：经过校验的文档入库描述，供 CLI 和上传 worker 共用。
class DocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: str = Field(default=DEFAULT_KB_ID, pattern=KB_ID_PATTERN)
    # path：CLI 文档路径或上传文件名，不由客户端指定服务端任意文件路径。
    path: str = Field(min_length=1)
    # title：文档展示标题，也参与向量化文本和本地文档身份计算。
    title: str = Field(min_length=1, max_length=200)
    # source_url：文档来源链接，仅用于溯源展示，不会自动抓取该链接。
    source_url: HttpUrl | Literal[""] = ""
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int | None = Field(default=None, gt=0, le=2147483647)
    # protocol_id：协议标识；为空表示文档或查询不限定协议。
    protocol_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    # doc_category：文档分类标签，用于精确过滤，例如概念、教程或 FAQ。
    doc_category: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    # version：管理员维护的文档业务版本；同版本内容变化时应提升版本号。
    version: str = Field(min_length=1, max_length=64)
    # reviewed：管理员是否已审核该文档；发布流程要求为真。
    reviewed: bool = False
    # valid_from：文档生效时间；检索只接受已生效的版本。
    valid_from: datetime
    # valid_until：文档失效时间；为空表示未设置到期时间。
    valid_until: datetime | None = None

    @field_validator("valid_from", "valid_until")
    @classmethod
    def timezone_required(cls, value):
        """拒绝不带时区的文档时间，避免不同环境解释出不同生效边界。

        Args:
            value: 待校验、转换或缓存的值。
        """
        if value is not None and value.tzinfo is None:
            raise ValueError("文档时间必须带时区")
        return value

    @model_validator(mode="after")
    def dates(self):
        """校验文档到期时间严格晚于生效时间。"""
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("文档过期时间必须晚于生效时间")
        return self

# 数据职责：可返回给回答模型和前端的检索证据，包含正文和来源定位。
class Citation(BaseModel):
    # kb_id：所属知识库标识；检索、上传和文档发布都按此字段隔离范围。
    kb_id: str = DEFAULT_KB_ID
    # token_count：切片的近似 token 数，用于估算切块大小，并非模型精确计费值。
    token_count: int = 0
    # token_count_method：近似 token 计数算法标识，便于解释计数口径。
    token_count_method: str = "approx_v1"
    # char_start：切片在归一化文档正文中的起始字符偏移，包含该位置。
    char_start: int | None = None
    # char_end：切片在归一化文档正文中的结束字符偏移，不包含该位置。
    char_end: int | None = None
    # page_number：切片起始 PDF 页码，从 1 开始；非 PDF 或无页码时为空。
    page_number: int | None = None
    # page_end：切片结束 PDF 页码，跨页时用于展示完整来源范围。
    page_end: int | None = None
    # metadata：切片额外来源信息，不是模型可执行的系统指令。
    metadata: dict = Field(default_factory=dict)
    # doc_category：文档分类标签，用于精确过滤，例如概念、教程或 FAQ。
    doc_category: str
    # valid_from：文档生效时间；检索只接受已生效的版本。
    valid_from: str = ""
    # id：本轮展示用引用编号，例如 S1；不是数据库切片 ID。
    id: str
    # chunk_id：文档切片的稳定标识，用于去重和引用定位。
    chunk_id: str
    # title：文档展示标题，也参与向量化文本和本地文档身份计算。
    title: str
    # section：切片所属章节或 FAQ 问题标题。
    section: str
    # source_url：文档来源链接，仅用于溯源展示，不会自动抓取该链接。
    source_url: str
    # version：被引用的文档业务版本，用于回溯回答依据。
    version: str
    # chain_id：EVM 网络编号，如 1 为以太坊主网；可空的场景表示未限定网络。
    chain_id: int | None = None
    # protocol_id：协议标识；为空表示文档或查询不限定协议。
    protocol_id: str | None = None
    # updated_at：最近更新时间，用于展示、追踪变更或判断数据新鲜度。
    updated_at: str
    # content：本轮允许用于回答的切片证据正文。
    content: str
    # distance：查询向量与切片向量的余弦距离，越小越相似；关键词结果可为空。
    distance: float | None
