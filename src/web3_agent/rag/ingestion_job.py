"""Shared offline pipeline for CLI and durable upload workers."""
import asyncio
import hashlib
import math
from pathlib import Path
from web3_agent.rag.chunker import CHUNKER_VERSION, split_document
from web3_agent.rag.document_loader import load_parsed_document, ParsedDocument
from web3_agent.rag.models import DocumentSpec
from web3_agent.persistence.pg.orm_models import EMBEDDING_DIM


def index_signature(checksum, settings):
    """索引构建指纹：内容 + 切块器 + 切块参数 + 嵌入配置。

    与 checksum 分开是因为两者回答的问题不同：checksum 说"这是不是同一份文档"，
    这个说"现有切片是不是按当前配置切出来的"。改切块参数时前者不变、后者变，
    已发布文档据此原地重建，而不是被当成 unchanged 留下旧切片。

    Args:
        checksum: 文档内容与元数据的摘要。
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    return hashlib.sha256("|".join([
        checksum, CHUNKER_VERSION, str(settings.rag_chunk_tokens),
        str(settings.rag_chunk_overlap), settings.ollama_embed_model,
        str(settings.ollama_embed_dim), settings.rag_embedding_version,
    ]).encode()).hexdigest()


def prepare_content(spec, content, settings):
    # 返回值：(切片列表, 内容与元数据摘要, 索引构建指纹)。
    """检查向量维度，对正文切块并计算内容摘要与索引指纹。

    Args:
        spec: 经过校验的文档入库元数据。
        content: 文档正文字符串或带页码的 ParsedDocument。
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    if settings.ollama_embed_dim != EMBEDDING_DIM:
        raise ValueError("embedding 维度必须与现有 vector(1024) 一致")
    parsed = content if isinstance(content, ParsedDocument) else ParsedDocument(content)
    content = parsed.text
    chunks = split_document(content, chunk_tokens=settings.rag_chunk_tokens,
                            overlap_ratio=settings.rag_chunk_overlap, page_spans=parsed.page_spans)
    if not chunks or len(chunks) > settings.rag_max_chunks:
        raise ValueError("切片为空或超过数量限制，请拆分文档")
    checksum = hashlib.sha256((content + spec.model_dump_json(
        exclude={"path", "reviewed", "kb_id"})).encode()).hexdigest()
    return chunks, checksum, index_signature(checksum, settings)


def prepare(spec: DocumentSpec, base_dir: Path, settings):
    """按清单路径读取文档后执行离线解析和切块准备。

    Args:
        spec: 经过校验的文档入库元数据。
        base_dir: CLI 清单所在目录，用于解析相对文档路径。
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    content = load_parsed_document((base_dir / spec.path).resolve(), max_bytes=settings.rag_upload_max_bytes,
                            max_text_chars=settings.rag_max_text_chars)
    return prepare_content(spec, content, settings)


async def embed_chunks(spec, chunks, settings, embedder, progress=None):
    # 返回值：与输入切片一一对应、通过数量及数值校验的向量列表。
    """分批向量化标题、章节和正文，校验数量、维度和有限数值并回报进度。

    Args:
        spec: 经过校验的文档入库元数据。
        chunks: 本批文档切片，或任务的切片统计数量。
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        embedder: 把文本转换为检索向量的服务。
        progress: 批次向量化后的进度回调。
    """
    vectors = []
    # 每次最多向量化 16 个切片，限制单次模型输入并及时汇报已完成数量。
    for start in range(0, len(chunks), 16):
        batch = await embedder.embed_many([f"{spec.title}\n{chunk.section}\n{chunk.content}" for chunk in chunks[start:start+16]])
        if len(batch) != len(chunks[start:start+16]) or any(
            len(vector) != settings.ollama_embed_dim or not all(math.isfinite(v) for v in vector)
            or not any(v != 0 for v in vector) for vector in batch
        ):
            raise ValueError("embedding 数量、维度或数值不正确")
        vectors.extend(batch)
        if progress:
            await progress(len(vectors))
    return vectors


async def ingest(spec, base_dir, settings, embedder, store):
    """运行 CLI 文档入库流程：准备内容、检查是否变化、向量化并原子发布。

    Args:
        spec: 经过校验的文档入库元数据。
        base_dir: CLI 清单所在目录，用于解析相对文档路径。
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
        embedder: 把文本转换为检索向量的服务。
        store: 文档和切片的持久仓储。
    """
    if not spec.reviewed:
        raise ValueError("发布前需要 reviewed=true，由管理员核验来源与规则")
    await store.validate_kb(spec.kb_id)
    chunks, checksum, signature = await asyncio.to_thread(prepare, spec, base_dir, settings)
    # 先问一句"用不用重建"，再决定要不要花掉整篇文档的向量化。
    # 以前是算完向量、进 publish 才发现 unchanged，白烧一遍模型。
    document_id, unchanged = await store.plan_index(spec, checksum, signature)
    # 内容和索引参数未变化时复用已发布切片，不再支付重复向量化成本。
    if unchanged:
        return {"document_id": document_id, "chunks": 0, "unchanged": True}
    vectors = await embed_chunks(spec, chunks, settings, embedder)
    return await store.publish(spec, chunks, vectors, checksum, signature=signature)
