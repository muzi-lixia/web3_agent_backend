"""Shared offline pipeline for CLI and durable upload workers."""
import asyncio
import hashlib
import math
from pathlib import Path
from web3_agent.rag.chunker import split_document
from web3_agent.rag.document_loader import load_parsed_document, ParsedDocument
from web3_agent.rag.models import DocumentSpec
from web3_agent.persistence.pg.orm_models import EMBEDDING_DIM


def prepare_content(spec, content, settings):
    if settings.ollama_embed_dim != EMBEDDING_DIM:
        raise ValueError("embedding 维度必须与现有 vector(1024) 一致")
    parsed = content if isinstance(content, ParsedDocument) else ParsedDocument(content)
    content = parsed.text
    chunks = split_document(content, chunk_tokens=settings.rag_chunk_tokens,
                            overlap_ratio=settings.rag_chunk_overlap, page_spans=parsed.page_spans)
    if not chunks or len(chunks) > settings.rag_max_chunks:
        raise ValueError("切片为空或超过数量限制，请拆分文档")
    fingerprint = hashlib.sha256((content + spec.model_dump_json(exclude={"path", "reviewed", "kb_id"})).encode()).hexdigest()
    return chunks, fingerprint


def prepare(spec: DocumentSpec, base_dir: Path, settings):
    content = load_parsed_document((base_dir / spec.path).resolve(), max_bytes=settings.rag_upload_max_bytes,
                            max_text_chars=settings.rag_max_text_chars)
    return prepare_content(spec, content, settings)


async def embed_chunks(spec, chunks, settings, embedder, progress=None):
    vectors = []
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
    if not spec.reviewed:
        raise ValueError("发布前需要 reviewed=true，由管理员核验来源与规则")
    await store.validate_kb(spec.kb_id)
    chunks, checksum = await asyncio.to_thread(prepare, spec, base_dir, settings)
    vectors = await embed_chunks(spec, chunks, settings, embedder)
    return await store.publish(spec, chunks, vectors, checksum)
