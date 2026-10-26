"""管理员 CLI：默认预览；--publish 才使用向量化与 PG。不会自动建表。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg.base import init_engine, dispose_engine
from web3_agent.rag.ingestion_job import prepare, ingest
from web3_agent.rag.models import DocumentSpec, embedding_profile
from web3_agent.rag.knowledge_bases import KnowledgeBaseRepository
from web3_agent.rag.vector_store import PgKnowledgeStore

async def main():
    """解析管理员命令行选项，按预览、发布或停用模式处理文档清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--disable", type=uuid.UUID, help="停用指定文档 UUID（不删除历史内容）")
    args = parser.parse_args()
    settings = get_settings()
    if args.disable:
        if args.publish or args.manifest:
            parser.error("--disable 单独使用")
        init_engine(settings)
        try:
            print(json.dumps({"disabled": await PgKnowledgeStore(embedding_profile(settings)).disable(str(args.disable))}))
        finally:
            await dispose_engine()
        return
    if args.manifest is None:
        parser.error("请提供 manifest JSON")
    specs = [DocumentSpec.model_validate(item) for item in json.loads(args.manifest.read_text())]
    # 全部文件先验证，避免清单中途才发现元数据错误。
    previews = [prepare(spec, args.manifest.resolve().parent, settings) for spec in specs]
    if not args.publish:
        print(json.dumps([{"kb_id": spec.kb_id, "title": spec.title, "chain_id": spec.chain_id, "protocol_id": spec.protocol_id,
                           "doc_category": spec.doc_category, "version": spec.version, "chunks": len(chunks), "reviewed": spec.reviewed}
                          for spec, (chunks, _) in zip(specs, previews)], ensure_ascii=False, indent=2))
        return
    if any(not spec.reviewed for spec in specs):
        parser.error("所有发布文档必须 reviewed=true")
    init_engine(settings)
    try:
        from web3_agent.rag.embedder import Embedder
        knowledge_bases = KnowledgeBaseRepository(embedding_profile(settings))
        await knowledge_bases.check_schema()
        await knowledge_bases.ensure_default()
        embedder, store = Embedder(settings), PgKnowledgeStore(embedding_profile(settings))
        for spec in specs:
            result = await ingest(spec, args.manifest.resolve().parent, settings, embedder, store)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        await dispose_engine()

if __name__ == "__main__":
    asyncio.run(main())
