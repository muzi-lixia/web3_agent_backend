"""One bounded-concurrency worker per API process; PostgreSQL coordinates processes."""
import asyncio
import logging
from web3_agent.common.exceptions import ModelUnavailable, AppError
from web3_agent.rag.document_loader import parse_document_with_spans, DocumentParseError
from web3_agent.rag.ingestion_job import prepare_content, embed_chunks
from web3_agent.rag.models import DocumentSpec
from web3_agent.rag.jobs import LostClaim

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, settings, embedder, store, jobs):
        self.settings, self.embedder, self.store, self.jobs = settings, embedder, store, jobs

    async def process(self, job):
        job_id, token = job["id"], job["token"]
        try:
            async with asyncio.timeout(self.settings.rag_job_timeout_seconds):
                spec = DocumentSpec(path=job["filename"], **job["metadata"])
                if not spec.reviewed:
                    raise ValueError("发布前必须审核文档")
                await self.store.validate_kb(spec.kb_id)
                await self.jobs.progress(job_id, token, "parsing")
                content = await asyncio.to_thread(parse_document_with_spans, job["data"], job["filename"],
                    max_bytes=self.settings.rag_upload_max_bytes, max_text_chars=self.settings.rag_max_text_chars)
                await self.jobs.progress(job_id, token, "chunking")
                chunks, checksum, signature = await asyncio.to_thread(
                    prepare_content, spec, content, self.settings)
                # 先判"要不要重建"，再决定花不花这篇文档的向量化。
                document_id, unchanged = await self.store.plan_index(spec, checksum, signature)
                if unchanged:
                    await self.jobs.succeed(job_id, token, document_id, 0)
                    return
                await self.jobs.progress(job_id, token, "embedding", total_chunks=len(chunks))
                async def progress(count):
                    await self.jobs.progress(job_id, token, "embedding", embedded_chunks=count)
                vectors = await embed_chunks(spec, chunks, self.settings, self.embedder, progress)
                await self.jobs.progress(job_id, token, "indexing")
                # Atomic vector publication and succeeded job state, with claim fencing.
                await self.store.publish(spec, chunks, vectors, checksum, signature=signature,
                                         job_id=job_id, claim_token=token)
        except asyncio.CancelledError:
            # Leave the durable lease for another worker to reclaim on restart.
            raise
        except LostClaim:
            logger.warning("rag job claim lost job=%s", job_id)
        except AppError as exc:
            await self.jobs.fail(job_id,token,exc.code,exc.message)
        except DocumentParseError as exc:
            await self.jobs.fail(job_id, token, "PARSE_FAILED", str(exc))
        except ModelUnavailable:
            await self.jobs.fail(job_id, token, "EMBEDDING_UNAVAILABLE", "向量模型不可用，请检查本地 Ollama 后重试")
        except TimeoutError:
            await self.jobs.fail(job_id, token, "PROCESSING_TIMEOUT", "处理超时，请拆分文档或检查模型后重试")
        except ValueError:
            logger.exception("rag validation failed job=%s", job_id)
            await self.jobs.fail(job_id, token, "VALIDATION_FAILED", "文档版本、有效期、切片或向量校验失败，请检查后重新上传")
        except Exception:
            logger.exception("rag ingestion failed job=%s", job_id)
            await self.jobs.fail(job_id, token, "INGESTION_FAILED", "文档处理或入库失败，请检查服务后重试")

    async def run_once(self):
        job = await self.jobs.claim(timeout_seconds=self.settings.rag_job_timeout_seconds,
                                     max_attempts=self.settings.rag_job_max_attempts)
        if job is None:
            return False
        await self.process(job)
        return True

    async def run(self):
        while True:
            try:
                busy = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("rag worker poll failed")
                busy = False
            if not busy:
                await asyncio.sleep(self.settings.rag_worker_poll_seconds)
