"""One bounded-concurrency worker per API process; PostgreSQL coordinates processes."""
import asyncio
import logging
from web3_agent.common.exceptions import ModelUnavailable, AppError
from web3_agent.rag.document_loader import parse_document_with_spans, DocumentParseError
from web3_agent.rag.ingestion_job import prepare_content, embed_chunks
from web3_agent.rag.models import DocumentSpec
from web3_agent.rag.jobs import LostClaim

logger = logging.getLogger(__name__)


# 数据职责：每个 API 进程运行的入库消费者，依靠数据库租约协调多进程执行。
class IngestionWorker:
    def __init__(self, settings, embedder, store, jobs):
        # settings：运行配置对象，提供本模块所需开关、阈值和服务地址。
        # embedder：把文本转换为检索向量的服务。
        # store：文档和切片的持久仓储。
        # jobs：入库任务队列仓储。
        """初始化 IngestionWorker 所需的依赖和运行状态。

        Args:
            settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
            embedder: 把文本转换为检索向量的服务。
            store: 文档和切片的持久仓储。
            jobs: 入库任务队列仓储。
        """
        self.settings, self.embedder, self.store, self.jobs = settings, embedder, store, jobs

    async def process(self, job):
        """在任务总超时内完成解析、切块、向量化和发布，并按故障类型记录失败。

        Args:
            job: 本次已领取的后台任务快照。
        """
        job_id, token = job["id"], job["token"]
        try:
            async with asyncio.timeout(self.settings.rag_job_timeout_seconds):
                spec = DocumentSpec(path=job["filename"], **job["metadata"])
                if not spec.reviewed:
                    raise ValueError("发布前必须审核文档")
                await self.store.validate_kb(spec.kb_id)
                await self.jobs.progress(job_id, token, "parsing")
                # PDF 解析可能消耗 CPU，将同步解析移到线程，避免阻塞 API 事件循环。
                content = await asyncio.to_thread(parse_document_with_spans, job["data"], job["filename"],
                    max_bytes=self.settings.rag_upload_max_bytes, max_text_chars=self.settings.rag_max_text_chars)
                await self.jobs.progress(job_id, token, "chunking")
                chunks, checksum, signature = await asyncio.to_thread(
                    prepare_content, spec, content, self.settings)
                # 先判"要不要重建"，再决定花不花这篇文档的向量化。
                document_id, unchanged = await self.store.plan_index(spec, checksum, signature)
                # 内容和索引参数未变化时复用已发布切片，不再支付重复向量化成本。
                if unchanged:
                    await self.jobs.succeed(job_id, token, document_id, 0)
                    return
                await self.jobs.progress(job_id, token, "embedding", total_chunks=len(chunks))
                async def progress(count):
                    """在仍持有有效租约时更新入库阶段和进度，失去资格则抛 LostClaim。

                    Args:
                        count: 已经处理完的切片数量。
                    """
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
        # 返回值：领取并处理过一个任务为 True；没有任务为 False。
        """领取并执行最多一个入库任务，返回本轮是否处理过任务。"""
        job = await self.jobs.claim(timeout_seconds=self.settings.rag_job_timeout_seconds,
                                     max_attempts=self.settings.rag_job_max_attempts)
        if job is None:
            return False
        await self.process(job)
        return True

    async def run(self):
        """持续领取入库任务，空闲或轮询失败时等待配置间隔，取消时退出。"""
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
