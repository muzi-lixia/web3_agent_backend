"""Administrator upload/status/search API for the shared knowledge base."""
import re
from pathlib import PurePosixPath
from uuid import UUID
from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException
from starlette.exceptions import HTTPException
from pydantic import ValidationError
from web3_agent.api.deps import Identity, require_identity
from web3_agent.api.schemas.rag_schema import UploadMetadata, SearchRequest, KnowledgeBaseCreate, KnowledgeBaseUpdate
from web3_agent.common.exceptions import AppError, error_tuple
from web3_agent.rag.models import KB_ID_PATTERN


async def require_rag_admin(request: Request, identity: Identity = Depends(require_identity)):
    """验证登录钱包在管理员名单内且知识库管理上传功能已启用。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    settings = request.app.state.container.settings
    if identity.address.casefold() not in {address.casefold() for address in settings.rag_admin_addresses}:
        raise AppError(*error_tuple("E605"))
    if not settings.rag_upload_enabled:
        raise AppError(*error_tuple("E606"))
    return identity


router = APIRouter(prefix="/rag", tags=["rag-admin"], dependencies=[Depends(require_rag_admin)])


@router.post(
    "/documents",
    status_code=202,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file", "metadata"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "metadata": {"type": "string", "description": "UploadMetadata JSON，见 docs/rag-upload-api.md"}
                        }
                    }
                }
            }
        }
    }
)
async def upload_document(
    request: Request,
    identity: Identity = Depends(require_rag_admin),
    idempotency_key: str | None = Header(default=None)
):
    """限制 multipart 请求体大小，校验文件及元数据后把入库任务放入持久队列。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
        idempotency_key: 客户端重试使用的同一请求标识。
    """
    container = request.app.state.container
    maximum = container.settings.rag_upload_max_bytes
    if idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", idempotency_key):
        raise AppError(*error_tuple("E607"))
    length = request.headers.get("content-length")
    if length and (not length.isdecimal() or int(length) > maximum + 65_536):
        raise AppError(*error_tuple("E604"))
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise AppError(*error_tuple("E607"))
    received = 0
    async def bounded_receive():
        """累计接收的请求体字节数，超过上传边界时中止 multipart 解析。"""
        nonlocal received
        message = await request.receive()
        received += len(message.get("body", b""))
        if received > maximum + 65_536:
            # Multipart parser closes its temporary files on MultiPartException.
            raise MultiPartException("RAG_UPLOAD_TOO_LARGE")
        return message
    bounded_request = Request(request.scope, receive=bounded_receive)
    try:
        # 限制 multipart 文件和表单字段数，并用上下文管理器确保临时文件关闭。
        async with bounded_request.form(max_files=1, max_fields=1, max_part_size=16_384) as form:
            if len(form.multi_items()) != 2 or set(form) != {"file", "metadata"}:
                raise AppError(*error_tuple("E607"))
            file, metadata = form["file"], form["metadata"]
            if not isinstance(file, UploadFile) or not isinstance(metadata, str) or len(metadata) > 16_384:
                raise AppError(*error_tuple("E607"))
            filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
            if not filename or len(filename) > 255 or any(ord(c)<32 for c in filename) or PurePosixPath(filename).suffix.lower() not in {".md", ".txt", ".pdf"}:
                raise AppError(*error_tuple("E607"))
            spec = UploadMetadata.model_validate_json(metadata).to_spec(filename)
            if file.size is not None and file.size > maximum:
                raise AppError(*error_tuple("E604"))
            # 最多多读一个字节即可判定文件超限，避免无界读入内存。
            data = await file.read(maximum+1)
            if len(data) > maximum:
                raise AppError(*error_tuple("E604"))
            if not data:
                raise AppError(*error_tuple("E607"))
            job, deduplicated = await container.rag_jobs.enqueue(spec=spec, filename=filename, data=data,
                created_by=identity.user_id, idempotency_key=idempotency_key)
    except HTTPException as exc:
        code = "E604" if exc.detail == "RAG_UPLOAD_TOO_LARGE" else "E607"
        raise AppError(*error_tuple(code)) from exc
    except (ValidationError, ValueError) as exc:
        raise AppError(*error_tuple("E607")) from exc
    # Parsing/embedding intentionally happens outside the HTTP request.
    return JSONResponse(status_code=202, headers={"Location": f"/api/v1/rag/jobs/{job['job_id']}"},
        content=jsonable_encoder({"data": job, "deduplicated": deduplicated,
                                  "status_url": f"/api/v1/rag/jobs/{job['job_id']}"}))


@router.get("/jobs")
async def list_jobs(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=100000), kb_id: str | None = Query(None,pattern=KB_ID_PATTERN)):
    """按知识库和分页参数列出入库任务。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        limit: 本次查询或展示返回条数上限。
        offset: 分页跳过的记录条数。
        kb_id: 目标知识库标识，默认 default。
    """
    return {"data": await request.app.state.container.rag_jobs.list(limit=limit, offset=offset,kb_id=kb_id)}


@router.get("/jobs/{job_id}")
async def job_status(job_id: UUID, request: Request):
    """按任务 UUID 返回入库进度和失败信息。

    Args:
        job_id: 入库任务唯一标识。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    return {"data": await request.app.state.container.rag_jobs.get(str(job_id))}


@router.post("/jobs/{job_id}/retry", status_code=202)
async def retry_job(job_id: UUID, request: Request):
    """把允许重试的失败任务重新放回入库队列。

    Args:
        job_id: 入库任务唯一标识。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    c = request.app.state.container
    return {"data": await c.rag_jobs.retry(str(job_id), c.settings.rag_job_max_attempts)}


@router.get("/documents")
async def list_documents(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=100000), kb_id: str | None = Query(None,pattern=KB_ID_PATTERN)):
    """按知识库分页列出文档元数据和切片数量。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        limit: 本次查询或展示返回条数上限。
        offset: 分页跳过的记录条数。
        kb_id: 目标知识库标识，默认 default。
    """
    return {"data": await request.app.state.container.rag_jobs.documents(limit=limit, offset=offset,kb_id=kb_id)}


@router.post("/documents/{document_id}/disable")
async def disable_document(document_id: UUID, request: Request):
    """停用指定文档并推进索引版本，使后续检索排除该文档。

    Args:
        document_id: 目标文档唯一标识，关联该文档的所有切片。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    if not await request.app.state.container.rag_store.disable(str(document_id)):
        raise AppError(*error_tuple("E608"))
    return {"data": {"document_id": str(document_id), "status": "disabled"}}


@router.post("/search")
async def search(payload: SearchRequest, request: Request):
    """执行管理员检索验证请求，知识库不可用时转换为统一业务错误。

    Args:
        payload: 经过字段校验的管理或检索请求。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    result = await request.app.state.container.rag_retriever.search(payload.query_str,
        **payload.model_dump(exclude={"query_str"}))
    if result["status"] == "unavailable":
        raise AppError(*error_tuple("E610"))
    return {"data": result}


@router.get("/knowledge-bases")
async def list_knowledge_bases(request: Request):
    """列出管理界面可见的知识库，包含已停用的条目。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    return {"data":{"items":await request.app.state.container.knowledge_bases.list(include_disabled=True)}}


@router.post("/knowledge-bases",status_code=201)
async def create_knowledge_base(payload: KnowledgeBaseCreate, request: Request, identity: Identity=Depends(require_rag_admin)):
    """使用当前嵌入配置创建一个共享知识库。

    Args:
        payload: 经过字段校验的管理或检索请求。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    return {"data":await request.app.state.container.knowledge_bases.create(payload.name,payload.description,identity.user_id)}


@router.patch("/knowledge-bases/{kb_id}")
async def update_knowledge_base(payload: KnowledgeBaseUpdate, request: Request,kb_id: str):
    """校验知识库标识并更新名称、说明或启用状态。

    Args:
        payload: 经过字段校验的管理或检索请求。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        kb_id: 目标知识库标识，默认 default。
    """
    if not re.fullmatch(KB_ID_PATTERN,kb_id):
        raise AppError(*error_tuple("E101"))
    return {"data":await request.app.state.container.knowledge_bases.update(kb_id,payload.model_dump(exclude_unset=True))}


# All current knowledge bases are shared. This is discovery, not an ACL bypass for private libraries.
public_router=APIRouter(prefix="/knowledge-bases",tags=["knowledge-bases"],dependencies=[Depends(require_identity)])


@public_router.get("")
async def available_knowledge_bases(request: Request):
    """为已登录用户列出当前可选的有效共享知识库。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
    """
    c=request.app.state.container
    if not c.settings.rag_enabled:
        return {"data":{"items":[]}}
    return {"data":{"items":await c.knowledge_bases.list()}}
