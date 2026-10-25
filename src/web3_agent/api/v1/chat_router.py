"""POST /api/v1/chat：聊天主入口。只做参数、鉴权、调 service、组装响应。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from web3_agent.api.deps import Identity, chat_rate_limit, require_identity
from web3_agent.api.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    ClearResp,
    HistoryItem,
    HistoryResp,
    IntentPreview,
    PurgeResp,
)
from web3_agent.common.exceptions import (
    E102_CHAIN_NOT_ALLOWED,
    E509_CHAIN_IDENTITY_MISMATCH,
    E510_TURN_BUDGET,
    E511_DUPLICATE_REQUEST,
    E512_IDEMPOTENCY_CONFLICT,
    AppError,
)
from web3_agent.config.constants import QuotaMetric
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import (
    conversation_repo,
    idempotency_repo,
    memory_repo,
    profile_repo,
)
from web3_agent.common.metering import Resource, start_metering, take_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(chat_rate_limit)])

# _KIND_TO_TYPE：内部回答种类到 HTTP 响应种类的转换映射。
_KIND_TO_TYPE = {
    "answer": "answer",
    "ask": "ask",
    "reject": "answer",
    "intent_confirm": "intent_confirm",
}

# 幂等占位的存活上限 = 整轮硬超时 + 这段余量。余量留给超时判定后的事务收尾，
# 余量之后仍在 running 的行，只可能是进程在写回前就不在了。
# _IDEMPOTENCY_STALE_GRACE：聊天硬超时之外预留的幂等收尾宽限，单位为秒。
_IDEMPOTENCY_STALE_GRACE = 30

# 上一轮还在跑时告诉客户端等多久再来。真实一轮十几秒，3 秒足够让第一次
# 重试落在「还在跑」之外又不至于空等；拿不到答案就再来一次，有上限兜底。
# _IDEMPOTENCY_RETRY_AFTER：幂等请求仍在处理时建议客户端等待的秒数。
_IDEMPOTENCY_RETRY_AFTER = 3

# _SSE_HEARTBEAT_SECONDS：两次事件之间允许的最长静默。图执行中途没有新进展时发一行注释，
# 免得中间的代理按「长时间无数据」把连接断掉。
_SSE_HEARTBEAT_SECONDS = 15

# 后台任务（摘要、长期记忆抽取）。不 await：都慢，不能拖住回复。
# 集合持有引用，否则任务可能被 GC 掉。
# _BACKGROUND：对尚未结束的请求辅助任务保持强引用，结束后自动移除。
_BACKGROUND: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    """后台任务的消耗单独记账：它跑在请求结束之后，不能算进任何一轮对话。

    Args:
        coro: 待派发为后台任务的协程。
    """

    async def tracked():
        """包装后台协程，确保退出时取走并记录该任务的资源用量。"""
        try:
            return await coro
        finally:
            # 先取走并清空本范围的资源账本，防止成功、取消或后台路径重复结算。
            usage = take_usage()
            if any(usage.values()):
                logger.info("background usage=%s", {r.value: n for r, n in usage.items() if n})

    task = asyncio.create_task(tracked())
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


async def _start_aux_tasks(container, identity: Identity) -> None:
    """准备摘要与记忆抽取任务。

    两条水位线互不依赖：一条探测失败不挡住另一条。任何失败都只落日志，
    因为这时候答案已经在库里、响应也已经在路上。

    Args:
        container: 应用生命周期内共享的依赖集合。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    try:
        job = await container.summarizer.collect(identity.user_id)
    except Exception:  # noqa: BLE001
        logger.exception("summary collect failed user=%s", identity.user_id)
    else:
        if job is not None:
            _spawn_background(
                container.summarizer.run(
                    siwe_user_id=identity.user_id,
                    own_address=identity.address,
                    job=job,
                )
            )
    try:
        job = await container.memory.collect(identity.user_id)
    except Exception:  # noqa: BLE001
        logger.exception("memory collect failed user=%s", identity.user_id)
    else:
        if job is not None:
            _spawn_background(
                container.memory.run(siwe_user_id=identity.user_id, job=job)
            )


async def _settle(
    container,
    *,
    quota_key: str | None,
    release: tuple[str, str] | None = None,
) -> None:
    """把本轮已经发生的消耗记账，顺带撤掉没跑成的幂等占位。

    放在 finally 里：超时、异常、取消时这一轮照样花掉了模型和下游请求，
    按成功返回后的 outcome 记账会把这些消耗全部漏掉。

    Args:
        container: 应用生命周期内共享的依赖集合。
        quota_key: 本轮资源消耗归属的配额主体键。
        release: 失败时需要释放的 (用户 ID, 幂等键) 元组。
    """
    # 先取走并清空本范围的资源账本，防止成功、取消或后台路径重复结算。
    usage = take_usage()
    steps = usage[Resource.MODEL]
    downstream = usage[Resource.RPC] + usage[Resource.EXPLORER] + usage[Resource.EMBEDDING]
    try:
        if quota_key and (steps or downstream):
            async with session_scope() as db:
                await container.quota.charge(
                    db, key=quota_key, metric=QuotaMetric.STEPS, amount=steps
                )
                await container.quota.charge(
                    db, key=quota_key, metric=QuotaMetric.RPC_CALLS, amount=downstream
                )
    except Exception:  # noqa: BLE001 结算失败不能盖掉本轮的结果
        logger.exception("quota settlement failed key=%s usage=%s", quota_key, usage)
    if release is None:
        return
    try:
        async with session_scope() as db:
            await idempotency_repo.release(db, siwe_user_id=release[0], key=release[1])
    except Exception:  # noqa: BLE001 放不掉也还有一天后的过期清理兜着
        logger.exception("idempotency release failed key=%s", release[1])


async def _settle_turn(
    container,
    *,
    quota_key: str | None,
    release: tuple[str, str] | None = None,
) -> None:
    """结算收尾。正常路径当场写完；请求被取消时把写入交给独立任务，别把它一起带走。

    Args:
        container: 应用生命周期内共享的依赖集合。
        quota_key: 本轮资源消耗归属的配额主体键。
        release: 失败时需要释放的 (用户 ID, 幂等键) 元组。
    """
    task = asyncio.create_task(_settle(container, quota_key=quota_key, release=release))
    try:
        # 只隔离结算任务的取消传播，HTTP 等待者被取消时写账仍可完成。
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # 取消的是这个等待者，不是那次写入；留个引用，免得任务中途被 GC 掉。
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)
        raise


def _sse(event: str, data: dict) -> str:
    """组装一帧事件流。正文压成单行 JSON，客户端按空行切帧即可。

    Args:
        event: 事件名，客户端据此分派。
        data: 该事件的负载。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _frames(
    queue: asyncio.Queue[tuple[str, dict] | None], task: asyncio.Task
) -> AsyncIterator[str]:
    """把队列里的事件转发成 SSE 帧，直到收到结束标记。

    客户端断开时生成器被取消，顺带取消还在跑的那一轮：没人在听了，
    继续跑只会把模型和 RPC 的额度白花掉。

    Args:
        queue: 本轮产生的事件；None 表示结束。
        task: 跑这一轮的任务，用于断开时取消。
    """
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
            except TimeoutError:
                # 图执行期间可能长时间没有新阶段，用注释行占住连接。
                yield ": ping\n\n"
                continue
            if item is None:
                break
            event, data = item
            yield _sse(event, data)
        yield _sse("done", {})
    finally:
        if not task.done():
            task.cancel()


def _split_citations(calls: list[dict] | None) -> tuple[list[dict] | None, list[dict]]:
    """引用正文从 tool_calls 里提出来单独给。

    库里仍然存在 tool_calls 内（刷新后要能复原引用了哪几段），
    但响应里只发一份 —— 同一段正文发两遍等于把响应体积翻倍。

    Args:
        calls: 历史消息保存的工具调用元数据。

    Returns:
        (去掉引用正文的工具轨迹, 引用列表)，避免传输重复正文。
    """
    if not calls:
        return calls, []
    citations: list[dict] = []
    kept: list[dict] = []
    for call in calls:
        if call.get("name") == "rag_general_knowledge" and call.get("citations"):
            citations = call["citations"]
            call = {key: value for key, value in call.items() if key != "citations"}
        kept.append(call)
    return kept, citations


def _request_hash(payload: ChatRequest) -> str:
    """同一个 key 换了内容就拒绝：拿旧答案顶新问题，用户只会觉得消息丢了。

    Args:
        payload: 经过字段校验的聊天请求。

    Returns:
        请求内容的 SHA-256 十六进制摘要。
    """
    wallet = payload.wallet.model_dump(by_alias=True) if payload.wallet else None
    material = json.dumps(
        [payload.user_message, payload.knowledge_base_id, wallet],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


@router.post("")
async def chat(
    payload: ChatRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
) -> StreamingResponse:
    # 每帧的 data 都是 JSON：stage 报进度，result 是本轮完整结论，error 表示中途失败。
    """以事件流返回一轮对话：先报阶段，最后下发完整结论。

    身份、网络与幂等占位在开始推流之前完成，那类错误仍用 HTTP 状态码表达；
    推流之后的工作失败，只能作为 error 事件送出。

    Args:
        payload: 经过字段校验的聊天请求。
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    container = request.app.state.container

    if payload.wallet and payload.wallet.chain_id not in container.settings.allowed_chain_ids:
        raise AppError(*E102_CHAIN_NOT_ALLOWED)

    chain_id = payload.wallet.chain_id if payload.wallet else None
    if chain_id is not None and chain_id != identity.chain_id:
        raise AppError(*E509_CHAIN_IDENTITY_MISMATCH)
    # 请求没带 wallet 时用登录身份里的链：档案要记代币属于哪条链，缺了就只能丢弃。
    effective_chain = chain_id if chain_id is not None else identity.chain_id

    # 幂等键：客户端重试时带同一个值，服务端只跑一轮。占位必须先提交出去，
    # 否则并发的第二个请求看不到第一个已经在跑。
    key = payload.idempotency_key
    cached: dict | None = None
    if key:
        async with session_scope() as db:
            # 先在短事务中提交幂等占位，再开始慢调用，确保并发重试看得见处理中状态。
            state, cached = await idempotency_repo.claim(
                db,
                siwe_user_id=identity.user_id,
                key=key,
                request_hash=_request_hash(payload),
                # 占位从插入到 complete 最多经历一整轮 + 收尾；超出还没写的行一定是孤儿。
                stale_after=container.settings.turn_timeout_seconds + _IDEMPOTENCY_STALE_GRACE,
            )
        if state == "conflict":
            # 换个 key 重新发起就能跑，同一个 key 换内容重试多少次都是这个结果。
            raise AppError(*E512_IDEMPOTENCY_CONFLICT)
        if state not in ("new", "done"):
            raise AppError(
                *E511_DUPLICATE_REQUEST,
                retryable=True,
                retry_after=_IDEMPOTENCY_RETRY_AFTER,
            )

    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    async def emit(stage: str) -> None:
        """把当前阶段写进事件队列。

        Args:
            stage: 阶段标识，由客户端解释成展示文案。
        """
        await queue.put(("stage", {"stage": stage}))

    async def turn() -> None:
        """跑完这一轮，把结论或失败放进队列。

        响应头在这之前已经发出去了，所以这里的异常不能再改 HTTP 状态码。
        """
        if cached is not None:
            await queue.put(("result", cached))
            await queue.put(None)
            return
        # 工具在 gather 的子任务里累加计数，必须在这里（父上下文）先开一次。
        start_metering(downstream=container.settings.turn_max_downstream_calls)
        outcome = None
        accepted = False
        # 一轮对话要十几秒（意图模型 + 最多 5 步 ReAct），这里不能持连接：
        # agent 自己按需开短事务，连接只在真正读写的几十毫秒里被占住。
        # 整轮时间是第四条边界：步数和调用数都有限，链路卡在模型或上游时还得有个上限。
        deadline = asyncio.timeout(container.settings.turn_timeout_seconds)
        try:
            try:
                # 整轮超时覆盖模型及工具调用，达到截止线时中止本轮处理。
                async with deadline:
                    outcome = await container.agent.handle(
                        siwe_user_id=identity.user_id,
                        wallet_address=identity.address,
                        chain_id=effective_chain,
                        user_message=payload.user_message,
                        knowledge_base_id=payload.knowledge_base_id,
                        on_stage=emit,
                    )
            except TimeoutError as exc:
                # 只处理自己这条超时：模型或上游自己抛的 TimeoutError 已经被各自包装过，
                # 漏到这里的必须原样上抛，不能被误报成整轮超时。
                if not deadline.expired():
                    raise
                raise AppError(*E510_TURN_BUDGET) from exc
            accepted = not outcome.discarded

            response = ChatResponse(
                type=_KIND_TO_TYPE.get(outcome.kind, "answer"),
                content=outcome.content,
                task_id=outcome.task_id,
                intent_preview=(
                    IntentPreview.model_validate(outcome.intent_preview)
                    if outcome.intent_preview
                    else None
                ),
                data={
                    "steps": outcome.steps,
                    "toolCalls": [call.model_dump() for call in outcome.tool_calls],
                    **({"citations": outcome.citations} if outcome.citations else {}),
                    **({"discarded": True} if outcome.discarded else {}),
                },
            )
            # 失效轮次不进去重表：记下来就成了「这个 key 的答案是空的」，
            # 客户端重试永远拿到空答案，再也跑不起来。
            if key and not outcome.discarded:
                try:
                    async with session_scope() as db:
                        # 把已组装的响应保存到去重记录，后续相同请求可直接返回。
                        await idempotency_repo.complete(
                            db,
                            siwe_user_id=identity.user_id,
                            key=key,
                            response=response.model_dump(by_alias=True),
                        )
                except Exception:  # noqa: BLE001 答案已经在库里，去重记录不能拖垮这次响应
                    logger.exception("idempotency complete failed key=%s", key)
            await queue.put(("result", response.model_dump(by_alias=True)))
        except AppError as exc:
            await queue.put(
                (
                    "error",
                    {
                        "code": exc.code,
                        "message": exc.message,
                        "retryable": exc.retryable,
                        "retryAfter": exc.retry_after,
                    },
                )
            )
        except Exception:  # noqa: BLE001 流已开始，只能作为事件告知
            logger.exception("chat turn failed")
            await queue.put(
                ("error", {"code": "E900", "message": "服务内部错误", "retryable": False})
            )
        finally:
            # 这一轮没跑完就把占位撤掉，客户端拿同一个 key 重试才能真的重跑。
            await _settle_turn(
                container,
                quota_key=getattr(request.state, "quota_key", None),
                release=(identity.user_id, key) if key and not accepted else None,
            )
            # 答案已经提交事务，后台辅助任务是纯附加动作：这里任何异常都不该变成
            # 用户看到的错误 —— 否则库里有一轮完整答案，客户端却以为失败并重试。
            # 全部挪进后台协程，包括两次 collect 的水位线探测。
            # 失效轮次没有内容可压缩，跑一遍只是白花模型调用。
            if outcome is not None and not outcome.discarded:
                _spawn_background(_start_aux_tasks(container, identity))
            await queue.put(None)

    task = asyncio.create_task(turn())
    return StreamingResponse(
        _frames(queue, task),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history", response_model=HistoryResp, response_model_by_alias=True)
async def history(
    before: int | None = None,
    limit: int = 20,
    identity: Identity = Depends(require_identity),
) -> HistoryResp:
    """按游标读取历史页，恢复工具引用，并按前端展示顺序返回。

    Args:
        before: 历史分页的消息 ID 游标；为空时读取最新页。
        limit: 本次查询或展示返回条数上限。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    limit = max(1, min(limit, 50))
    async with session_scope() as db:
        rows = await conversation_repo.list_page(db, identity.user_id, limit, before)
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = []
    for row in reversed(rows):
        tool_calls, citations = _split_citations(row.tool_calls)
        items.append(
            HistoryItem(
                id=row.id,
                role=row.role,
                content=row.content,
                tool_calls=tool_calls,
                task_id=row.task_id,
                intent_preview=(
                    IntentPreview.model_validate(row.intent_preview)
                    if row.intent_preview
                    else None
                ),
                citations=citations,
                created_at=row.created_at,
            )
        )
    return HistoryResp(items=items, has_more=has_more)


@router.post("/clear", response_model=ClearResp, response_model_by_alias=True)
async def clear(request: Request, identity: Identity = Depends(require_identity)) -> ClearResp:
    """推进会话版本，清除消息和聊天幂等缓存，保留长期档案与记忆。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    container = request.app.state.container
    started = datetime.now(timezone.utc)
    async with session_scope() as db:
        # 推进生命周期版本，同时拿排他行锁。正在跑的那轮对话、后台摘要、
        # 记忆抽取都会在写回时发现版本变了并放弃写入 —— 用户在删除，
        # 这几秒内产出的东西一律不落库。advisory lock 挡不住它们：那些路径
        # 从来不来拿这把锁，只有 clear 自己跟自己互斥。
        generation = await conversation_repo.bump_generation(db, identity.user_id)
        # 摘要就是 conversation_message 里 role='summary' 的一行，跟消息一起删掉，
        # 不需要再单独清一张表。
        cleared = await conversation_repo.clear_user_messages(db, identity.user_id)
        # 幂等响应里存着上一轮的正文和引用，也就是刚被清掉的内容，必须一起丢。
        # 同一事务里删：清空返回后不可能再点到一条旧答案。
        await idempotency_repo.purge_user(db, siwe_user_id=identity.user_id)
    await container.audit.log(
        "conversation_cleared",
        siwe_user_id=identity.user_id,
        cleared=cleared,
        generation=generation,
    )
    return ClearResp(
        cleared_messages=cleared,
        cleared_summary=False,
        started_at=started,
    )


@router.delete("/memory", response_model=PurgeResp, response_model_by_alias=True)
async def purge_memory(
    request: Request, identity: Identity = Depends(require_identity)
) -> PurgeResp:
    """彻底删除该用户的全部对话数据：消息（含摘要行）、档案、长期记忆、抽取水位线。

    与 `/chat/clear` 的差别是连长期记忆一起删，且不可恢复 —— 合规诉求走这里，
    清空对话走那里。审计日志不删，它要留下「什么时候删过」这条记录本身。

    Args:
        request: 当前 HTTP 请求，可读取应用依赖和请求上下文。
        identity: 经服务端验证的登录用户与钱包身份。
    """
    container = request.app.state.container
    deleted_at = datetime.now(timezone.utc)
    async with session_scope() as db:
        # 与 clear 用同一个版本号：删的过程中不能再有 /chat 或后台任务往这个用户身上写。
        # generation 行本身保留 —— 它只是一个计数器，不含对话内容；
        # 删掉再重建会让版本回到 0，正好与删除前启动的任务撞上。
        generation = await conversation_repo.bump_generation(db, identity.user_id)
        messages = await conversation_repo.clear_user_messages(db, identity.user_id)
        profile_deleted = await profile_repo.purge(db, identity.user_id)
        memories = await memory_repo.purge(db, identity.user_id)
        # 与 clear 同理：合规删除不能只删表里的对话，去重缓存里的同一批内容也要删。
        await idempotency_repo.purge_user(db, siwe_user_id=identity.user_id)

    await container.audit.log(
        "data_purged",
        siwe_user_id=identity.user_id,
        messages=messages,
        profile=profile_deleted,
        memories=memories,
        generation=generation,
    )
    return PurgeResp(
        deleted_messages=messages,
        deleted_profile=profile_deleted,
        deleted_memories=memories,
        deleted_at=deleted_at,
    )

