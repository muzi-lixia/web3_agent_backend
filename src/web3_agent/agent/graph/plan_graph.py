"""多步计划的执行图：取数 → 求值 → 建任务 → 挂起等人。

模型只给出计划（`hops`），执行按计划走：取数调只读工具，条件用工具回报的原始整数
求值，模型的口头结论不参与。走到写操作先建任务再 `interrupt()` 挂起；用户的选择由
确认接口落进 workflow 状态机，签名、广播、回执都不经过图。

节点拆成 prepare / gate 两段，是为了 resume 安全：`interrupt()` 所在的节点在恢复时
会从第一行重跑，副作用必须在它之前的节点里做完。
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from web3_agent.agent import multi_hop, transfer_task
from web3_agent.agent.graph.read_graph import ADDRESS_ARG
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.common.exceptions import error_message
from web3_agent.common.types import Hop
from web3_agent.config.constants import (
    MSG_AMOUNT_MISSING,
    MSG_CLARIFY,
    MSG_NO_DATA,
    MSG_WRITE_NOT_ENABLED,
    MSG_WRITE_TASK_FAILED,
    REQUIRED_SLOTS,
    TOOL_GET_ASSET_BALANCE,
    TOOL_POST_TRANSFER,
    CallSource,
    Intent,
)

logger = logging.getLogger(__name__)

# _READ_TOOLS：计划里允许出现的取数工具。写工具不在其中，它只能出现在 write 步。
_READ_TOOLS = frozenset(ADDRESS_ARG)


class PlanState(TypedDict, total=False):
    # hops：模型给出的执行计划，按顺序执行。
    hops: list[dict[str, Any]]
    # user_message：用户本轮原话，用于核对金额是否出自本人之口。
    user_message: str
    # wallet_address / siwe_user_id：会话身份，注入工具参数并作为任务归属。
    wallet_address: str
    siwe_user_id: str
    # chain_id：交易所在网络编号。
    chain_id: int
    # thread_id：本计划的检查点标识，随任务落库，供确认接口恢复。
    thread_id: str
    # ttl_seconds：建出的任务有效期秒数。
    ttl_seconds: int
    # payloads：取数步的返回结果，按执行顺序排列。
    payloads: list[dict[str, Any]]
    # tool_calls：取数步的执行轨迹。
    tool_calls: list[dict[str, Any]]
    # verdicts：条件步的求值结论，含实际数量与阈值。
    verdicts: list[dict[str, Any]]
    # task_id / preview：建出的任务标识与待确认参数快照。
    task_id: str
    preview: dict[str, Any]
    # kind / content：本轮的出口形态与文案；kind 非空表示不再往下执行。
    kind: str
    content: str


def _notice(code: str | None) -> str:
    """把工具失败错误码换成用户能照着做的文案。

    Args:
        code: 工具执行返回的错误码。
    """
    if not code:
        return MSG_NO_DATA
    try:
        return error_message(code)
    except KeyError:
        return MSG_NO_DATA


def _with_condition_reads(hops: list[dict], conds: list[dict]) -> list[dict]:
    """条件要比较的量必须先取回来，模型没给取数步时补一步。

    用户可以说「如果 BERA 超过 10 就转 5 给 0x…」而不提「查一下」；这时计划里只有
    cond 和 write，没有可比较的数。补一步读数而不是回一句「没查到」，才不会把一条
    完整指令变成死胡同。

    Args:
        hops: 模型给出的执行计划。
        conds: 计划里的条件步。
    """
    fetched = {
        str((hop.get("args") or {}).get("symbol") or "").upper()
        for hop in hops
        if hop.get("kind") == "read" and hop.get("tool") == TOOL_GET_ASSET_BALANCE
    }
    added: list[dict] = []
    for hop in conds:
        symbol = str(hop.get("symbol") or "").strip()
        if not symbol or symbol.upper() in fetched:
            continue
        # 模型给了读数步却没写资产：补上资产名，而不是再查一次。
        blank = next(
            (
                item
                for item in hops
                if item.get("kind") == "read"
                and item.get("tool") == TOOL_GET_ASSET_BALANCE
                and not str((item.get("args") or {}).get("symbol") or "").strip()
            ),
            None,
        )
        if blank is not None:
            blank.setdefault("args", {})["symbol"] = symbol
            fetched.add(symbol.upper())
            continue
        added.append(
            {
                "kind": "read",
                "tool": TOOL_GET_ASSET_BALANCE,
                "args": {"symbol": symbol},
            }
        )
        fetched.add(symbol.upper())
    if not added:
        return hops
    # 插在第一个条件步之前，保持「先取数、再比较、最后写」的顺序。
    index = next(
        (i for i, hop in enumerate(hops) if hop.get("kind") == "cond"), len(hops)
    )
    return [*hops[:index], *added, *hops[index:]]


def build_plan_graph(runner: ToolRunner, checkpointer: Any):
    """构造多步计划图。

    Args:
        runner: 工具统一执行器，负责来源检查和调用轨迹。
        checkpointer: 检查点存储，`interrupt` 依赖它保存挂起位置。
    """

    async def collect(state: PlanState) -> dict:
        """执行计划里的取数步，收集可供条件比较的原始结果。

        Args:
            state: 当前计划图状态。
        """
        # 门槛必须在这一轮说过，而且要和金额分得开：模型会把上一轮的条件带进来，
        # 也会把「转 5 BERA」里的 5 复制成门槛。两种都会让代码拿错条件去动钱。
        hops = list(state.get("hops", []))
        shape = multi_hop.plan_shape_error([Hop(**hop) for hop in hops])
        if shape:
            return {"kind": "ask", "content": shape}
        write = next((hop for hop in hops if hop.get("kind") == "write"), None)
        args = dict((write or {}).get("args") or {})
        amount = str(args.get("humanAmount") or "")
        conds = [hop for hop in hops if hop.get("kind") == "cond"]
        # 门槛只要求原话里出现过，同一个阈值被两条条件引用不算「没说过」。
        for hop in conds:
            if not multi_hop.number_quoted(
                str(hop.get("right") or ""),
                state["user_message"],
                quote=str(hop.get("rightText") or ""),
            ):
                return {
                    "kind": "ask",
                    "content": f"这一轮没有提到「{hop.get('right')}」这个门槛：是沿用上一轮的条件，还是按新指令处理？",
                }
        reserved = tuple(dict.fromkeys(str(hop.get("right") or "") for hop in conds))
        quoted = multi_hop.amount_is_quoted(
            amount,
            state["user_message"],
            reserved=reserved,
            quote=str(args.get("humanAmountText") or ""),
        )
        if amount and not quoted:
            only = len(multi_hop.numbers_in(state["user_message"])) == 1
            if only and conds and multi_hop.same_number(reserved[0], amount):
                # 原话里只有这一个数，模型却把它同时当成门槛和金额：这句话是单步指令。
                logger.info(
                    "plan drop phantom condition thread=%s right=%s",
                    state.get("thread_id"),
                    amount,
                )
                hops = [hop for hop in hops if hop.get("kind") != "cond"]
            else:
                return {"kind": "ask", "content": MSG_AMOUNT_MISSING}

        hops = _with_condition_reads(
            hops, [hop for hop in hops if hop.get("kind") == "cond"]
        )
        payloads: list[dict] = []
        records: list[dict] = []
        for hop in hops:
            if hop.get("kind") != "read":
                continue
            name = str(hop.get("tool") or "")
            if name not in _READ_TOOLS:
                logger.warning("plan read rejected tool=%s", name)
                return {"kind": "answer", "content": MSG_NO_DATA, "tool_calls": records}
            call_args = dict(hop.get("args") or {})
            # 地址一律用会话身份覆盖：条件比较的就是「我」的余额，模型给的地址不采纳。
            call_args[ADDRESS_ARG[name]] = state["wallet_address"]
            text, record = await runner.run(
                name, call_args, source=CallSource.STATE_MACHINE
            )
            records.append(record.model_dump())
            if not record.ok:
                return {
                    "kind": "answer",
                    "content": _notice(record.error),
                    "tool_calls": records,
                }
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return {"kind": "answer", "content": MSG_NO_DATA, "tool_calls": records}
            payloads.append(payload)
        return {"hops": hops, "payloads": payloads, "tool_calls": records}

    async def judge(state: PlanState) -> dict:
        """按取数结果求值条件步；结论不成立就结束，不建任务。

        Args:
            state: 当前计划图状态。
        """
        if state.get("kind"):
            return {}
        payloads = state.get("payloads", [])
        verdicts: list[dict] = []
        for hop in state.get("hops", []):
            if hop.get("kind") != "cond":
                continue
            verdict = multi_hop.evaluate(Hop(**hop), payloads, state["chain_id"])
            if verdict.holds is None:
                # 算不出结论时只能追问：默认成立等于替用户扣钱。
                return {"kind": "ask", "content": verdict.reason, "verdicts": verdicts}
            symbol = str(hop.get("symbol", "")).upper()
            evidence = {
                "symbol": symbol,
                "actual": multi_hop.format_amount(verdict.actual or Decimal("0")),
                "op": hop.get("op") or "",
                "right": str(hop.get("right") or ""),
            }
            verdicts.append(evidence)
            if not verdict.holds:
                return {
                    "kind": "answer",
                    "content": (
                        f"你持有 {evidence['actual']} {symbol}，"
                        f"没有达到「{multi_hop.OP_TEXT.get(evidence['op'], evidence['op'])} "
                        f"{evidence['right']} {symbol}」的条件，没有发起转账。"
                    ),
                    "verdicts": verdicts,
                }
        return {"verdicts": verdicts}

    async def afford(state: PlanState) -> dict:
        """条件成立也要看余额够不够：不够的转账只会在链上回滚。

        只在同一资产、同一条链的余额已经查过时拦截。查不到就放行 —— 构造工具只校验
        地址与精度，不查余额，这一层不拦就没有别的地方会拦。

        Args:
            state: 当前计划图状态。
        """
        if state.get("kind"):
            return {}
        write = next(
            (hop for hop in state.get("hops", []) if hop.get("kind") == "write"), None
        )
        args = dict((write or {}).get("args") or {})
        symbol = str(args.get("tokenSymbol") or "")
        amount = str(args.get("humanAmount") or "")
        if not symbol or not amount:
            return {}
        verdict = multi_hop.evaluate(
            Hop(kind="cond", symbol=symbol, op="gte", right=amount),
            state.get("payloads", []),
            state["chain_id"],
        )
        if verdict.holds is not False:
            return {}
        return {
            "kind": "answer",
            "content": (
                f"余额不足：你持有 {multi_hop.format_amount(verdict.actual or Decimal('0'))} "
                f"{symbol.upper()}，这笔要转 {amount} {symbol.upper()}。"
            ),
        }

    async def prepare(state: PlanState) -> dict:
        """组装待签交易并建立等待确认的任务。

        有副作用，所以必须和 `interrupt` 分在两个节点：恢复时 `gate` 会重跑，
        这个节点不会。

        Args:
            state: 当前计划图状态。
        """
        if state.get("kind"):
            return {}
        write = next(
            (hop for hop in state.get("hops", []) if hop.get("kind") == "write"), None
        )
        if write is None or write.get("tool") != TOOL_POST_TRANSFER:
            return {"kind": "answer", "content": MSG_WRITE_NOT_ENABLED}
        args = dict(write.get("args") or {})
        if any(
            not str(args.get(key, "")).strip()
            for key in REQUIRED_SLOTS[Intent.TRANSFER]
        ):
            return {"kind": "ask", "content": MSG_CLARIFY}
        amount = str(args["humanAmount"])
        text, record = await runner.run(
            TOOL_POST_TRANSFER,
            {
                "from_address": state["wallet_address"],
                "to_address": str(args["toAddress"]),
                "token_symbol": str(args["tokenSymbol"]),
                "human_amount": amount,
                "chain_id": state["chain_id"],
            },
            source=CallSource.STATE_MACHINE,
        )
        calls = [*state.get("tool_calls", []), record.model_dump()]
        try:
            payload = json.loads(text)
            if payload.get("status") != "ready":
                return {
                    "kind": "ask",
                    "content": str(payload.get("message") or _notice(record.error)),
                    "tool_calls": calls,
                }
            task_id, preview = await transfer_task.create(
                payload,
                siwe_user_id=state["siwe_user_id"],
                wallet_address=state["wallet_address"],
                chain_id=state["chain_id"],
                ttl_seconds=state["ttl_seconds"],
                plan_thread_id=state.get("thread_id") or None,
                evidence=state.get("verdicts") or [],
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # 工具返回的不是可用的待签交易：当作没有数据，不能让异常穿到外层。
            logger.warning(
                "plan prepare got unusable payload thread=%s", state.get("thread_id")
            )
            return {"kind": "answer", "content": MSG_NO_DATA, "tool_calls": calls}
        except Exception:  # noqa: BLE001 建不出任务就没有可确认的对象
            logger.exception("plan create task failed thread=%s", state.get("thread_id"))
            return {
                "kind": "answer",
                "content": MSG_WRITE_TASK_FAILED,
                "tool_calls": calls,
            }
        return {"task_id": task_id, "preview": preview, "tool_calls": calls}

    async def gate(state: PlanState) -> dict:
        """挂起图等用户决定。

        节点里只有 `interrupt()`，没有任何副作用：恢复时它会被整体重跑，
        把建任务放在这里就会一笔转账建两次。用户的选择由确认接口落进状态机，
        图只负责从挂起处走到终点。

        Args:
            state: 当前计划图状态。
        """
        if state.get("kind"):
            return {}
        interrupt({"taskId": state["task_id"]})
        return {}

    def after_judge(state: PlanState) -> str:
        """条件不成立或算不出来时直接结束，否则去看余额够不够。

        Args:
            state: 当前计划图状态。
        """
        return END if state.get("kind") else "afford"

    def after_afford(state: PlanState) -> str:
        """余额不够就到此为止，够才去建任务。

        Args:
            state: 当前计划图状态。
        """
        return END if state.get("kind") else "prepare"

    def after_prepare(state: PlanState) -> str:
        """建出任务才挂起，其余情况按已给出的结论结束。

        Args:
            state: 当前计划图状态。
        """
        return END if state.get("kind") else "gate"

    builder = StateGraph(PlanState)
    builder.add_node("collect", collect)
    builder.add_node("judge", judge)
    builder.add_node("afford", afford)
    builder.add_node("prepare", prepare)
    builder.add_node("gate", gate)
    builder.set_entry_point("collect")
    builder.add_edge("collect", "judge")
    builder.add_conditional_edges("judge", after_judge, {"afford": "afford", END: END})
    builder.add_conditional_edges(
        "afford", after_afford, {"prepare": "prepare", END: END}
    )
    builder.add_conditional_edges(
        "prepare", after_prepare, {"gate": "gate", END: END}
    )
    builder.add_edge("gate", END)
    return builder.compile(checkpointer=checkpointer)


def initial(
    *,
    hops: list[Hop],
    user_message: str,
    siwe_user_id: str,
    wallet_address: str,
    chain_id: int,
    thread_id: str,
    ttl_seconds: int,
) -> PlanState:
    """组装计划图的初始状态。

    Args:
        hops: 模型给出的执行计划。
        user_message: 用户本轮原始输入。
        siwe_user_id: 数据所属用户的稳定标识。
        wallet_address: 服务端确认的当前钱包地址。
        chain_id: 交易所在网络编号。
        thread_id: 本计划的检查点标识。
        ttl_seconds: 建出的任务有效期秒数。
    """
    return {
        "hops": [hop.model_dump() for hop in hops],
        "user_message": user_message,
        "siwe_user_id": siwe_user_id,
        "wallet_address": wallet_address,
        "chain_id": chain_id,
        "thread_id": thread_id,
        "ttl_seconds": ttl_seconds,
        "payloads": [],
        "tool_calls": [],
        "verdicts": [],
        "task_id": "",
        "preview": {},
        "kind": "",
        "content": "",
    }


def opening(verdicts: list[dict], content: str) -> str:
    """在确认文案前补一句判断依据，用户签字前能看见为什么发起。

    Args:
        verdicts: 条件步的求值结论。
        content: 确认文案正文。
    """
    if not verdicts:
        return content
    lines = [
        f"你持有 {item['actual']} {item['symbol']}，"
        f"满足「{multi_hop.OP_TEXT.get(item['op'], item['op'])} {item['right']} {item['symbol']}」。"
        for item in verdicts
    ]
    return f"{''.join(lines)}{content}"


__all__ = [
    "PlanState",
    "build_plan_graph",
    "initial",
    "opening",
]
