"""工具结果直出：把结构化数据排成给人读的文本，不经模型改写。

列表类查询（交易记录）的结果条数多、字段密，塞进 8k 上下文会被截断，
模型也容易在复述时改错金额或哈希。这类结果直接成形返回，模型只负责取数。
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META


def _symbol(event: dict, chain_id: int) -> tuple[str, int]:
    symbol, decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)
    return event.get("tokenSymbol") or symbol, int(event.get("decimals") or decimals)


def _amount(value: str | None, decimals: int, symbol: str) -> str:
    try:
        raw = Decimal(str(value or "0")) / Decimal(10) ** decimals
    except (InvalidOperation, ValueError):
        return f"{value} {symbol}"
    # 链上金额有效位差异很大：0.000001 要留得住，1.0 不该显示成 1.000000。
    text = f"{raw:.6f}".rstrip("0").rstrip(".") or "0"
    return f"{text} {symbol}"


def _is_zero(value: object) -> bool:
    try:
        return Decimal(str(value or 0)) == 0
    except (InvalidOperation, ValueError):
        return False


def _short(address: str | None) -> str:
    if not address:
        return "—"
    return f"{address[:6]}…{address[-4:]}" if len(address) > 12 else address


def _time(ts: int | str | None) -> str:
    try:
        moment = datetime.fromtimestamp(int(ts or 0), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return "时间未知"
    return moment.strftime("%m-%d %H:%M")


def _direction(record: dict, wallet: str) -> str:
    owner = wallet.lower()
    if (record.get("from") or "").lower() == owner:
        return "转出"
    if (record.get("to") or "").lower() == owner:
        return "转入"
    return "关联"


# 单笔交易的资产变动展示上限。批量空投一笔可能有几十条，全排出来会挤爆上下文；
# 超出部分明确写「另有 N 项」，不做静默截断。
_MAX_EVENTS_SHOWN = 8


def _assets(record: dict, wallet: str) -> str:
    """一笔交易里的资产变动，例如「转出 100 USDC / 转入 0.02 WETH」。

    零值事件是合约调用本身，没有资产移动；只要这笔交易还有非零事件就不展示它们，
    否则一笔 swap 会因为「转出 0 ETH」占掉一格。
    """
    events = record.get("events") or []
    moved = [event for event in events if not _is_zero(event.get("value"))]
    if not moved:
        return "无资产变动" if events else "—"
    parts = []
    for event in moved[:_MAX_EVENTS_SHOWN]:
        symbol, decimals = _symbol(event, int(record.get("chainId") or 0))
        parts.append(
            f"{_direction(event, wallet)} {_amount(event.get('value'), decimals, symbol)}"
        )
    if len(moved) > _MAX_EVENTS_SHOWN:
        parts.append(f"另有 {len(moved) - _MAX_EVENTS_SHOWN} 项")
    return " / ".join(parts)


def render_transactions(payload: dict) -> str:
    """交易记录 → 列表文本。hash 保留全量，用户要拿去浏览器核对。"""
    records = payload.get("records") or []
    wallet = payload.get("address") or ""
    errors = payload.get("errors") or []
    partial = payload.get("status") == "partial" or bool(errors)
    if payload.get("error") or payload.get("status") == "failed":
        return payload.get("message") or "交易记录查询失败，请稍后重试。"
    warning = "部分交易记录数据源查询失败，本次结果不完整。"
    if errors:
        scopes = "、".join(f"链 {e.get('chainId')} / {e.get('scope')}" for e in errors)
        warning += f" 受影响范围：{scopes}。"
    if not records:
        if partial:
            return f"地址 {_short(wallet)} 在成功查询的范围内未查到交易记录。{warning}"
        return f"地址 {_short(wallet)} 在已配置的链上没有查到交易记录。"

    lines = [f"最近 {len(records)} 笔交易（{_short(wallet)}）", ""]
    for index, record in enumerate(records, 1):
        lines.append(
            f"{index}. {_time(record.get('timeStamp'))} UTC · "
            f"链 {record.get('chainId')} · {_assets(record, wallet)}"
        )
        lines.append(
            f"   {_short(record.get('from'))} → {_short(record.get('to'))} · {record.get('hash')}"
        )
    if partial:
        lines.extend(["", warning])
    return "\n".join(lines)
