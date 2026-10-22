"""工具结果直出：把结构化数据排成给人读的文本，不经模型改写。

列表类查询（交易记录）的结果条数多、字段密，塞进 8k 上下文会被截断，
模型也容易在复述时改错金额或哈希。这类结果直接成形返回，模型只负责取数。
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META


def _symbol(record: dict) -> tuple[str, int]:
    chain_id = int(record.get("chainId") or 0)
    symbol, decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)
    return record.get("tokenSymbol") or symbol, int(record.get("decimals") or decimals)


def _amount(value: str | None, decimals: int, symbol: str) -> str:
    try:
        raw = Decimal(str(value or "0")) / Decimal(10) ** decimals
    except (InvalidOperation, ValueError):
        return f"{value} {symbol}"
    # 链上金额有效位差异很大：0.000001 要留得住，1.0 不该显示成 1.000000。
    text = f"{raw:.6f}".rstrip("0").rstrip(".") or "0"
    return f"{text} {symbol}"


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


def render_transactions(payload: dict) -> str:
    """交易记录 → 列表文本。hash 保留全量，用户要拿去浏览器核对。"""
    records = payload.get("records") or []
    wallet = payload.get("address") or ""
    if not records:
        return f"地址 {_short(wallet)} 在已配置的链上没有查到交易记录。"

    lines = [f"最近 {len(records)} 笔交易（{_short(wallet)}）", ""]
    for index, record in enumerate(records, 1):
        symbol, decimals = _symbol(record)
        lines.append(
            f"{index}. {_time(record.get('timeStamp'))} UTC · "
            f"{_direction(record, wallet)} {_amount(record.get('value'), decimals, symbol)} · "
            f"链 {record.get('chainId')}"
        )
        lines.append(
            f"   {_short(record.get('from'))} → {_short(record.get('to'))} · {record.get('hash')}"
        )
    return "\n".join(lines)
