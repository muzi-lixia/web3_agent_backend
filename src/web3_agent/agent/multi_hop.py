"""多步计划的结构校验与条件求值。

模型负责说清楚查什么、和什么比、做什么（`Hop`）；这里只做两件事：判断这份计划能
不能被执行器表达，以及按工具回报的原始整数求值。金额是否出自用户原话优先看模型给
的原话片段，片段缺失时才退回数字比对。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from web3_agent.common.types import Hop

# _ADDRESS_RE：整段剥掉地址，避免地址里的十六进制数字被当成金额。
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
# _NUMBER_RE：原话里的数字，兼容千分位写法。
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# _LEADING_NUMBER_RE：阈值写成「10 BERA」这类带单位形式时，取出开头的数字部分。
_LEADING_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# OP_TEXT：比较符的中文说法，向用户解释条件时使用。
OP_TEXT = {"gt": "大于", "gte": "不低于", "lt": "小于", "lte": "不高于", "eq": "等于"}

# _OPS：运算符到比较函数的映射。闭集，模型给出别的值一律按无效处理。
_OPS = {
    "gt": lambda actual, threshold: actual > threshold,
    "gte": lambda actual, threshold: actual >= threshold,
    "lt": lambda actual, threshold: actual < threshold,
    "lte": lambda actual, threshold: actual <= threshold,
    "eq": lambda actual, threshold: actual == threshold,
}


# 数据职责：一次前置比较的结论。无法判断时 holds 为 None，调用方必须追问。
@dataclass(frozen=True)
class Verdict:
    # holds：条件是否成立；无法判断时为 None。
    holds: bool | None
    # actual：按原始整数换算出的实际数量；取不到时为 None。
    actual: Decimal | None = None
    # reason：无法判断的原因；得出结论时为空串。
    reason: str = ""


def needs_plan(hops: list[Hop]) -> bool:
    """判断这份计划是否需要按步骤编排。

    纯取数等同一次普通查询，纯写操作等同现有的单步转账，两者都不值得挂起一个图。

    Args:
        hops: 模型给出的执行计划。
    """
    kinds = {hop.kind for hop in hops}
    return "write" in kinds and ("read" in kinds or "cond" in kinds)


def has_condition(hops: list[Hop]) -> bool:
    """判断计划里有没有前置条件。

    Args:
        hops: 模型给出的执行计划。
    """
    return any(hop.kind == "cond" for hop in hops)


def plan_shape_error(hops: list[Hop]) -> str:
    """检查计划能否被执行器表达，返回给用户的说明；可以执行时为空串。

    执行器一次只构造一笔交易：多笔写会被静默丢弃，写之后还有取数会被挪到前面。
    那两种都不是「尽力执行」，是执行成另一条指令，只能拒绝。

    Args:
        hops: 模型给出的执行计划。
    """
    if not needs_plan(hops):
        return ""
    writes = [index for index, hop in enumerate(hops) if hop.kind == "write"]
    if len(writes) != 1:
        return "这条指令里有多笔转账，我一次只能办一笔，先办哪一笔？"
    if writes[0] != len(hops) - 1:
        return "转账之后还有别的操作，我目前只能把一笔转账放在最后，请分开说。"
    if sum(1 for hop in hops if hop.kind == "cond") > 1:
        return "这条指令里有不止一个前提，我一次只能判断一个，先说满足哪个才转。"
    return ""


def numbers_in(message: str) -> list[Decimal]:
    """抽出原话里出现的数值，先把地址整段剥掉。

    Args:
        message: 用户本轮原始输入。
    """
    return [
        Decimal(item.replace(",", ""))
        for item in _NUMBER_RE.findall(_ADDRESS_RE.sub(" ", message))
    ]


def _as_decimal(value: str) -> Decimal | None:
    """把数值文本解析成十进制；解析不出返回 None。

    Args:
        value: 待解析的数值文本。
    """
    try:
        return Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _threshold(value: str) -> Decimal | None:
    """解析条件阈值：整段是数字就用它，带单位时取开头的数字部分。

    Args:
        value: 模型给出的阈值文本，可能是「10」「10 BERA」「1,000」。
    """
    exact = _as_decimal(value)
    if exact is not None:
        return exact
    text = str(value).strip()
    # 地址以 0 开头，取首位数字会把地址读成 0。
    if not text or text.lower().startswith("0x"):
        return None
    match = _LEADING_NUMBER_RE.match(text)
    return Decimal(match.group().replace(",", "")) if match else None


def same_number(left: str, right: str) -> bool:
    """判断两个数值文本是否相等。

    Args:
        left: 第一个数值文本。
        right: 第二个数值文本。
    """
    first, second = _as_decimal(left), _as_decimal(right)
    return first is not None and first == second


def number_quoted(value: str, message: str, *, quote: str = "") -> bool:
    """判断一个数值是否出现在用户本轮原话里。

    用来区分「这个门槛是这一轮说的」和「模型从上一轮带过来的」。写成「十」「一万」
    的说法抽不出阿拉伯数字，改用模型给出的原话片段。

    Args:
        value: 待核对的数值文本，允许带单位。
        message: 用户本轮原始输入。
        quote: 模型给出的、原话里表示该数值的片段。
    """
    wanted = _threshold(value)
    if wanted is None:
        return False
    if wanted in numbers_in(message):
        return True
    text = quote.strip()
    if not text or text not in message:
        return False
    # 片段是阿拉伯数字时它必须就是这个阈值；是「十」这种写法就无法换算，按原话采纳。
    spelled = _threshold(text)
    return spelled is None or spelled == wanted


def amount_is_quoted(
    human_amount: str,
    message: str,
    *,
    reserved: tuple[str, ...] = (),
    quote: str = "",
) -> bool:
    """判断待签名的金额是否出自用户原话。

    一句里出现两个数字时（「超过 10 就转 5」），条件阈值已经被条件步用掉，不能再拿来
    当转账金额；所以先把 `reserved` 里的数值逐个从原话里扣掉，剩下的才参与匹配。
    「五个」「一半」这类抽不出数字的说法，改用模型给出的原话片段。

    Args:
        human_amount: 模型抽取的转账数量。
        message: 用户本轮原始输入。
        reserved: 已被条件步占用的数值，每个消耗一次。
        quote: 模型给出的、原话里表示该金额的片段。
    """
    wanted = _as_decimal(human_amount)
    if wanted is None:
        return False
    spoken = numbers_in(message)
    for value in reserved:
        used = _threshold(value)
        if used is not None and used in spoken:
            spoken.remove(used)
    if wanted in spoken:
        return True

    text = quote.strip()
    if not text or text not in message:
        return False
    # 片段本身是个数字时，它不能正是被条件步占用的那个阈值。
    quoted = _as_decimal(text)
    if quoted is not None and any(quoted == _as_decimal(value) for value in reserved):
        return False
    return quoted is None or quoted == wanted


def human_amount(raw: str, decimals: int) -> Decimal:
    """把最小单位整数换算成人类可读数量，全程用十进制。

    Args:
        raw: 最小单位的整数字符串。
        decimals: 该资产的精度位数。
    """
    return Decimal(raw).scaleb(-decimals)


def format_amount(value: Decimal) -> str:
    """输出数量文本，去掉无意义的尾随零。

    Args:
        value: 待展示的数量。
    """
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _holding(payloads: list[dict], symbol: str, chain_id: int) -> Decimal | None:
    """从已完成的取数结果里捞出某个资产在指定网络上的数量。

    Args:
        payloads: 按执行顺序排列的取数步返回结果。
        symbol: 要找的资产符号，忽略大小写。
        chain_id: 会话所在网络；余额与交易必须在同一条链上比较。
    """
    wanted = symbol.strip().upper()
    for payload in payloads:
        for item in payload.get("balances") or ():
            if str(item.get("symbol", "")).upper() != wanted:
                continue
            if item.get("chainId") != chain_id:
                continue
            raw, decimals = item.get("raw"), item.get("decimals")
            if raw is None or decimals is None:
                continue
            # 用 raw 与精度重算，不用 human：展示值是截断过的，比较会失真。
            return human_amount(str(raw), int(decimals))
    return None


def evaluate(hop: Hop, payloads: list[dict], chain_id: int) -> Verdict:
    """求值一个条件步。

    Args:
        hop: 模型给出的条件步。
        payloads: 已完成的取数步返回结果，按执行顺序排列。
        chain_id: 会话所在网络，只比较这条链上的余额。
    """
    if hop.subject != "balance":
        return Verdict(holds=None, reason="暂不支持比较这类数据")
    if not hop.symbol.strip():
        return Verdict(holds=None, reason="条件没有说明比较哪个资产")
    compare = _OPS.get(hop.op or "")
    if compare is None:
        return Verdict(holds=None, reason="条件的比较方式无法识别")
    threshold = _threshold(hop.right)
    if threshold is None:
        return Verdict(holds=None, reason="条件的阈值不是有效数字")

    actual = _holding(payloads, hop.symbol, chain_id)
    if actual is None:
        return Verdict(holds=None, reason=f"没有查到 {hop.symbol.upper()} 的数量")
    return Verdict(holds=compare(actual, threshold), actual=actual)
