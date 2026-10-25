"""T4 post_transfer：构造转账交易，不签名、不广播。

服务端不持有私钥，签名由钱包完成，所以这里只产出待签名的交易参数。
本工具属于写路径，只允许状态机调用（ToolGroup.STATEMACHINE_ONLY），
不进入模型可绑定的工具集合。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from langchain_core.tools import StructuredTool

from web3_agent.config.constants import NATIVE_FALLBACK, NATIVE_META
from web3_agent.persistence.pg.base import session_scope
from web3_agent.persistence.pg.repositories import token_registry_repo
from web3_agent.web3.chain_adapter import ChainAdapter, is_evm_address

# DESCRIPTION：暴露给调用方的能力说明；签名与广播都不在服务端完成。
DESCRIPTION = (
    "构造转账交易：原生币或白名单 ERC20，返回待签名的 to/value/data；"
    "chainId与付款地址由后端会话注入，服务端不签名、不广播"
)

# _TRANSFER_SELECTOR：ERC20 transfer(address,uint256) 的函数选择器。
_TRANSFER_SELECTOR = "a9059cbb"


def _encode_transfer(recipient: str, amount: int) -> str:
    """按 ABI 拼接 transfer 调用数据：选择器 + 32 字节地址 + 32 字节金额。

    Args:
        recipient: 收款地址，带 0x 前缀。
        amount: 代币最小单位的整数金额。
    """
    address_word = recipient.lower().removeprefix("0x").rjust(64, "0")
    amount_word = format(amount, "x").rjust(64, "0")
    return f"0x{_TRANSFER_SELECTOR}{address_word}{amount_word}"


def _to_raw(human_amount: str, decimals: int) -> int:
    """把人类可读金额换算成最小单位整数，精度超出代币精度时拒绝。

    Args:
        human_amount: 用户确认过的金额文本。
        decimals: 代币或原生币的小数精度。

    Raises:
        ValueError: 金额不是数字、不为正，或小数位超过该资产精度。
    """
    try:
        value = Decimal(human_amount.strip().replace(",", ""))
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError("金额格式不合法") from exc
    if value <= 0:
        raise ValueError("金额必须大于 0")
    scaled = value.scaleb(decimals)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"金额小数位超过该资产精度（{decimals} 位）")
    return int(scaled)


def _short(raw: int, decimals: int) -> str:
    """把最小单位数量写成人类可读文本，去掉无意义的尾随零。

    Args:
        raw: 最小单位的整数数量。
        decimals: 该资产的精度位数。
    """
    text = format(Decimal(raw).scaleb(-decimals).normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _over_limit(
    *, symbol: str, human: str, limit_usd: float, prices: dict[str, float]
) -> dict | None:
    """单笔金额是否超过美元等值上限；折算不出来时同样拦下。

    限额是资金侧最后一道闸，宁可不出卡片也不能按「没有单价就不算超额」放行。
    单价来自配置（write_usd_prices），不是实时行情，只用于这道粗粒度的风控。

    Args:
        symbol: 转账资产符号。
        human: 人类可读的转账数量。
        limit_usd: 单笔美元等值上限，0 表示不限制。
        prices: 资产符号到单价的映射。
    """
    if limit_usd <= 0:
        return None
    price = next(
        (value for key, value in prices.items() if key.upper() == symbol.upper()), None
    )
    if price is None:
        return {
            "error": "E502",
            "status": "failed",
            "message": f"没有配置 {symbol} 的单价，无法校验单笔限额，这笔转账被拦下",
        }
    amount_usd = Decimal(human) * Decimal(str(price))
    if amount_usd <= Decimal(str(limit_usd)):
        return None
    return {
        "error": "E502",
        "status": "failed",
        "message": (
            f"单笔金额超出限额：{human} {symbol} 约 {amount_usd:.2f} 美元，"
            f"上限 {limit_usd:g} 美元"
        ),
    }


# _SIMULATION_TEXT：预演失败原因对应的用户文案。
_SIMULATION_TEXT = {
    "reverted": "预演时这笔交易被合约拒绝，换一个收款地址或资产再试",
    "insufficient_funds": "余额不足以支付转账金额与手续费",
}
async def _insufficient(
    adapter: ChainAdapter,
    *,
    chain_id: int,
    holder: str,
    need: int,
    decimals: int,
    symbol: str,
    token: str | None,
) -> dict | None:
    """余额不够时返回失败响应，够或查不到都返回 None。

    这里是两条写路径共用的最后一关：拼不出可执行的交易就不出卡片，
    免得用户签名后因余额不足在链上回滚、白烧 gas。查不到余额不拦，
    签名后的链上结论才是最终的。

    Args:
        adapter: 按网络管理 RPC 客户端与只读缓存的链适配器。
        chain_id: 交易所在网络编号。
        holder: 付款地址。
        need: 本次要转出的最小单位整数。
        decimals: 该资产精度。
        symbol: 资产符号，用于文案。
        token: ERC20 合约地址；原生币为空。
    """
    try:
        held = (
            await adapter.get_native_balance(chain_id, holder)
            if token is None
            else await adapter.get_erc20_balance(chain_id, token, holder)
        )
    except Exception:  # noqa: BLE001 余额查不到不阻断构造，链上会给结论
        return None
    if held >= need:
        return None
    return {
        "error": "E405",
        "status": "failed",
        "message": (
            f"余额不足：当前 {_short(held, decimals)} {symbol}，"
            f"需要 {_short(need, decimals)} {symbol}"
        ),
    }


def build_post_transfer(
    adapter: ChainAdapter,
    *,
    max_amount_usd: float,
    usd_prices: dict[str, float],
) -> StructuredTool:
    """把链适配器绑定为转账交易构造工具。

    Args:
        adapter: 按网络管理 RPC 客户端与只读缓存的链适配器。
        max_amount_usd: 单笔美元等值上限，0 表示不限制。
        usd_prices: 资产符号到单价的映射，用于折算美元等值。
    """

    async def post_transfer(
        from_address: str,
        to_address: str,
        token_symbol: str,
        human_amount: str,
        chain_id: int,
    ) -> dict:
        # 返回字段 status 为 ready 表示可以交给钱包签名，failed 表示参数不可执行。
        # tx 是待签名交易：to 为收款方（原生币）或代币合约（ERC20），value 为 wei 十进制字符串，
        # data 为 ERC20 调用数据、原生币为空。
        """校验地址、金额、资产归属、限额、余额与可执行性，拼出待签名交易参数。

        Args:
            from_address: 付款钱包地址，由会话身份注入。
            to_address: 收款地址。
            token_symbol: 原生币或白名单代币的符号。
            human_amount: 人类可读的转账数量文本。
            chain_id: 交易所在网络编号，由会话注入。
        """
        if not is_evm_address(from_address):
            return {"error": "E101", "status": "failed", "message": "付款地址格式不合法"}
        if not is_evm_address(to_address):
            return {"error": "E101", "status": "failed", "message": "收款地址格式不合法"}
        if chain_id not in adapter.available_chains():
            return {"error": "E102", "status": "failed", "message": "该网络未配置 RPC"}

        wanted = (token_symbol or "").strip()
        native_symbol, native_decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)

        # 原生币没有合约地址，也不依赖代币注册表；白名单里查不到它，必须先判。
        token = None
        if wanted.upper() != native_symbol.upper():
            try:
                async with session_scope() as db:
                    tokens = await token_registry_repo.list_whitelisted(db, chain_id)
            except Exception:  # noqa: BLE001 注册表不可用时不猜代币地址
                return {
                    "error": "E301",
                    "status": "failed",
                    "message": "代币注册表不可用，无法确定代币合约地址",
                }
            token = next(
                (item for item in tokens if item.symbol.upper() == wanted.upper()), None
            )
            if token is None:
                return {
                    "error": "E202",
                    "status": "failed",
                    "message": f"{wanted} 不在该网络的白名单内",
                }

        symbol = native_symbol if token is None else token.symbol
        decimals = native_decimals if token is None else token.decimals
        try:
            raw = _to_raw(human_amount, decimals)
        except ValueError as exc:
            return {"error": "E104", "status": "failed", "message": str(exc)}

        over = _over_limit(
            symbol=symbol,
            human=human_amount,
            limit_usd=max_amount_usd,
            prices=usd_prices,
        )
        if over is not None:
            return over

        short = await _insufficient(
            adapter,
            chain_id=chain_id,
            holder=from_address,
            need=raw,
            decimals=decimals,
            symbol=symbol,
            token=None if token is None else token.token_address,
        )
        if short is not None:
            return short

        tx = (
            {"to": to_address, "value": str(raw), "data": "0x"}
            if token is None
            else {
                "to": token.token_address,
                "value": "0",
                "data": _encode_transfer(to_address, raw),
            }
        )
        # 余额够也不代表能转：黑名单、税率、授权不足、手续费不够都只在执行时才暴露。
        # 预演一次能把这些挡在出卡片之前，而不是让用户签名后白烧 gas。
        rejected = await adapter.simulate(chain_id, {"from": from_address, **tx})
        if rejected is not None:
            return {
                "error": "E401",
                "status": "failed",
                "message": _SIMULATION_TEXT.get(rejected, "预演未通过，这笔交易无法执行"),
            }

        asset = {
            "symbol": symbol,
            "isNative": token is None,
            "decimals": decimals,
            "human": human_amount,
        }
        if token is not None:
            asset["address"] = token.token_address
            asset["raw"] = str(raw)
        return {
            "status": "ready",
            "chainId": chain_id,
            "from": from_address,
            "tx": tx,
            "asset": asset,
            "recipient": to_address,
        }

    return StructuredTool.from_function(
        coroutine=post_transfer,
        name="post_transfer",
        description=DESCRIPTION,
    )
