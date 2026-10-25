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


def build_post_transfer(adapter: ChainAdapter) -> StructuredTool:
    """把链适配器绑定为转账交易构造工具。

    Args:
        adapter: 按网络管理 RPC 客户端与只读缓存的链适配器。
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
        """校验地址、金额与资产归属，拼出待签名交易参数。

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

        symbol = (token_symbol or "").strip()
        native_symbol, native_decimals = NATIVE_META.get(chain_id, NATIVE_FALLBACK)

        # 原生币没有合约地址，也不依赖代币注册表；白名单里查不到它，必须先判。
        if symbol.upper() == native_symbol.upper():
            try:
                value = _to_raw(human_amount, native_decimals)
            except ValueError as exc:
                return {"error": "E104", "status": "failed", "message": str(exc)}
            return {
                "status": "ready",
                "chainId": chain_id,
                "from": from_address,
                "tx": {"to": to_address, "value": str(value), "data": "0x"},
                "asset": {
                    "symbol": native_symbol,
                    "isNative": True,
                    "decimals": native_decimals,
                    "human": human_amount,
                },
                "recipient": to_address,
            }

        try:
            async with session_scope() as db:
                tokens = await token_registry_repo.list_whitelisted(db, chain_id)
        except Exception:  # noqa: BLE001 注册表不可用时不猜代币地址
            return {
                "error": "E301",
                "status": "failed",
                "message": "代币注册表不可用，无法确定代币合约地址",
            }

        token = next((item for item in tokens if item.symbol.upper() == symbol.upper()), None)
        if token is None:
            return {
                "error": "E202",
                "status": "failed",
                "message": f"{symbol} 不在该网络的白名单内",
            }

        try:
            raw = _to_raw(human_amount, token.decimals)
        except ValueError as exc:
            return {"error": "E104", "status": "failed", "message": str(exc)}

        return {
            "status": "ready",
            "chainId": chain_id,
            "from": from_address,
            "tx": {
                "to": token.token_address,
                "value": "0",
                "data": _encode_transfer(to_address, raw),
            },
            "asset": {
                "symbol": token.symbol,
                "isNative": False,
                "address": token.token_address,
                "decimals": token.decimals,
                "human": human_amount,
                "raw": str(raw),
            },
            "recipient": to_address,
        }

    return StructuredTool.from_function(
        coroutine=post_transfer,
        name="post_transfer",
        description=DESCRIPTION,
    )
