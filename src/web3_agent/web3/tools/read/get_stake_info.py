"""T3 get_stake_info：质押仓位查询。

走 StakeAdapterRegistry，一个合约一个适配器。V1a 只落地"未命中就拦截"这一段：
合约不在白名单、或该合约没有适配器，一律返回 E204，不做猜测性读取。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from web3_agent.common.exceptions import RpcUnavailable
from web3_agent.web3.chain_adapter import ChainAdapter, is_evm_address

DESCRIPTION = "查询用户在质押合约的质押信息；chainId后端会话注入；只读，不校验siweAddress"


def build_get_stake_info(adapter: ChainAdapter, stake_contracts: dict[str, dict]) -> StructuredTool:
    """stake_contracts: {alias 或 address(lower) -> {"chain_id", "address", "adapter"}}"""

    async def get_stake_info(userAddress: str, stakeContractAliasOrAddress: str) -> dict:
        if not is_evm_address(userAddress):
            return {"error": "E101", "message": "地址格式不合法"}

        key = stakeContractAliasOrAddress.lower()
        entry = stake_contracts.get(key) or stake_contracts.get(stakeContractAliasOrAddress)
        if entry is None:
            return {"error": "E204", "message": "质押合约不在白名单内"}

        chain_id = int(entry["chain_id"])
        if chain_id not in adapter.available_chains():
            return {"error": "E102", "message": "该合约所在链未配置 RPC"}

        try:
            block_number = await adapter.get_block_number(chain_id)
        except RpcUnavailable as exc:
            return {"error": "E301", "message": str(exc)}

        # 适配器实现待接入（M2）：此处只返回链高度，证明链路打通，不编造仓位数字。
        return {
            "userAddress": userAddress,
            "stakeContract": entry["address"],
            "chainId": chain_id,
            "observedAtBlock": block_number,
            "adapter": entry.get("adapter"),
            "position": None,
            "note": "适配器未接入，仓位数据未读取",
        }

    return StructuredTool.from_function(
        coroutine=get_stake_info,
        name="get_stake_info",
        description=DESCRIPTION,
    )
