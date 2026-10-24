"""T3 get_stake_info：质押仓位查询。

走 StakeAdapterRegistry，一个合约一个适配器。V1a 只落地"未命中就拦截"这一段：
合约不在白名单、该合约没有适配器，一律返回 E202，不做猜测性读取。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

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
            return {"error": "E202", "status": "failed", "message": "质押合约不在白名单内"}

        chain_id = int(entry["chain_id"])
        if chain_id not in adapter.available_chains():
            return {"error": "E102", "status": "failed", "message": "该合约所在链未配置 RPC"}

        # 适配器实现待接入（M2）。V1a 没有任何仓位读取能力，就必须报「不支持」：
        # 返回 ok=True 却没有 position，证据校验会以为查到了，模型只能自己编仓位数字。
        return {
            "error": "E202",
            "status": "failed",
            "userAddress": userAddress,
            "stakeContract": entry["address"],
            "chainId": chain_id,
            "message": "该质押合约暂未接入仓位查询",
        }

    return StructuredTool.from_function(
        coroutine=get_stake_info,
        name="get_stake_info",
        description=DESCRIPTION,
    )
