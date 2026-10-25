"""把构造工具产出的待签交易落成 workflow 任务。

单步转账与多步计划的收尾是同一件事：确认卡片上的字段必须和即将签名的交易出自同一份
payload，所以只在这里组装一次。
"""

from __future__ import annotations

from web3_agent.domain.workflow import service as workflow_service


async def create(
    payload: dict,
    *,
    siwe_user_id: str,
    wallet_address: str,
    chain_id: int,
    ttl_seconds: int,
    plan_thread_id: str | None = None,
    evidence: list[dict] | None = None,
) -> tuple[str, dict]:
    """建立等待确认的转账任务，返回任务标识与卡片摘要。

    Args:
        payload: `post_transfer` 返回的待签交易，`status` 必须为 ready。
        siwe_user_id: 数据所属用户的稳定标识。
        wallet_address: 服务端确认的当前钱包地址。
        chain_id: 交易所在网络编号。
        ttl_seconds: 任务有效期秒数。
        plan_thread_id: 发起本任务的多步计划标识；单步建出来的任务为空。
        evidence: 多步计划里条件成立的判断依据；单步转账为空。

    Raises:
        KeyError: payload 缺少待签交易的必要字段。
        Exception: 任务写库失败，由调用方决定如何回用户。
    """
    asset = payload["asset"]
    preview = {
        "action": "transfer",
        "tokenSymbol": asset["symbol"],
        "humanAmount": asset["human"],
        "target": payload["recipient"],
        "fromAddress": wallet_address,
        "chainId": chain_id,
        "evidence": evidence or [],
    }
    task_id = await workflow_service.create_transfer(
        siwe_user_id=siwe_user_id,
        siwe_address=wallet_address,
        chain_id=chain_id,
        to_address=payload["recipient"],
        token_symbol=asset["symbol"],
        human_amount=asset["human"],
        tx_intent={
            "from": payload["from"],
            "to": payload["tx"]["to"],
            "value": payload["tx"]["value"],
            "data": payload["tx"]["data"],
            "chainId": payload["chainId"],
        },
        ttl_seconds=ttl_seconds,
        plan_thread_id=plan_thread_id,
    )
    return task_id, preview
