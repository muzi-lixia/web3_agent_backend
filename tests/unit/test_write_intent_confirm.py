"""写意图出口：参数齐全回显待确认视图，参数不合法追问补参数。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from web3_agent.agent.agent_loop import _amount_quoted
from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.api.schemas.chat_schema import HistoryItem
from web3_agent.common.exceptions import ToolSourceViolation
from web3_agent.common.types import ChatOutcome, LlmOutput
from web3_agent.config.constants import (
    MSG_AMOUNT_MISSING,
    MSG_CHAIN_REQUIRED,
    MSG_WALLET_REQUIRED,
    MSG_WRITE_NOT_ENABLED,
    MSG_WRITE_TASK_FAILED,
    CallSource,
    Intent,
    ToolGroup,
)
from web3_agent.domain.workflow import service as workflow_service
from web3_agent.web3.tools.write.transfer import build_post_transfer
from tests.unit.test_agent_loop_evidence import _Graph, _Llm, _loop, _without_db
from tests.unit.test_chat_idempotency_route import _container, _run

# _CHAIN：测试用网络，原生币为 BERA，转账不依赖代币注册表。
_CHAIN = 80094
# _FROM：登录身份给出的付款钱包。
_FROM = "0x" + "1" * 40
# _TO：收款地址。
_TO = "0x" + "2" * 40
# _TASK：替身建任务返回的标识，用于验证 taskId 有没有传到响应。
_TASK = "task-0000-1111"
# _CREATED：替身建任务收到的参数，供断言交易与归属字段。
_CREATED: dict = {}


async def _fake_create(**kwargs) -> str:
    """替掉真实建任务：只在单测里回答一个固定 taskId，并把入参留档。

    Args:
        **kwargs: 建任务所需的归属、参数与交易意图。
    """
    _CREATED.clear()
    _CREATED.update(kwargs)
    return _TASK


class _Adapter:
    """只回答支持哪些网络与原生币余额，其余能力转账工具用不到。"""

    def __init__(self, balance: int = 10**18) -> None:
        self.balance = balance

    def available_chains(self) -> list[int]:
        return [_CHAIN]

    async def get_native_balance(self, chain_id: int, address: str) -> int:
        """返回替身余额，默认远大于用例里的 0.1 BERA。

        Args:
            chain_id: 交易所在网络编号。
            address: 待查询的付款地址。
        """
        return self.balance


class _SlotLlm(_Llm):
    """意图固定，槽位由用例给定。"""

    def __init__(self, intent: Intent, slots: dict) -> None:
        super().__init__(intent)
        self.slots = slots

    async def extract_intent(self, history, user_message, **kwargs) -> LlmOutput:
        return LlmOutput(intent=self.intent, confidence=0.9, slots=self.slots)


def _loop_with_transfer(monkeypatch, intent: Intent, slots: dict, *, balance: int = 10**18):
    loop = _loop(_Graph("unused"), intent)
    loop._llm = _SlotLlm(intent, slots)
    registry = ToolRegistry()
    registry.register(
        build_post_transfer(_Adapter(balance)), ToolGroup.STATEMACHINE_ONLY, mutating=True
    )
    loop._runner = ToolRunner(registry)
    monkeypatch.setattr(workflow_service, "create_transfer", _fake_create)
    return _without_db(loop, monkeypatch)


def _handle(
    loop,
    *,
    wallet: str | None = _FROM,
    chain: int | None = _CHAIN,
    message: str = "把 0.1 BERA 转给 0x2222",
):
    return asyncio.run(
        loop.handle(
            siwe_user_id="user-1",
            wallet_address=wallet,
            chain_id=chain,
            user_message=message,
        )
    )


def _native_slots(**overrides) -> dict:
    slots = {"toAddress": _TO, "tokenSymbol": "BERA", "humanAmount": "0.1"}
    slots.update(overrides)
    return slots


def test_complete_transfer_returns_confirm_preview(monkeypatch):
    loop = _loop_with_transfer(monkeypatch, Intent.TRANSFER, _native_slots())
    outcome = _handle(loop)

    assert outcome.kind == "intent_confirm"
    assert outcome.intent_preview == {
        "action": "transfer",
        "tokenSymbol": "BERA",
        "humanAmount": "0.1",
        "target": _TO,
    }
    assert "0.1 BERA" in outcome.content and _TO in outcome.content
    # 参数校验走的是状态机来源，轨迹要能说明「这一步真的跑过工具」。
    assert [call.name for call in outcome.tool_calls] == ["post_transfer"]
    assert outcome.tool_calls[0].ok is True
    # 卡片必须带 taskId，否则前端拿不到可以提交确认的对象。
    assert outcome.task_id == _TASK
    assert _CREATED["siwe_address"] == _FROM
    assert _CREATED["to_address"] == _TO
    assert _CREATED["tx_intent"]["value"] == str(10**17)
    assert _CREATED["tx_intent"]["chainId"] == _CHAIN


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"toAddress": "0x12"}, "收款地址"),
        ({"humanAmount": "一点"}, "金额"),
    ],
)
def test_unusable_parameters_ask_instead_of_confirm(monkeypatch, overrides, expected):
    loop = _loop_with_transfer(monkeypatch, Intent.TRANSFER, _native_slots(**overrides))
    outcome = _handle(loop)

    assert outcome.kind == "ask"
    assert expected in outcome.content
    assert outcome.intent_preview is None


def test_insufficient_balance_asks_instead_of_confirm(monkeypatch):
    """余额不够就不出卡片：签名后链上必然回滚，gas 白烧。"""
    loop = _loop_with_transfer(
        monkeypatch, Intent.TRANSFER, _native_slots(), balance=10**16
    )
    outcome = _handle(loop)

    assert outcome.kind == "ask"
    assert "余额不足" in outcome.content
    assert outcome.intent_preview is None
    assert outcome.task_id is None


@pytest.mark.parametrize(
    ("message", "amount", "quoted"),
    [
        ("把 0.1 BERA 转给朋友", "0.1", True),
        ("转 0.10 个 BERA", "0.1", True),
        ("转 1,000 USDC 给朋友", "1000", True),
        # 「转点」「一半」都没给出具体数字，模型却会替用户补一个。
        ("转点 BERA 给我朋友", "0.5", False),
        ("把我一半的 BERA 转给朋友", "0.05", False),
        # 地址里的十六进制数字不是金额。
        (f"转 0.1 BERA 给 {_TO}", "22", False),
    ],
)
def test_amount_must_be_quoted_by_user(message, amount, quoted):
    assert _amount_quoted(amount, message) is quoted


def test_invented_amount_asks_and_creates_no_task(monkeypatch):
    _CREATED.clear()
    loop = _loop_with_transfer(
        monkeypatch, Intent.TRANSFER, _native_slots(humanAmount="0.5")
    )
    outcome = _handle(loop, message="转点 BERA 给朋友")

    assert outcome.kind == "ask"
    assert outcome.content == MSG_AMOUNT_MISSING
    assert outcome.intent_preview is None
    assert outcome.task_id is None
    assert outcome.tool_calls == []
    assert _CREATED == {}


def test_thousands_separator_still_confirms(monkeypatch):
    loop = _loop_with_transfer(
        monkeypatch, Intent.TRANSFER, _native_slots(humanAmount="1000"), balance=10**21
    )
    outcome = _handle(loop, message="转 1,000 BERA 给朋友")

    assert outcome.kind == "intent_confirm"


def test_unconnected_wallet_asks_before_building_transaction(monkeypatch):
    loop = _loop_with_transfer(monkeypatch, Intent.TRANSFER, _native_slots())
    outcome = _handle(loop, wallet=None)

    assert outcome.kind == "ask"
    assert outcome.content == MSG_WALLET_REQUIRED
    assert outcome.tool_calls == []


def test_missing_chain_asks_before_building_transaction(monkeypatch):
    loop = _loop_with_transfer(monkeypatch, Intent.TRANSFER, _native_slots())
    outcome = _handle(loop, chain=None)

    assert outcome.kind == "ask"
    assert outcome.content == MSG_CHAIN_REQUIRED
    assert outcome.tool_calls == []


def test_unimplemented_write_intent_keeps_notice(monkeypatch):
    slots = {"spenderContractAddress": _TO, "tokenSymbol": "BERA", "humanAmount": "1"}
    loop = _loop_with_transfer(monkeypatch, Intent.APPROVE, slots)
    outcome = _handle(loop)

    assert outcome.kind == "answer"
    assert outcome.content == MSG_WRITE_NOT_ENABLED


def test_task_creation_failure_returns_no_confirm_card(monkeypatch):
    """建不出任务就不能给卡片：没有 taskId 的卡片点了也没有接口能接。"""
    loop = _loop_with_transfer(monkeypatch, Intent.TRANSFER, _native_slots())

    async def boom(**kwargs):
        """模拟建任务失败。

        Args:
            **kwargs: 建任务入参，此替身不关心。
        """
        raise RuntimeError("db down")

    monkeypatch.setattr(workflow_service, "create_transfer", boom)
    outcome = _handle(loop)

    assert outcome.kind == "answer"
    assert outcome.content == MSG_WRITE_TASK_FAILED
    assert outcome.intent_preview is None
    assert outcome.task_id is None


def test_model_call_source_cannot_reach_transfer_tool(monkeypatch):
    registry = ToolRegistry()
    registry.register(
        build_post_transfer(_Adapter()), ToolGroup.STATEMACHINE_ONLY, mutating=True
    )
    runner = ToolRunner(registry)

    with pytest.raises(ToolSourceViolation):
        asyncio.run(
            runner.run(
                "post_transfer",
                {
                    "from_address": _FROM,
                    "to_address": _TO,
                    "token_symbol": "BERA",
                    "human_amount": "0.1",
                    "chain_id": _CHAIN,
                },
                source=CallSource.LLM,
            )
        )


def test_history_item_exposes_task_and_preview():
    item = HistoryItem(
        id=1,
        role="assistant",
        content="请确认转账：0.1 BERA。",
        task_id=_TASK,
        intent_preview={
            "action": "transfer",
            "tokenSymbol": "BERA",
            "humanAmount": "0.1",
            "target": _TO,
        },
        created_at=datetime.now(timezone.utc),
    )

    dumped = item.model_dump(by_alias=True)
    assert dumped["taskId"] == _TASK
    assert dumped["intentPreview"]["humanAmount"] == "0.1"


def test_router_renders_confirm_preview_in_camel_case(monkeypatch):
    outcome = ChatOutcome(
        kind="intent_confirm",
        content="请确认转账：0.1 BERA 转给 0x2222。",
        intent_preview={
            "action": "transfer",
            "tokenSymbol": "BERA",
            "humanAmount": "0.1",
            "target": _TO,
        },
        task_id=_TASK,
    )
    dumped, *_ = _run(
        monkeypatch, ("new", None), container=_container(handle=AsyncMock(return_value=outcome))
    )

    assert dumped["type"] == "intent_confirm"
    assert dumped["intentPreview"] == outcome.intent_preview
    assert dumped["taskId"] == _TASK
    assert dumped["txIntent"] is None
