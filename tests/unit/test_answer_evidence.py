"""数据类意图的证据校验：没有对应工具的成功调用就不能给数据性回答。"""

from __future__ import annotations

import pytest

from web3_agent.agent.agent_loop import _lacks_evidence
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import Intent

_EXPECTED = {
    Intent.QUERY_ASSET: "get_asset_balance",
    Intent.QUERY_TX: "get_transaction_records",
    Intent.QUERY_STAKE: "get_stake_info",
}


def _record(name: str, *, ok: bool = True) -> ToolCallRecord:
    return ToolCallRecord(name=name, ok=ok, error=None if ok else "E303")


@pytest.mark.parametrize("intent", list(_EXPECTED))
def test_no_tool_call_at_all_has_no_evidence(intent: Intent) -> None:
    assert _lacks_evidence(intent, []) is True


@pytest.mark.parametrize("intent", list(_EXPECTED))
def test_all_tool_calls_failed_has_no_evidence(intent: Intent) -> None:
    tool = _EXPECTED[intent]
    assert _lacks_evidence(intent, [_record(tool, ok=False), _record(tool, ok=False)]) is True


@pytest.mark.parametrize("intent", list(_EXPECTED))
def test_one_successful_call_is_enough(intent: Intent) -> None:
    tool = _EXPECTED[intent]
    assert _lacks_evidence(intent, [_record(tool, ok=False), _record(tool, ok=True)]) is False


@pytest.mark.parametrize("intent", list(_EXPECTED))
def test_other_tool_success_is_not_evidence(intent: Intent) -> None:
    # 问余额却调了质押：调了、也成功了，但没查到余额。
    other = next(tool for i, tool in _EXPECTED.items() if i is not intent)
    assert _lacks_evidence(intent, [_record(other, ok=True)]) is True


def test_document_success_is_not_live_chain_evidence() -> None:
    assert _lacks_evidence(Intent.QUERY_ASSET, [_record("rag_general_knowledge", ok=True)]) is True


def test_chitchat_needs_no_evidence() -> None:
    assert _lacks_evidence(Intent.CHITCHAT, []) is False


def test_out_of_scope_needs_no_evidence() -> None:
    assert _lacks_evidence(Intent.OUT_OF_SCOPE, []) is False


def test_write_intent_is_judged_by_validators_not_here() -> None:
    # 写链路的护栏在校验层，这里不该顺带拦它。
    assert _lacks_evidence(Intent.TRANSFER, []) is False
