"""数据类意图的证据校验：没有成功的工具调用就不能给数据性回答。"""

from __future__ import annotations

import pytest

from web3_agent.agent.agent_loop import _lacks_evidence
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import Intent


def _record(name: str = "get_asset_balance", *, ok: bool = True) -> ToolCallRecord:
    return ToolCallRecord(name=name, ok=ok, error=None if ok else "E303")


@pytest.mark.parametrize(
    "intent", [Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE]
)
def test_no_tool_call_at_all_has_no_evidence(intent: Intent) -> None:
    assert _lacks_evidence(intent, []) is True


@pytest.mark.parametrize(
    "intent", [Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE]
)
def test_all_tool_calls_failed_has_no_evidence(intent: Intent) -> None:
    assert _lacks_evidence(intent, [_record(ok=False), _record(ok=False)]) is True


@pytest.mark.parametrize(
    "intent", [Intent.QUERY_ASSET, Intent.QUERY_TX, Intent.QUERY_STAKE]
)
def test_one_successful_call_is_enough(intent: Intent) -> None:
    assert _lacks_evidence(intent, [_record(ok=False), _record(ok=True)]) is False


def test_chitchat_needs_no_evidence() -> None:
    assert _lacks_evidence(Intent.CHITCHAT, []) is False


def test_out_of_scope_needs_no_evidence() -> None:
    assert _lacks_evidence(Intent.OUT_OF_SCOPE, []) is False


def test_write_intent_is_judged_by_validators_not_here() -> None:
    # 写链路的护栏在校验层，这里不该顺带拦它。
    assert _lacks_evidence(Intent.TRANSFER, []) is False
