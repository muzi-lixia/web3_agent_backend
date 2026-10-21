"""摘要触发判定：阈值与失败熔断。纯函数，不碰库、不调模型。"""

from __future__ import annotations

from web3_agent.config.settings import Settings
from web3_agent.domain.memory.summarizer import Summarizer


def _summarizer() -> Summarizer:
    return Summarizer(settings=Settings(), llm=None, audit=None, quota=None)  # type: ignore[arg-type]


def test_below_threshold_does_not_run() -> None:
    assert (
        _summarizer().should_run(pending=9, status=None, fail_count=0) is False
    )


def test_first_summary_runs_at_threshold() -> None:
    assert _summarizer().should_run(pending=10, status=None, fail_count=0) is True


def test_ok_status_keeps_running() -> None:
    assert _summarizer().should_run(pending=10, status="ok", fail_count=0) is True


def test_failed_status_retries_until_limit() -> None:
    s = _summarizer()
    assert s.should_run(pending=10, status="failed", fail_count=2) is True
    assert s.should_run(pending=10, status="failed", fail_count=3) is False


def test_failed_status_ignores_threshold_once_melted() -> None:
    s = _summarizer()
    assert s.should_run(pending=100, status="failed", fail_count=99) is False
