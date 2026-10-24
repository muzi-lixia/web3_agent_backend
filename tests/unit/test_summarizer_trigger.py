"""摘要触发判定：只看挤出窗口的条数。纯函数，不碰库、不调模型。"""

from __future__ import annotations

from web3_agent.config.settings import Settings
from web3_agent.domain.memory.summarizer import Summarizer


def _summarizer() -> Summarizer:
    return Summarizer(settings=Settings(), llm=None, audit=None, quota=None)  # type: ignore[arg-type]


def test_below_threshold_does_not_run() -> None:
    assert _summarizer().should_run(pending=9) is False


def test_runs_at_threshold() -> None:
    assert _summarizer().should_run(pending=10) is True


def test_runs_on_backlog() -> None:
    # 积压多少都只压一批，但判定上不该因为「积压太多」而停下来。
    assert _summarizer().should_run(pending=100) is True
