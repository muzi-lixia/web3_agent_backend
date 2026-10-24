"""按 tests/golden_suite/tool_cases.json 跑批，统计意图识别与工具选择。

用法：
    .venv/bin/python scripts/eval_tools.py
    .venv/bin/python scripts/eval_tools.py --only tf- --json .tmp/eval.json

每条用例只跑第一次决策（意图抽取 + 只读图的 agent 节点），不执行工具：
评估对象是模型选了什么，工具执行结果由集成测试覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# ROOT：后端项目根目录，供离线评估定位源码与用例。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from web3_agent.agent.graph.read_graph import initial_state  # noqa: E402
from web3_agent.agent.intent_router import route_intent  # noqa: E402
from web3_agent.agent.json_guard import JsonGuard  # noqa: E402
from web3_agent.agent.llm_adapter import LlmAdapter  # noqa: E402
from web3_agent.agent.tool_registry import ToolRegistry  # noqa: E402
from web3_agent.config.constants import Route, ToolGroup  # noqa: E402
from web3_agent.config.settings import Settings  # noqa: E402
from web3_agent.web3.chain_adapter import ChainAdapter  # noqa: E402
from web3_agent.web3.tools.read.get_asset_balance import build_get_asset_balance  # noqa: E402
from web3_agent.web3.tools.read.get_stake_info import build_get_stake_info  # noqa: E402
from web3_agent.web3.tools.read.get_transaction_records import (  # noqa: E402
    build_get_transaction_records,
)

# CASES_PATH：离线工具评估用例 JSON 的默认路径。
CASES_PATH = ROOT / "tests" / "golden_suite" / "tool_cases.json"
# 只判工具选择、不执行工具，所以地址用固定值，不需要真实钱包。
# WALLET：离线选工具评估使用的固定钱包占位地址。
WALLET = "0x63a9f8df422e1983d42565234557d46d18dbece9"


# 数据职责：一条离线工具评估用例的实测结果与各维度差异。
@dataclass
class CaseResult:
    # case：原始评估用例，包含输入及期望意图、路由、槽位和工具。
    case: dict[str, Any]
    # intent：本轮识别出的业务意图；取值受 Intent 枚举约束。
    intent: str | None = None
    # route：后端决定的处理路径：read 查询、write 写操作、clarify 澄清、reject 拒绝。
    route: str | None = None
    # tools：模型本次选择的工具名称列表。
    tools: list[str] | None = None
    # degraded：是否使用降级结果，例如缺少数据证据或结构化输出失败。
    degraded: bool = False
    # intent_ok：实际识别意图是否符合用例预期。
    intent_ok: bool = False
    # route_ok：实际路由是否符合用例预期。
    route_ok: bool = False
    # tools_ok：实际工具集合是否与用例期望一致。
    tools_ok: bool = False
    # slot_ok：必需槽位是否存在且值与预期一致。
    slot_ok: bool = True
    # missing_slots：评估用例要求、但模型未返回的槽位名列表。
    missing_slots: list[str] = None  # type: ignore[assignment]
    # wrong_slots：值不符的槽位，映射到 (期望值, 实际值)。
    wrong_slots: dict[str, tuple[str, str]] = None  # type: ignore[assignment]
    # extra_tools：模型多选出的工具名称。
    extra_tools: list[str] = None  # type: ignore[assignment]
    # missing_tools：模型漏选的工具名称。
    missing_tools: list[str] = None  # type: ignore[assignment]
    # redundant：重复选择工具的次数。
    redundant: int = 0
    # error：评估执行时捕获的异常说明，空字符串表示未捕获异常。
    error: str = ""

    def __post_init__(self) -> None:
        """初始化评估结果的可变集合，避免空默认值影响后续比较。"""
        self.missing_slots = self.missing_slots or []
        self.wrong_slots = self.wrong_slots or {}
        self.extra_tools = self.extra_tools or []
        self.missing_tools = self.missing_tools or []

    @property
    def known_gap(self) -> bool:
        """读取该评估用例是否标记为已知能力缺口。"""
        return bool(self.case.get("known_gap"))

    @property
    def passed(self) -> bool:
        """判断该评估用例的意图、路由、工具及槽位是否全部通过。"""
        return self.intent_ok and self.route_ok and self.tools_ok and self.slot_ok


def _norm(key: str, value: Any) -> Any:
    """按槽位类型归一，避免因为大小写或 10 / 10.0 这种差别判错。

    Args:
        key: 当前字段名、缓存键或幂等键，具体以当前操作为准。
        value: 待校验、转换或缓存的值。
    """
    if value is None:
        return None
    if key == "limit":
        try:
            return int(str(value))
        except ValueError:
            return str(value)
    if key == "tokenSymbol":
        return str(value).strip().upper()
    if key == "humanAmount":
        try:
            return Decimal(str(value).replace(",", "").strip())
        except InvalidOperation:
            return str(value).strip()
    # 地址与合约别名的共同点：都不该大小写敏感。
    return str(value).strip().lower()


def _score_slots(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str], dict]:
    """比较必需槽位，返回是否通过、缺失项和数值不符项。

    Args:
        expected: 用例要求的期望值。
        actual: 模型实际输出值。
    """
    missing: list[str] = []
    wrong: dict[str, tuple[str, str]] = {}
    for key, want in expected.items():
        if key not in actual:
            missing.append(key)
            continue
        if _norm(key, actual[key]) != _norm(key, want):
            wrong[key] = (str(want), str(actual[key]))
    return (not missing and not wrong), missing, wrong


def _intent_ok(expected: Any, actual: str | None) -> bool:
    """判断实际意图是否在用例允许的单个或多个意图中。

    Args:
        expected: 用例要求的期望值。
        actual: 模型实际输出值。
    """
    if actual is None:
        return False
    wanted = expected if isinstance(expected, list) else [expected]
    return actual in wanted


async def _evaluate(cases: list[dict], settings: Settings) -> list[CaseResult]:
    """遍历用例执行意图识别及首次模型选工具，并汇总各维度结果。

    Args:
        cases: 本批待执行的评估用例列表。
        settings: 运行配置对象，提供本模块所需开关、阈值和服务地址。
    """
    adapter = ChainAdapter(settings)
    registry = ToolRegistry()
    registry.register(build_get_asset_balance(adapter), ToolGroup.LLM_VISIBLE, mutating=False)
    registry.register(
        build_get_transaction_records(settings), ToolGroup.LLM_VISIBLE, mutating=False
    )
    registry.register(
        build_get_stake_info(adapter, settings.stake_contracts),
        ToolGroup.LLM_VISIBLE,
        mutating=False,
    )

    llm = LlmAdapter(settings)
    model = llm.bind_tools(registry.llm_tools())
    guard = JsonGuard()

    results: list[CaseResult] = []
    for case in cases:
        history = [tuple(pair) for pair in case.get("history", [])]
        message = case["message"]
        result = CaseResult(case=case)

        guarded = await guard.run(
            lambda h=history, m=message: llm.extract_intent(h, m)
        )
        if guarded.degraded or guarded.output is None:
            result.degraded = True
            results.append(result)
            continue

        result.intent = guarded.output.intent.value
        # 以确定性后端规则选择路径，不让模型输出直接决定执行权限。
        decision = route_intent(
            guarded.output, confidence_threshold=settings.confidence_threshold
        )
        result.route = decision.route.value
        result.intent_ok = _intent_ok(case["intent"], result.intent)
        result.route_ok = result.route == case["route"]
        result.slot_ok, result.missing_slots, result.wrong_slots = _score_slots(
            case.get("slots", {}), guarded.output.slots
        )

        if decision.route is Route.READ:
            state = initial_state(
                history, message, wallet_address=WALLET, intent=guarded.output.intent
            )
            try:
                response = await model.ainvoke(state["messages"])
            except Exception as exc:  # noqa: BLE001 单条失败不该中断整批
                result.error = f"{type(exc).__name__}: {exc}"
                results.append(result)
                continue
            calls = [str(call.get("name", "")) for call in getattr(response, "tool_calls", [])]
        else:
            calls = []

        result.tools = calls
        expected_tools = set(case.get("tools", []))
        result.extra_tools = sorted(set(calls) - expected_tools)
        result.missing_tools = sorted(expected_tools - set(calls))
        result.tools_ok = not result.extra_tools and not result.missing_tools
        result.redundant = len(calls) - len(set(calls))
        results.append(result)

    return results


def _rate(hit: int, total: int) -> str:
    """把命中数与总数格式化为 hit/total 文本。

    Args:
        hit: 通过检查的用例数。
        total: 参与检查的用例总数。
    """
    return f"{hit}/{total}" if total else "0/0"


def _pct(hit: int, total: int) -> str:
    """把命中率格式化为百分比，无样本时显示占位符。

    Args:
        hit: 通过检查的用例数。
        total: 参与检查的用例总数。
    """
    return f"{hit / total * 100:.1f}%" if total else "-"


def _print_group(title: str, results: list[CaseResult]) -> None:
    """汇总并打印一组用例的意图、路由、工具和槽位准确率。

    Args:
        title: 报告分组标题。
        results: 本批评估结果列表。
    """
    total = len(results)
    if not total:
        return
    degraded = sum(1 for r in results if r.degraded)
    intent_hit = sum(1 for r in results if r.intent_ok)
    route_hit = sum(1 for r in results if r.route_ok)
    tools_hit = sum(1 for r in results if r.tools_ok)
    slots_total = sum(1 for r in results if r.case.get("slots"))
    slots_hit = sum(1 for r in results if r.case.get("slots") and r.slot_ok)
    redundant = sum(r.redundant for r in results)

    print(f"\n{title}（{total} 条）")
    print(f"  {'意图':<8}{_rate(intent_hit, total):>8}  {_pct(intent_hit, total):>7}")
    print(f"  {'路由':<8}{_rate(route_hit, total):>8}  {_pct(route_hit, total):>7}")
    print(f"  {'工具集合':<6}{_rate(tools_hit, total):>8}  {_pct(tools_hit, total):>7}")
    print(f"  {'槽位':<8}{_rate(slots_hit, slots_total):>8}  {_pct(slots_hit, slots_total):>7}")
    print(f"  冗余调用 {redundant} 次 ｜ JSON 降级 {degraded} 次")


def _print_failures(results: list[CaseResult]) -> None:
    """打印失败或结构化输出降级用例的具体差异。

    Args:
        results: 本批评估结果列表。
    """
    failed = [r for r in results if not r.degraded and not r.passed]
    degraded = [r for r in results if r.degraded]
    if not failed and not degraded:
        return

    print("\n失败明细")
    for r in failed:
        marks = "".join(
            [
                f"意图{'✓' if r.intent_ok else '✗'} ",
                f"路由{'✓' if r.route_ok else '✗'} ",
                f"工具{'✓' if r.tools_ok else '✗'} ",
                f"槽位{'✓' if r.slot_ok else '✗'}",
            ]
        )
        print(f"  {r.case['id']:<8} {marks}")
        print(f"           {r.case['message']}")
        want_intent = r.case["intent"]
        if not r.intent_ok:
            print(f"           意图：期望 {want_intent} 实际 {r.intent}")
        if not r.route_ok:
            print(f"           路由：期望 {r.case['route']} 实际 {r.route}")
        if not r.tools_ok:
            detail = []
            if r.missing_tools:
                detail.append(f"漏调 {r.missing_tools}")
            if r.extra_tools:
                detail.append(f"多调 {r.extra_tools}")
            print(f"           工具：{'，'.join(detail)}（实际 {r.tools}）")
        if not r.slot_ok:
            if r.missing_slots:
                print(f"           缺槽位 {r.missing_slots}")
            for key, (want, got) in r.wrong_slots.items():
                print(f"           槽位 {key}：期望 {want} 实际 {got}")
        if r.error:
            print(f"           异常：{r.error}")

    for r in degraded:
        print(f"  {r.case['id']:<8} JSON 降级（重试后仍未拿到合法结构）")
        print(f"           {r.case['message']}")


async def main() -> None:
    """筛选评估用例，执行模型选工具评估并输出统计和可选 JSON 报告。"""
    parser = argparse.ArgumentParser(description="工具调用评估集跑批")
    parser.add_argument("--only", default=None, help="只跑 id 包含该子串的用例")
    parser.add_argument("--json", default=None, help="把逐条结果写到该路径")
    args = parser.parse_args()

    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if not cases:
        print("没有匹配的用例")
        return

    settings = Settings()
    print(f"模型 {settings.ollama_model} ｜ 用例 {len(cases)} 条 ｜ {CASES_PATH.name}")

    started = time.perf_counter()
    results = await _evaluate(cases, settings)
    elapsed = time.perf_counter() - started

    scored = [r for r in results if not r.known_gap]
    gaps = [r for r in results if r.known_gap]

    _print_group("计分组", scored)
    _print_group("known_gap 组（写链路未接，只看意图与槽位）", gaps)
    _print_failures(results)
    print(f"\n耗时 {elapsed:.1f}s")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                [
                    {
                        "id": r.case["id"],
                        "message": r.case["message"],
                        "known_gap": r.known_gap,
                        "expected": {
                            "intent": r.case["intent"],
                            "route": r.case["route"],
                            "tools": r.case.get("tools", []),
                            "slots": r.case.get("slots", {}),
                        },
                        "actual": {
                            "intent": r.intent,
                            "route": r.route,
                            "tools": r.tools,
                            "degraded": r.degraded,
                            "error": r.error,
                        },
                        "ok": {
                            "intent": r.intent_ok,
                            "route": r.route_ok,
                            "tools": r.tools_ok,
                            "slots": r.slot_ok,
                        },
                        "redundant": r.redundant,
                        "missing_slots": r.missing_slots,
                        "wrong_slots": r.wrong_slots,
                    }
                    for r in results
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"明细已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
