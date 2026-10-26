"""按 tests/golden_suite/rag_cases.json 跑检索层评估。

用法：
    .venv/bin/python scripts/eval_rag.py                # 两套配置都跑，打印对比
    .venv/bin/python scripts/eval_rag.py --profile prod # 只跑生产配置
    .venv/bin/python scripts/eval_rag.py --json .tmp/rag.json

只调向量化与检索，不调生成模型；问题改写开启时额外调用一次改写模型。
答案层的正确性（有没有编造数字、有没有答全）不在本脚本范围，靠 cases 里的
gold / gold_terms / must_not 人工或另一层判分。

两套配置：
    prod —— 完全按 .env，反映线上真实表现
    wide —— top_k=5，用来判断「检索不到」还是「装不下」
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ROOT：后端项目根目录，供离线评估定位源码与用例。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg.base import dispose_engine, init_engine
from web3_agent.rag.embedder import Embedder
from web3_agent.rag.models import embedding_profile
from web3_agent.rag.retriever import KnowledgeRetriever
from web3_agent.rag.vector_store import PgKnowledgeStore

# CASES_PATH：检索层评估用例 JSON 的默认路径。
CASES_PATH = ROOT / "tests" / "golden_suite" / "rag_cases.json"

# PROFILES：两套检索配置。prod 不改任何参数，wide 把证据条数放到上限。
PROFILES: dict[str, dict] = {
    "prod": {},
    "wide": {"rag_top_k": 5},
}

# ANSWERABLE：有期望切块、按命中率统计的类型。
ANSWERABLE = ("fact", "paraphrase", "zh", "multi", "confusable")


@dataclass
class CaseResult:
    """单条用例的检索实测结果。"""

    case_id: str
    case_type: str
    query: str
    status: str
    expect: list[str]
    got: list[str] = field(default_factory=list)
    top1_distance: float | None = None
    # rank：第一个期望切块在结果里的名次，0 表示没命中。
    rank: int = 0
    # recall：期望切块在 top_k 内的覆盖率，跨块题看这个。
    recall: float = 0.0


def _score(case: dict, hits: list[dict], status: str) -> CaseResult:
    """把一次检索返回映射成可统计的结果。

    Args:
        case: 单条评估用例。
        hits: 检索返回的证据列表，含 chunk_id 与 distance。
        status: 检索状态，success / empty / invalid / unavailable。
    """
    expect = case.get("expect_chunks") or []
    got = [h.get("chunk_id", "") for h in hits]
    res = CaseResult(
        case_id=case["id"],
        case_type=case["type"],
        query=case["query"],
        status=status,
        expect=expect,
        got=got,
        top1_distance=hits[0].get("distance") if hits else None,
    )
    if not expect:
        return res
    ranks = [i + 1 for i, cid in enumerate(got) if cid in expect]
    res.rank = min(ranks) if ranks else 0
    res.recall = len({c for c in got if c in expect}) / len(expect)
    return res


def _summarize(rows: list[CaseResult], cases_by_id: dict[str, dict]) -> dict:
    """按类型汇总命中率、召回与 MRR。

    Args:
        rows: 全部用例的检索结果。
        cases_by_id: 用例 id 到原始用例的映射，用于取 match 模式。
    """
    out: dict = {}
    for ctype in ANSWERABLE:
        group = [r for r in rows if r.case_type == ctype]
        if not group:
            continue
        hits = []
        for r in group:
            need_all = cases_by_id[r.case_id].get("match") == "all"
            ok = r.recall == 1.0 if need_all else r.rank > 0
            hits.append(ok)
        out[ctype] = {
            "n": len(group),
            "hit@1": sum(1 for r, h in zip(group, hits) if h and r.rank == 1)
            / len(group),
            "hit@3": sum(1 for r, h in zip(group, hits) if h and 0 < r.rank <= 3)
            / len(group),
            "hit": sum(hits) / len(group),
            "recall": statistics.mean(r.recall for r in group),
            "mrr": statistics.mean(1 / r.rank if r.rank else 0.0 for r in group),
        }
    unans = [r for r in rows if r.case_type == "unanswerable"]
    if unans:
        refused = sum(1 for r in unans if r.status == "empty" or not r.got)
        out["unanswerable"] = {
            "n": len(unans),
            "refused": refused,
            "false_positive": [r.case_id for r in unans if r.got],
        }
    return out


async def _eval_profile(
    name: str, overrides: dict, cases: list[dict]
) -> list[CaseResult]:
    """在一套配置下跑完整用例集。

    Args:
        name: 配置名，仅用于打印。
        overrides: 覆盖到 Settings 上的字段。
        cases: 全部评估用例。
    """
    settings = get_settings()
    if overrides:
        settings = settings.model_copy(update=overrides)
    init_engine(settings)
    try:
        # 改写与其他可选阶段要和生产一样走同一个模型入口，否则评估测的是降级路径。
        llm = LlmAdapter(settings)
        retriever = KnowledgeRetriever(
            settings, Embedder(settings), PgKnowledgeStore(embedding_profile(settings)),
            llm.client,
        )
        rows: list[CaseResult] = []
        for case in cases:
            result = await retriever.search(
                case["query"], kb_id=case.get("kb_id", "default")
            )
            rows.append(
                _score(case, result.get("hits") or [], result.get("status", ""))
            )
        return rows
    finally:
        await dispose_engine()


def _print(
    name: str,
    overrides: dict,
    summary: dict,
    rows: list[CaseResult],
    cases_by_id: dict[str, dict],
) -> None:
    """打印单套配置的汇总与失败明细。

    Args:
        name: 配置名。
        overrides: 该配置覆盖的字段。
        summary: 按类型汇总的指标。
        rows: 全部用例结果，用于列出失败项。
        cases_by_id: 用例 id 到原始用例的映射，用于区分跨块题的判定口径。
    """
    head = "、".join(f"{k}={v}" for k, v in overrides.items()) or "按 .env"
    print(f"\n{'=' * 74}\n配置 {name}（{head}）\n{'=' * 74}")
    print(
        f"{'类型':<14}{'条数':>4}{'hit@1':>8}{'hit@3':>8}{'命中':>8}{'召回':>8}{'MRR':>8}"
    )
    for ctype, m in summary.items():
        if ctype == "unanswerable":
            continue
        print(
            f"{ctype:<14}{m['n']:>4}{m['hit@1']:>8.0%}{m['hit@3']:>8.0%}"
            f"{m['hit']:>8.0%}{m['recall']:>8.0%}{m['mrr']:>8.2f}"
        )
    unans = summary.get("unanswerable")
    if unans:
        print(
            f"\n无答案拦截：{unans['refused']}/{unans['n']} 条正确返回空"
            f"；误召回 {len(unans['false_positive'])} 条"
        )
        if unans["false_positive"]:
            print("  误召回：" + "、".join(unans["false_positive"]))

    misses = [r for r in rows if r.expect and r.rank == 0]
    if misses:
        print(f"\n未命中 {len(misses)} 条：")
        for r in misses:
            top = r.got[0][:12] if r.got else "（空）"
            print(f"  {r.case_id:<8} {r.query[:44]:<46} top1={top} d={r.top1_distance}")

    # 跨块题只召回一半时 rank>0 但不算命中，单列出来：这类会答成半个答案且不自知。
    partials = [
        r
        for r in rows
        if r.expect
        and r.rank > 0
        and r.recall < 1.0
        and cases_by_id[r.case_id].get("match") == "all"
    ]
    if partials:
        print(f"\n跨块题只召回部分 {len(partials)} 条：")
        for r in partials:
            print(f"  {r.case_id:<8} {r.query[:44]:<46} 召回 {r.recall:.0%}")


async def main() -> None:
    """解析参数、逐套配置跑评估并输出对比。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    parser.add_argument("--only", default="", help="只跑 id 前缀匹配的用例")
    parser.add_argument("--json", default="", help="把明细写到该路径")
    args = parser.parse_args()

    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.only:
        cases = [c for c in cases if c["id"].startswith(args.only)]
    cases_by_id = {c["id"]: c for c in cases}

    names = list(PROFILES) if args.profile == "all" else [args.profile]
    dump: dict[str, list[dict]] = {}
    for name in names:
        rows = await _eval_profile(name, PROFILES[name], cases)
        summary = _summarize(rows, cases_by_id)
        _print(name, PROFILES[name], summary, rows, cases_by_id)
        dump[name] = [r.__dict__ for r in rows]

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n明细已写入 {path}")


if __name__ == "__main__":
    asyncio.run(main())
