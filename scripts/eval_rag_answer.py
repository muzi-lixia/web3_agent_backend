"""按 tests/golden_suite/rag_cases.json 跑答案层评估。

检索层评估（eval_rag.py）只看「正确切块有没有召回」；本脚本继续往后走一步：
把召回的证据按生产提示词喂给本地模型，检查答案是否覆盖要点、该拒的有没有拒。

用法：
    .venv/bin/python scripts/eval_rag_answer.py                # 全量
    .venv/bin/python scripts/eval_rag_answer.py --only rn      # 只跑无答案题
    .venv/bin/python scripts/eval_rag_answer.py --json .tmp/answer.json

只调本地 Ollama（聊天 + 向量化），不产生外部费用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from web3_agent.agent.agent_loop import _untraceable
from web3_agent.agent.llm_adapter import LlmAdapter
from web3_agent.config.settings import get_settings
from web3_agent.persistence.pg.base import dispose_engine, init_engine
from web3_agent.rag.answer_prompt import answer_messages
from web3_agent.rag.embedder import Embedder
from web3_agent.rag.models import embedding_profile
from web3_agent.rag.retriever import NO_DOCUMENTS, KnowledgeRetriever
from web3_agent.rag.vector_store import PgKnowledgeStore

CASES_PATH = ROOT / "tests" / "golden_suite" / "rag_cases.json"

# 拒答句式：出现任意一条，说明模型没有硬给一个值。
REFUSAL = (
    "不足以回答", "无法回答", "无法确认", "未提及", "未说明", "没有说明",
    "不包含", "查不到", "不确定", "没有相关信息", "没有该", "暂未",
    "not enough", "does not contain", "no information", "not specified",
    "cannot", "can't", "unable to", "not mentioned", "no data",
    "do not provide", "does not provide", "do not include", "does not include",
    "not provide", "no address", "not available",
)

# 泄漏检测：无答案题被要求「不得给出具体数值」，按题型给不同的表达式。
LEAK_PATTERNS = {
    "price": r"(?:\$|USD|USDT?)\s?\d|\d+(?:\.\d+)?\s?(?:美元|美金)",
    "percent": r"\d+(?:\.\d+)?\s?%",
    "address": r"0x[0-9a-fA-F]{6,}",
    "date": r"\d{4}\s?年|\d{1,2}\s?月|Q[1-4]\s?20\d\d|20\d\d-\d{2}",
    "spec": r"\d+\s?(?:GB|TB|核|cores?)",
    "amount": r"\d[\d,]*\s?(?:BERA|ETH|USDC)",
}
LEAK_BY_CASE = {
    "rn-01": ["price"],
    "rn-02": ["price", "amount"],
    "rn-03": ["percent"],
    "rn-04": ["address"],
    "rn-05": ["address"],
    "rn-06": ["date"],
    "rn-07": ["amount"],
    "rn-08": [],
    "rn-09": ["spec"],
}


def _norm(text: str) -> str:
    """比较前抹掉千分位与多余空白，避免格式差异被当成答错。"""
    return re.sub(r"\s+", " ", text.replace(",", "")).lower()


def _numbers(text: str) -> set[str]:
    """抽出文本里的数字，用于跨语言比对。

    gold 与答案经常一边写「7 天」一边写「7-day」，术语比对会漏；数字不受语言影响。
    """
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def _lang_ok(question: str, answer: str) -> bool:
    """回答语言是否与提问一致。

    中文提问时，`gold_terms` 是英文、证据也是英文，模型容易整段照抄英文；
    这类答案内容可能全对，但对中文用户是错的。只统计汉字数与英文词数，
    允许保留协议名、代币名等必要术语，也允许「BUSD 以前叫 HONEY」这种极短回答。

    Args:
        question: 用户原始问题。
        answer: 模型回答正文。
    """
    if not re.search(r"[\u4e00-\u9fff]", question):
        return True
    han = len(re.findall(r"[\u4e00-\u9fff]", answer))
    words = len(re.findall(r"[A-Za-z]{3,}", answer))
    return han >= 3 and han >= words * 0.5


def _grade(case: dict, answer: str, citations: list[dict]) -> dict:
    """按用例类型给出可复核的判定。

    有答案题看「要点有没有答到」：术语命中或 gold 里的关键数字命中都算答到，
    只说了「资料不足」又什么都没给才算失败。术语一个没中但数字对上的情况很常见
    （英文 gold 与中文答案），单靠字面会把正确答案判成错。
    无答案题看「有没有拒答」且「有没有给出被禁的具体值」。

    Args:
        case: 单条评估用例。
        answer: 模型最终回答正文（已过溯源校验）。
        citations: 本轮证据列表。
    """
    text = _norm(answer)
    # answer_terms 是答案层专用的判定词：中文问句的金标是中文，而 gold_terms
    # 记的是检索语料里的英文原词，直接拿来比对中文回答必然落空。
    terms = [_norm(t) for t in (case.get("answer_terms") or case.get("gold_terms") or [])]
    term_hits = [t for t in terms if t in text]
    gold_numbers = _numbers(case.get("gold", ""))
    number_hits = sorted(n for n in gold_numbers if n in _numbers(answer))
    refused = any(p.lower() in text for p in REFUSAL)
    lang_ok = _lang_ok(case["query"], answer)
    leaked = [
        kind
        for kind in LEAK_BY_CASE.get(case["id"], [])
        if re.search(LEAK_PATTERNS[kind], answer)
    ]
    covered = bool(term_hits or number_hits)
    if case["type"] == "unanswerable":
        ok = refused and not leaked
        full = ok
    else:
        ok = covered and lang_ok
        # 全部要点都答到：术语全中，或 gold 里的数字全中。
        full = ok and (
            (bool(terms) and len(term_hits) == len(terms))
            or (bool(gold_numbers) and len(number_hits) == len(gold_numbers))
        )
    return {
        "ok": ok,
        "full": full,
        "refused": refused,
        "lang_ok": lang_ok,
        "term_hits": term_hits,
        "number_hits": number_hits,
        "leaked": leaked,
        "gold_terms": terms,
        # 先拒后答：两边都占，说明它其实读到了，只是表达上自相矛盾。
        "contradiction": refused and covered,
        "untraceable": _untraceable(answer, citations) if citations else False,
    }


async def _run(cases: list[dict], dump_path: str) -> None:
    """逐条跑检索、生成与判分，并打印可复核的明细。

    Args:
        cases: 待评估用例。
        dump_path: 明细 JSON 输出路径，空字符串表示不落盘。
    """
    settings = get_settings()
    init_engine(settings)
    try:
        retriever = KnowledgeRetriever(
            settings, Embedder(settings), PgKnowledgeStore(embedding_profile(settings))
        )
        llm = LlmAdapter(settings)
        rows: list[dict] = []
        for idx, case in enumerate(cases, 1):
            started = time.monotonic()
            result = await retriever.search(case["query"], kb_id=case.get("kb_id", "default"))
            citations = result.get("hits") or []
            if not citations:
                answer = NO_DOCUMENTS
                generated = False
            else:
                generated = True
                message = await llm.answer(answer_messages(case["query"], citations))
                answer = str(message.content).strip()
                if _untraceable(answer, citations):
                    answer = NO_DOCUMENTS
            grade = _grade(case, answer, citations)
            elapsed = time.monotonic() - started
            rows.append(
                {
                    "id": case["id"],
                    "type": case["type"],
                    "query": case["query"],
                    "gold": case.get("gold", ""),
                    "retrieved": [c.get("chunk_id", "")[:12] for c in citations],
                    "answer": answer,
                    "generated": generated,
                    "elapsed_s": round(elapsed, 1),
                    **grade,
                }
            )
            flag = "OK " if grade["ok"] else "FAIL"
            print(f"[{idx:>2}/{len(cases)}] {case['id']:<7} {flag} {elapsed:5.1f}s "
                  f"{'空' if not generated else ''}", flush=True)
            if not grade["ok"]:
                print(f"        Q: {case['query']}")
                print(f"        A: {answer[:220]}")
                print(f"        gold={case.get('gold','')} | terms={grade['term_hits']}"
                      f" | numbers={grade['number_hits']}"
                      f" | refused={grade['refused']} | leaked={grade['leaked']}")

        _print_summary(rows)
        if dump_path:
            path = Path(dump_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n明细已写入 {path}")
    finally:
        await dispose_engine()


def _print_summary(rows: list[dict]) -> None:
    """按类型汇总通过率；无答案题单列拒答与泄漏。"""
    print(f"\n{'=' * 70}\n答案层汇总\n{'=' * 70}")
    types = sorted({r["type"] for r in rows})
    for ctype in types:
        group = [r for r in rows if r["type"] == ctype]
        ok = sum(1 for r in group if r["ok"])
        full = sum(1 for r in group if r["full"])
        print(f"{ctype:<14}{ok:>2}/{len(group):<3}{ok / len(group):>6.0%}"
              f"   全部要点 {full}/{len(group)}")
    total_ok = sum(1 for r in rows if r["ok"])
    print(f"{'合计':<14}{total_ok:>2}/{len(rows):<3}{total_ok / len(rows):>6.0%}")

    unans = [r for r in rows if r["type"] == "unanswerable"]
    if unans:
        blocked = [r["id"] for r in unans if r["ok"]]
        leaked = [(r["id"], r["leaked"]) for r in unans if r["leaked"]]
        print(f"\n无答案题正确拒答 {len(blocked)}/{len(unans)}：{'、'.join(blocked) or '无'}")
        if leaked:
            print("  给出禁给数值：" + "；".join(f"{i} {k}" for i, k in leaked))
    missed = [r["id"] for r in rows if not r["ok"]]
    if missed:
        print("未通过：" + "、".join(missed))
    partial = [r["id"] for r in rows if r["ok"] and r["type"] != "unanswerable" and not r["full"]]
    if partial:
        print("要点不全：" + "、".join(partial))
    mixed = [r["id"] for r in rows if r["contradiction"]]
    if mixed:
        print("先拒后答：" + "、".join(mixed))
    untraced = [r["id"] for r in rows if r["untraceable"]]
    if untraced:
        print("溯源校验命中：" + "、".join(untraced))


def _regrade(path: str) -> None:
    """用已有明细重算判定，不再调用模型。

    Args:
        path: 上一次评估写出的明细 JSON。
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]}
    for row in rows:
        row.pop("gold_terms", None)
        kept = row.get("untraceable", False)
        row.update(_grade(cases[row["id"]], row["answer"], []))
        row["untraceable"] = kept
    for row in rows:
        mark = "OK " if row["ok"] else "FAIL"
        if row["ok"] and not row["full"]:
            mark = "PART"
        print(f"{row['id']:<8}{mark}  terms={row['term_hits']} numbers={row['number_hits']}"
              f" refused={row['refused']} leaked={row['leaked']}")
    _print_summary(rows)
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已用新判定回写 {path}")


async def main() -> None:
    """解析参数并跑答案层评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="只跑 id 前缀匹配的用例")
    parser.add_argument("--json", default="", help="把明细写到该路径")
    parser.add_argument("--regrade", default="", help="用已有明细重算判定，不调模型")
    args = parser.parse_args()

    if args.regrade:
        _regrade(args.regrade)
        return

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if c["id"].startswith(args.only)]
    await _run(cases, args.json)


if __name__ == "__main__":
    asyncio.run(main())
