"""引用检查与敏感规则逐句校验；无法确认时退回原文证据。"""
import re
import json

_CITATION = re.compile(r"\[(S[1-9]\d*)\]")
_SENSITIVE = re.compile(r"\d|%|罚|罚没|解锁|锁仓|周期|奖励|赎回|冷却|slashing|unlock|penalty|reward|手续费|滑点|权限|授权|fee", re.I)

# 句子边界。英文句号只在后面是空白或结尾时才算，避免把 2.5%、v1.2 切开。
_SENTENCE = re.compile(r"[。！？!?\n]|\.(?=\s|$)")

# 中文数词只在后面跟量词或单位时折算，避免把「一般」「一起」当成数字。
_CN_NUM = re.compile(r"[零一二两三四五六七八九十百千万]+(?=[天日周月年秒分时个次倍条笔枚位%％])")
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def _cn_to_int(text):
    """Only accept digit-by-digit notation and unambiguous integers below 100.

    Unsupported expressions stay unchanged; never guess a numerical equivalence.
    """
    if all(ch in _CN_DIGITS for ch in text):
        return int("".join(str(_CN_DIGITS[ch]) for ch in text))
    if re.fullmatch(r"[一二两三四五六七八九]?十[一二三四五六七八九]?", text):
        left, right = text.split("十")
        return (_CN_DIGITS[left] if left else 1) * 10 + (_CN_DIGITS[right] if right else 0)
    return text


def _normalized(text):
    # 去掉引用和句末标点：草稿里的「days.[S1]」去掉引用后会留下一个句点，
    # 而原文按句界切出来的分段没有，不剥掉就永远对不上。
    text = _CITATION.sub("", text).strip(" \t\n。！？!?.,；;：:")
    # 「七天」和原文的「7天」是同一个事实，比对前统一成阿拉伯数字。
    text = _CN_NUM.sub(lambda m: str(_cn_to_int(m.group())), text)
    return re.sub(r"[^\w.%<>=≤≥+/\-]+", "", text, flags=re.UNICODE).casefold()


def references(citations):
    return "\n\n来源：\n" + "\n".join(
        f"[{hit['id']}] {hit['title']} · {hit['section']} · 版本 {hit['version']} · 知识库 {hit.get('kb_id', 'default')} · "
        + (f"第 {hit['page_number']}-{hit.get('page_end') or hit['page_number']} 页 · " if hit.get('page_number') else "")
        + f"链 {hit.get('chain_id') or '通用'} / 协议 {hit.get('protocol_id') or '通用'} / 分类 {hit['doc_category']} · 索引更新 {hit['updated_at']}"
        + (f"\n{hit['source_url']}" if hit.get('source_url') else "")
        for hit in citations
    )


def _legacy_answer(draft, citations):
    """返回（有来源的回答，是否降级为摘录）。不把相似度当事实正确率。"""
    known = {hit["id"]: hit for hit in citations}
    cited = set(_CITATION.findall(draft))
    valid = bool(cited) and cited <= known.keys() and not re.search(r"https?://", draft)
    if valid:
        # 关键规则和数字只允许原文匹配；同义改写无法自动核实就展示原文。
        for sentence in _SENTENCE.split(draft):
            if not sentence.strip() or not _SENSITIVE.search(_CITATION.sub("", sentence)):
                continue
            normalized = _normalized(sentence)
            supported = {
                _normalized(part)
                for key in cited for part in _SENTENCE.split(known[key]["content"])
                if part.strip()
            }
            if normalized and normalized not in supported:
                valid = False
                break
    if valid:
        selected = [hit for hit in citations if hit["id"] in cited]
        return draft.strip() + references(selected), False
    selected = [
        hit for hit in citations
        if hit["id"] in cited
    ]
    message = (
        "已检索到相关资料，但当前回答未通过证据校验，暂时无法可靠确认。"
    )
    return message + references(citations) if selected else message, True


def _complete_quote(quote, content):
    # Match complete source sentences, retaining conditions/negation. A substring
    # such as "penalty of 2%" cannot strip "only during the lockup window".
    parts = [p.strip() for p in _SENTENCE.split(quote) if p.strip()]
    source = {p.strip() for p in _SENTENCE.split(content) if p.strip()}
    return bool(parts) and quote in content and all(p in source for p in parts)


def _verified_translation(text, quote):
    """Small deterministic equivalence, not a general semantic verifier.

    All other sensitive translations fall back to their verified source quote.
    """
    english = re.fullmatch(r"(?:The )?unlock period is ([0-9]+) days?\.?", quote.strip(), re.I)
    chinese = re.fullmatch(r"解锁期(?:为|是)([0-9]+)天", _normalized(text))
    return bool(english and chinese and english[1] == chinese[1])


def grounded_answer(draft, citations):
    """Validate structured per-claim evidence; retain legacy drafts for compatibility."""
    raw = draft.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    if not raw.startswith("{"):
        return _legacy_answer(draft, citations)
    failure = "已检索到相关资料，但当前回答未通过证据校验，暂时无法可靠确认。"
    try:
        data = json.loads(raw)
        claims = data["claims"]
        if not isinstance(claims, list) or not 1 <= len(claims) <= 12:
            raise ValueError("invalid claims")
        known = {hit["id"]: hit for hit in citations}
        rendered, selected, degraded = [], {}, False
        for claim in claims:
            text, key, quote = claim["text"], claim["source_id"], claim["quote"]
            if not all(isinstance(v, str) and v.strip() for v in (text, key, quote)):
                raise ValueError("invalid evidence")
            if len(text) > 2000 or len(quote) > 2000 or re.search(r"https?://", text) or _CITATION.search(text):
                raise ValueError("invalid claim")
            hit = known[key]
            if not _complete_quote(quote, hit["content"]):
                raise ValueError("quote is not a complete source sentence")
            exact = _normalized(text) in {_normalized(p) for p in _SENTENCE.split(quote) if p.strip()}
            sensitive = bool(_SENSITIVE.search(text) or _CN_NUM.search(text))
            if sensitive and not exact and not _verified_translation(text, quote):
                # Do not publish an unverified translated number/condition as fact.
                rendered.append(f"该条结论暂时无法自动核实，相关文档原文：\n> {quote.replace(chr(10), chr(10)+'> ')} [{key}]")
                degraded = True
            else:
                rendered.append(f"{text.strip()} [{key}]")
                if not exact:
                    rendered.append(f"文档原文：\n> {quote.replace(chr(10), chr(10)+'> ')}")
            selected[key] = hit
        return "\n\n".join(rendered) + references(list(selected.values())), degraded
    except (ValueError, TypeError, KeyError):
        return failure, True
