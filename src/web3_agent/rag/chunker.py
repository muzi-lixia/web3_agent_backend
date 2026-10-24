"""Heading-aware overlapping chunks with normalized-text offsets and PDF page provenance."""
from dataclasses import dataclass, field
import re

# 切块逻辑变了就改这个版本号：它进内容指纹，决定已发布文档是否需要重建索引。
CHUNKER_VERSION = "heading_overlap_v1"


@dataclass(frozen=True)
class Chunk:
    section: str
    content: str
    token_count: int = 0
    token_count_method: str = "approx_v1"
    char_start: int | None = None
    char_end: int | None = None
    page_number: int | None = None
    page_end: int | None = None
    metadata: dict = field(default_factory=dict)


def _units(text):
    # Explicit approximation, not a claim about a model's exact tokenizer.
    for match in re.finditer(r"[A-Za-z0-9_]+|[^\s]",text):
        for offset in range(0,len(match.group()),4):
            start=match.start()+offset
            yield start,min(start+4,match.end())


def split_document(text: str, *, chunk_tokens=400, overlap_ratio=.1, page_spans=()) -> list[Chunk]:
    if chunk_tokens < 16 or not 0 <= overlap_ratio < .5:
        raise ValueError("无效切块参数")
    sections=[]
    chapter = "正文"
    section,body_start,position=chapter,0,0
    for line in text.splitlines(keepends=True):
        heading=re.match(r"^#{1,6}\s+(.+?)\s*$",line)

        # 识别当前 FAQ 的格式：
        question = re.fullmatch(
            r"\s*\*\*(.+[?？])\*\*\s*",
            line,
        )

        if heading or question:
            if text[body_start:position].strip():
                sections.append((section, body_start, position))
            if heading:
                chapter = heading.group(1)[:200]
                section = chapter
                body_start = position+len(line)
            else:
                section = f"{chapter} / {question.group(1)}"[:200]
                # 保留问题本身，保证每一个块同时包含问题和答案。
                body_start = position
        position+=len(line)
    if text[body_start:].strip():
        sections.append((section, body_start, len(text)))
    chunks=[]
    overlap=int(chunk_tokens*overlap_ratio)
    for section,body_start,body_end in sections:
        body=text[body_start:body_end]
        units=list(_units(body))
        start=0
        while start < len(units):
            end=min(start+chunk_tokens,len(units))
            char_start,char_end=body_start+units[start][0],body_start+units[end-1][1]
            pages=[number for a,b,number in page_spans if a<char_end and b>char_start and b>a]
            chunks.append(Chunk(section,text[char_start:char_end],token_count=end-start,
                char_start=char_start,char_end=char_end,page_number=min(pages) if pages else None,
                page_end=max(pages) if pages else None))
            if end==len(units):
                break
            start=end-overlap
    return chunks
