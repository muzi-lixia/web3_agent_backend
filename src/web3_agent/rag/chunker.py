"""Heading-aware overlapping chunks with normalized-text offsets and PDF page provenance."""
from dataclasses import dataclass, field
import re

# 切块逻辑变了就改这个版本号：它进内容指纹，决定已发布文档是否需要重建索引。
# CHUNKER_VERSION：文档切块算法版本，参与索引指纹计算。
CHUNKER_VERSION = "heading_overlap_v1"


# 数据职责：离线切块阶段的不可变结果，尚未绑定数据库文档和切片 ID。
@dataclass(frozen=True)
class Chunk:
    # section：切片所属章节或 FAQ 问题标题。
    section: str
    # content：该切片保留的归一化原文，不包括额外拼接的文档标题。
    content: str
    # token_count：切片的近似 token 数，用于估算切块大小，并非模型精确计费值。
    token_count: int = 0
    # token_count_method：近似 token 计数算法标识，便于解释计数口径。
    token_count_method: str = "approx_v1"
    # char_start：切片在归一化文档正文中的起始字符偏移，包含该位置。
    char_start: int | None = None
    # char_end：切片在归一化文档正文中的结束字符偏移，不包含该位置。
    char_end: int | None = None
    # page_number：切片起始 PDF 页码，从 1 开始；非 PDF 或无页码时为空。
    page_number: int | None = None
    # page_end：切片结束 PDF 页码，跨页时用于展示完整来源范围。
    page_end: int | None = None
    # metadata：解析或切块阶段附加的切片来源信息。
    metadata: dict = field(default_factory=dict)


def _units(text):
    # Explicit approximation, not a claim about a model's exact tokenizer.
    """生成用于近似 token 计数的字符范围，不修改源正文。

    Args:
        text: 待检查、解析或处理的原始文本。
    """
    for match in re.finditer(r"[A-Za-z0-9_]+|[^\s]",text):
        for offset in range(0,len(match.group()),4):
            start=match.start()+offset
            yield start,min(start+4,match.end())


def split_document(text: str, *, chunk_tokens=400, overlap_ratio=.1, page_spans=()) -> list[Chunk]:
    # 返回值：带章节、字符偏移及可选 PDF 页码的 Chunk 列表。
    """按章节和 FAQ 问题拆分正文，再按近似 token 大小生成重叠切片及来源偏移。

    Args:
        text: 待检查、解析或处理的原始文本。
        chunk_tokens: 每个切片的近似 token 数上限。
        overlap_ratio: 同章节相邻切片的重叠比例。
        page_spans: PDF 各页在归一化正文中的字符区间。
    """
    if chunk_tokens < 16 or not 0 <= overlap_ratio < .5:
        raise ValueError("无效切块参数")
    sections=[]
    chapter = "正文"
    section,body_start,position=chapter,0,0
    # 先按标题和 FAQ 问题建立章节边界，保留换行以正确累计正文字符偏移。
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
    # 将比例转换为近似 token 重叠数，后续同章节相邻切片保留上下文。
    overlap=int(chunk_tokens*overlap_ratio)
    for section,body_start,body_end in sections:
        body=text[body_start:body_end]
        units=list(_units(body))
        start=0
        while start < len(units):
            end=min(start+chunk_tokens,len(units))
            char_start,char_end=body_start+units[start][0],body_start+units[end-1][1]
            # 通过字符区间相交确定切片跨越的 PDF 页码，不把页间分隔符算作页面内容。
            pages=[number for a,b,number in page_spans if a<char_end and b>char_start and b>a]
            chunks.append(Chunk(section,text[char_start:char_end],token_count=end-start,
                char_start=char_start,char_end=char_end,page_number=min(pages) if pages else None,
                page_end=max(pages) if pages else None))
            if end==len(units):
                break
            start=end-overlap
    return chunks
