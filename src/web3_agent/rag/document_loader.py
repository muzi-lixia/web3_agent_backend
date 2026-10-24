"""Local/offline parsing. Source URLs are provenance only and are never fetched."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


class DocumentParseError(ValueError):
    """Only fixed, safe messages are exposed in job errors."""


# 数据职责：归一化后的文档正文及 PDF 页码范围。
@dataclass(frozen=True)
class ParsedDocument:
    # text：归一化后的完整文档正文，切片字符偏移均相对于它。
    text: str
    # (start, end, 1-based PDF page); offsets refer to normalized text.
    # page_spans：PDF 每页在归一化正文中的 (起始偏移, 结束偏移, 从 1 开始页码) 元组。
    page_spans: tuple[tuple[int, int, int], ...] = ()


def _clean(text):
    """清除空字符、统一换行并去除首尾空白。

    Args:
        text: 待检查、解析或处理的原始文本。
    """
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_document_with_spans(data: bytes, filename: str, *, max_bytes=5_000_000, max_text_chars=1_000_000) -> ParsedDocument:
    # 读取或解析前限制文件字节数，避免空文件和超大输入进入解析器。
    """安全解析 UTF-8 文本或可提取文字的 PDF，返回正文及页码区间。

    Args:
        data: 待解析或持久化的原始上传文件字节。
        filename: 安全文件名，用于选择格式解析器。
        max_bytes: 文件允许的最大字节数。
        max_text_chars: 解析后正文允许的最大字符数。

    Raises:
        DocumentParseError: 文件格式、大小、编码或可提取文本不符合限制。
    """
    if not data or len(data) > max_bytes:
        raise DocumentParseError("文档为空或超过大小限制")
    suffix = Path(filename).suffix.lower()
    spans = []
    try:
        # 按安全文件名后缀选择解析器；文本仅接受 UTF-8，PDF 另做文件头和资源上限校验。
        if suffix in {".md", ".txt"}:
            content = data.decode("utf-8-sig")
        elif suffix == ".pdf":
            if not data.startswith(b"%PDF-"):
                raise DocumentParseError("PDF 文件头不正确")
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise DocumentParseError("不支持加密 PDF，请上传未加密版本")
            if len(reader.pages) > 200:
                raise DocumentParseError("PDF 不能超过 200 页")
            parts, length = [], 0
            for page_number,page in enumerate(reader.pages,1):
                stream = page.get_contents()
                if stream is not None and len(stream.get_data()) > 10_000_000:
                    raise DocumentParseError("PDF 页面内容过大，请拆分文档")
                part = _clean(page.extract_text() or "")
                start = length + (2 if parts else 0)
                length = start + len(part)
                # 记录清洗后的页文字在最终正文中的偏移，供切片精确回溯页码。
                spans.append((start,length,page_number))
                if length > max_text_chars:
                    raise DocumentParseError("解析文本超过长度限制")
                parts.append(part)
            content = "\n\n".join(parts)
        else:
            raise DocumentParseError("只支持 .md / .txt / 可提取文本的 .pdf")
    except DocumentParseError:
        raise
    except UnicodeError as exc:
        raise DocumentParseError("文本必须采用 UTF-8 编码") from exc
    except Exception as exc:
        raise DocumentParseError("文档无法解析，请检查文件或转换为 Markdown/TXT") from exc
    content = content if spans else _clean(content)
    if not content.strip():
        raise DocumentParseError("文档没有可提取文本；扫描 PDF 请先完成 OCR")
    if len(content) > max_text_chars:
        raise DocumentParseError("解析文本超过长度限制")
    return ParsedDocument(content,tuple(spans))


def parse_document(data: bytes, filename: str, **kwargs) -> str:
    """解析文件字节并仅返回归一化正文，不返回页码信息。

    Args:
        data: 待解析或持久化的原始上传文件字节。
        filename: 安全文件名，用于选择格式解析器。
        **kwargs: 透传给底层调用的其余关键字参数。
    """
    return parse_document_with_spans(data,filename,**kwargs).text


def load_document(path: Path, *, max_bytes: int = 5_000_000, max_text_chars: int = 1_000_000) -> str:
    """先检查本地文件大小，再读取并解析为正文。

    Args:
        path: 本地文档路径。
        max_bytes: 文件允许的最大字节数。
        max_text_chars: 解析后正文允许的最大字符数。
    """
    if path.stat().st_size > max_bytes:
        raise DocumentParseError("文档超过大小限制")
    return parse_document(path.read_bytes(), path.name, max_bytes=max_bytes, max_text_chars=max_text_chars)


def load_parsed_document(path: Path, *, max_bytes=5_000_000, max_text_chars=1_000_000):
    """读取本地文件并保留解析后的正文及 PDF 页码区间。

    Args:
        path: 本地文档路径。
        max_bytes: 文件允许的最大字节数。
        max_text_chars: 解析后正文允许的最大字符数。
    """
    if path.stat().st_size > max_bytes:
        raise DocumentParseError("文档超过大小限制")
    return parse_document_with_spans(path.read_bytes(),path.name,max_bytes=max_bytes,max_text_chars=max_text_chars)
