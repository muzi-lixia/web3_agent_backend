"""Local/offline parsing. Source URLs are provenance only and are never fetched."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from web3_agent.agent.secret_detector import detect


class DocumentParseError(ValueError):
    """Only fixed, safe messages are exposed in job errors."""


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    # (start, end, 1-based PDF page); offsets refer to normalized text.
    page_spans: tuple[tuple[int, int, int], ...] = ()


def _clean(text):
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_document_with_spans(data: bytes, filename: str, *, max_bytes=5_000_000, max_text_chars=1_000_000) -> ParsedDocument:
    if not data or len(data) > max_bytes:
        raise DocumentParseError("文档为空或超过大小限制")
    suffix = Path(filename).suffix.lower()
    spans = []
    try:
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
    if detect(content) is not None:
        raise DocumentParseError("文档疑似含有私钥或助记词，拒绝入库")
    return ParsedDocument(content,tuple(spans))


def parse_document(data: bytes, filename: str, **kwargs) -> str:
    return parse_document_with_spans(data,filename,**kwargs).text


def load_document(path: Path, *, max_bytes: int = 5_000_000, max_text_chars: int = 1_000_000) -> str:
    if path.stat().st_size > max_bytes:
        raise DocumentParseError("文档超过大小限制")
    return parse_document(path.read_bytes(), path.name, max_bytes=max_bytes, max_text_chars=max_text_chars)


def load_parsed_document(path: Path, *, max_bytes=5_000_000, max_text_chars=1_000_000):
    if path.stat().st_size > max_bytes:
        raise DocumentParseError("文档超过大小限制")
    return parse_document_with_spans(path.read_bytes(),path.name,max_bytes=max_bytes,max_text_chars=max_text_chars)
