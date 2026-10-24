"""Deterministic mixed-language lexical analysis; source text is never modified."""
import re
import unicodedata

# VERSION：关键词分词算法版本标识。
VERSION = "unicode_bigram_v1"

def normalize(text):
    """对检索文本执行 Unicode NFKC 和大小写折叠，保持词项匹配一致。

    Args:
        text: 待检查、解析或处理的原始文本。
    """
    return unicodedata.normalize("NFKC", text).casefold()

def tokenize(text):
    """对中英文混合文本分词，中文使用单字与相邻双字，英文保留标识符。

    Args:
        text: 待检查、解析或处理的原始文本。
    """
    tokens = []
    for match in re.finditer(r"[a-z0-9]+(?:[_.:-][a-z0-9]+)*|[\u3400-\u9fff]+", normalize(text)):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(word)
            tokens.extend(word[i:i+2] for i in range(len(word)-1))
        else:
            tokens.append(word)
    return tokens
