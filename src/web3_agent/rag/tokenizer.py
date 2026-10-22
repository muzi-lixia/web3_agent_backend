"""Deterministic mixed-language lexical analysis; source text is never modified."""
import re
import unicodedata

VERSION = "unicode_bigram_v1"

def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold()

def tokenize(text):
    tokens = []
    for match in re.finditer(r"[a-z0-9]+(?:[_.:-][a-z0-9]+)*|[\u3400-\u9fff]+", normalize(text)):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(word)
            tokens.extend(word[i:i+2] for i in range(len(word)-1))
        else:
            tokens.append(word)
    return tokens
