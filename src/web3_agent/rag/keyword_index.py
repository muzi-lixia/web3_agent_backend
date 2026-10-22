"""Dependency-free BM25 over a bounded, scoped PostgreSQL snapshot."""
import math
from collections import Counter
from web3_agent.rag.tokenizer import tokenize

class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.terms = [Counter(tokenize(f"{c.title} {c.section} {c.content}")) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / max(len(chunks), 1) or 1
        self.df = Counter(term for terms in self.terms for term in terms)

    def search(self, query, limit=20):
        results = []
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths):
            score = 0.0
            for term in set(tokenize(query)):
                frequency = terms[term]
                if frequency:
                    idf = math.log(1 + (len(self.chunks)-self.df[term]+.5)/(self.df[term]+.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5*(.25+.75*length/self.average))
            if score > 0:
                results.append((chunk, score))
        return sorted(results, key=lambda item: (-item[1], item[0].chunk_id))[:limit]
