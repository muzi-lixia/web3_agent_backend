"""Dependency-free BM25 over a bounded, scoped PostgreSQL snapshot."""
import math
from array import array
from collections import Counter

from web3_agent.rag.tokenizer import tokenize


class BM25Index:
    """构建一次、只读复用：查询只扫命中词项的切片，不遍历整个知识库。

    倒排里同时存词频，逐切片的 Counter 构建完就可以丢掉 —— 否则同一份词频
    在内存里存两份，知识库越大越占地方。
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.lengths = []
        self.postings = {}
        for index, chunk in enumerate(chunks):
            terms = Counter(tokenize(f"{chunk.title} {chunk.section} {chunk.content}"))
            self.lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                posting = self.postings.get(term)
                if posting is None:
                    posting = self.postings[term] = (array("I"), array("I"))
                posting[0].append(index)
                posting[1].append(frequency)
        self.average = sum(self.lengths) / max(len(chunks), 1) or 1

    def search(self, query, limit=20):
        # 查询词项只解析一次；不在索引里的词项直接丢掉，不参与打分。
        query_terms = [term for term in set(tokenize(query)) if term in self.postings]
        if not query_terms:
            return []
        total = len(self.chunks)
        scores = {}
        for term in query_terms:
            indexes, frequencies = self.postings[term]
            idf = math.log(1 + (total - len(indexes) + .5) / (len(indexes) + .5))
            for position, index in enumerate(indexes):
                frequency = frequencies[position]
                scores[index] = scores.get(index, 0.0) + idf * frequency * 2.5 / (
                    frequency + 1.5 * (.25 + .75 * self.lengths[index] / self.average))
        return sorted(((self.chunks[index], score) for index, score in scores.items()),
                      key=lambda item: (-item[1], item[0].chunk_id))[:limit]
