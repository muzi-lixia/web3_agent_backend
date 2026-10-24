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
        """一次性分词并构建切片长度、词频倒排和平均长度，供后续只读查询复用。

        Args:
            chunks: 本批文档切片，或任务的切片统计数量。
        """
        self.chunks = chunks
        # lengths：每个切片分词后的词项总数，索引与 chunks 保持一致。
        self.lengths = []
        # postings：词项倒排表：每项保存切片索引数组与对应词频数组。
        self.postings = {}
        # 对标题、章节和正文一次性分词，把词频写入倒排表供后续查询复用。
        for index, chunk in enumerate(chunks):
            terms = Counter(tokenize(f"{chunk.title} {chunk.section} {chunk.content}"))
            self.lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                posting = self.postings.get(term)
                if posting is None:
                    posting = self.postings[term] = (array("I"), array("I"))
                posting[0].append(index)
                posting[1].append(frequency)
        # average：索引内切片的平均词项长度，用于 BM25 长度归一化。
        self.average = sum(self.lengths) / max(len(chunks), 1) or 1

    def search(self, query, limit=20):
        # 查询词项只解析一次；不在索引里的词项直接丢掉，不参与打分。
        """仅对命中查询词项的切片累加 BM25 分数，按分数和切片 ID 稳定排序。

        Args:
            query: 当前检索问题文本。
            limit: 本次查询或展示返回条数上限。

        Returns:
            list[tuple]: 按相关性排序的 (切片, BM25 分数)。
        """
        query_terms = [term for term in set(tokenize(query)) if term in self.postings]
        if not query_terms:
            return []
        total = len(self.chunks)
        scores = {}
        # 只遍历查询命中的倒排项，按词频、逆文档频率和长度归一化累计 BM25 得分。
        for term in query_terms:
            indexes, frequencies = self.postings[term]
            idf = math.log(1 + (total - len(indexes) + .5) / (len(indexes) + .5))
            for position, index in enumerate(indexes):
                frequency = frequencies[position]
                scores[index] = scores.get(index, 0.0) + idf * frequency * 2.5 / (
                    frequency + 1.5 * (.25 + .75 * self.lengths[index] / self.average))
        return sorted(((self.chunks[index], score) for index, score in scores.items()),
                      key=lambda item: (-item[1], item[0].chunk_id))[:limit]
