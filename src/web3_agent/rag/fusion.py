"""Rank fusion: raw cosine distance and BM25 scores are not comparable."""
# 向量是语义匹配，关键词只补字面重合。等权时，一个碰巧含查询词面的块就能拿到关键词路
# 第一名，与向量第一名同分，把语义命中顶到后面 —— 所以关键词这一路的分量要压低。
# DENSE_WEIGHT：融合时向量路的名次权重。
DENSE_WEIGHT = 1.0
# KEYWORD_WEIGHT：融合时关键词路的名次权重。
KEYWORD_WEIGHT = 0.6


def fuse(rankings, *, limit=15, k=60, weights=None):
    # 返回值：(融合后的候选列表, 切片 ID 到 RRF 分数的映射)。
    """使用加权 RRF 按各路名次融合候选，去重并返回排序切片及融合分数。

    Args:
        rankings: 多路召回各自按相关性排序的候选列表。
        limit: 本次查询或展示返回条数上限。
        k: RRF 平滑常量，用于控制靠前名次的权重差。
        weights: 与 rankings 等长的各路名次权重，省略时各路等权。
    """
    weights = list(weights or [1.0] * len(rankings))
    if len(weights) != len(rankings):
        raise ValueError("weights 必须与 rankings 等长")
    scores, chunks = {}, {}
    # 每一路单独防重复；同一切片可从多路获得名次加权，但一路内只计一次。
    for weight, ranking in zip(weights, rankings):
        seen = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + weight / (k + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [chunks[key] for key in ordered], {key: scores[key] for key in ordered}
