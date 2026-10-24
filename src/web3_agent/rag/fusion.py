"""Rank fusion: raw cosine distance and BM25 scores are not comparable."""
def fuse(rankings, *, limit=15, k=60):
    # 返回值：(融合后的候选列表, 切片 ID 到 RRF 分数的映射)。
    """使用 RRF 按各路名次融合候选，去重并返回排序切片及融合分数。

    Args:
        rankings: 多路召回各自按相关性排序的候选列表。
        limit: 本次查询或展示返回条数上限。
        k: RRF 平滑常量，用于控制靠前名次的权重差。
    """
    scores, chunks = {}, {}
    # 每一路单独防重复；同一切片可从多路获得名次加权，但一路内只计一次。
    for ranking in rankings:
        seen = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1/(k+rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [chunks[key] for key in ordered], {key: scores[key] for key in ordered}
