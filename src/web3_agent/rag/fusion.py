"""Rank fusion: raw cosine distance and BM25 scores are not comparable."""
def fuse(rankings, *, limit=15, k=60):
    scores, chunks = {}, {}
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
