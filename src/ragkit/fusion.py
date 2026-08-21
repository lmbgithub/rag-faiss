"""Combining dense and lexical rankings.

Score-level fusion is the obvious approach and the wrong one: BM25 scores are
unbounded and corpus-dependent while cosine similarity sits in [-1, 1], so any
weighted sum silently becomes "whatever BM25 said" on some corpora and
"whatever the embedder said" on others, with no warning either way.

Reciprocal Rank Fusion combines *ranks*, which are directly comparable across
scoring systems by construction. That is why it needs no per-corpus tuning and
no score normalization to work.
"""

from __future__ import annotations

from typing import Sequence

from ragkit.index import Hit

RRF_K = 60  # standard smoothing constant; damps the top-rank advantage


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], *, k: int = 5, rrf_k: int = RRF_K
) -> list[Hit]:
    """Fuse ranked lists by summing 1 / (rrf_k + rank).

    An item ranked highly by both retrievers beats an item ranked first by one
    and missing from the other, which is the behaviour hybrid retrieval is for.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for hit in ranking:
            fused[hit.key] = fused.get(hit.key, 0.0) + 1.0 / (rrf_k + hit.rank)
            first_seen.setdefault(hit.key, hit.rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]], kv[0]))
    return [
        Hit(key=key, score=score, rank=rank)
        for rank, (key, score) in enumerate(ordered[:k], start=1)
    ]


def weighted_score_fusion(
    dense: Sequence[Hit],
    lexical: Sequence[Hit],
    *,
    alpha: float = 0.5,
    k: int = 5,
) -> list[Hit]:
    """Min-max normalize each ranking, then blend with weight `alpha` on dense.

    Provided for comparison against RRF. It works when both score distributions
    are well behaved and degrades unpredictably when they are not — the ablation
    in `evaluate.py` is there to show which one wins on your corpus rather than
    asking you to trust a default.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    def normalize(hits: Sequence[Hit]) -> dict[str, float]:
        if not hits:
            return {}
        scores = [h.score for h in hits]
        low, high = min(scores), max(scores)
        span = high - low
        if span <= 0:
            return {h.key: 1.0 for h in hits}
        return {h.key: (h.score - low) / span for h in hits}

    dense_norm, lexical_norm = normalize(dense), normalize(lexical)
    combined: dict[str, float] = {}
    for key in set(dense_norm) | set(lexical_norm):
        combined[key] = alpha * dense_norm.get(key, 0.0) + (1 - alpha) * lexical_norm.get(key, 0.0)

    ordered = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Hit(key=key, score=score, rank=rank) for rank, (key, score) in enumerate(ordered[:k], start=1)]
