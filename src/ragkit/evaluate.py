"""Retrieval metrics.

The reason this module exists at all: most RAG systems are evaluated by reading
a handful of generated answers and deciding they look reasonable. That measures
the generator, not the retriever, and it cannot distinguish "the model wrote a
plausible answer" from "the right chunk was actually retrieved".

If the correct chunk is not in the context window, no prompt can fix it. Recall
is the ceiling on everything downstream, so it is measured first and separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ragkit.index import Hit


@dataclass(frozen=True)
class Query:
    """An evaluation query with its known-relevant chunk keys."""

    query_id: str
    text: str
    relevant: frozenset[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Query":
        relevant = data.get("relevant") or data.get("relevant_ids") or []
        if isinstance(relevant, str):
            relevant = [relevant]
        return cls(
            query_id=str(data.get("id", data.get("query_id", ""))),
            text=str(data["query"] if "query" in data else data.get("text", "")),
            relevant=frozenset(str(r) for r in relevant),
        )


def recall_at_k(hits: Sequence[Hit], relevant: frozenset[str], k: int) -> float:
    """Share of relevant items appearing in the top k."""
    if not relevant:
        return 0.0
    retrieved = {h.key for h in hits[:k]}
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(hits: Sequence[Hit], relevant: frozenset[str], k: int) -> float:
    if k < 1 or not hits:
        return 0.0
    retrieved = [h.key for h in hits[:k]]
    return sum(1 for key in retrieved if key in relevant) / min(k, len(retrieved))


def reciprocal_rank(hits: Sequence[Hit], relevant: frozenset[str]) -> float:
    """1 / rank of the first relevant hit; 0 when none is retrieved.

    The metric that matters when the context window fits only a few chunks: it
    cares where the first good answer landed, not how many good answers exist.
    """
    for hit in hits:
        if hit.key in relevant:
            return 1.0 / hit.rank
    return 0.0


def ndcg_at_k(hits: Sequence[Hit], relevant: frozenset[str], k: int) -> float:
    """Binary-relevance nDCG with log2 discount."""
    import math

    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(hit.rank + 1) for hit in hits[:k] if hit.key in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate metrics for one retriever over a query set."""

    name: str
    queries: int
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float
    ndcg: dict[int, float]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "queries": self.queries,
            "recall": {f"@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "precision": {f"@{k}": round(v, 4) for k, v in sorted(self.precision.items())},
            "mrr": round(self.mrr, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in sorted(self.ndcg.items())},
            "zero_recall_queries": list(self.failures),
        }


def evaluate(
    name: str,
    retrieve: Callable[[str, int], Sequence[Hit]],
    queries: Sequence[Query],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalReport:
    """Score a retriever over a query set.

    `retrieve` is called once per query at the largest k, and the smaller cutoffs
    are computed by truncation. Calling it once per (query, k) would multiply the
    cost for identical results.
    """
    if not ks:
        raise ValueError("at least one k is required")
    max_k = max(ks)

    recall = {k: 0.0 for k in ks}
    precision = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr_total = 0.0
    failures: list[str] = []

    for query in queries:
        hits = list(retrieve(query.text, max_k))
        for k in ks:
            recall[k] += recall_at_k(hits, query.relevant, k)
            precision[k] += precision_at_k(hits, query.relevant, k)
            ndcg[k] += ndcg_at_k(hits, query.relevant, k)
        mrr_total += reciprocal_rank(hits, query.relevant)
        if recall_at_k(hits, query.relevant, max_k) == 0.0:
            failures.append(query.query_id)

    n = len(queries) or 1
    return RetrievalReport(
        name=name,
        queries=len(queries),
        recall={k: v / n for k, v in recall.items()},
        precision={k: v / n for k, v in precision.items()},
        mrr=mrr_total / n,
        ndcg={k: v / n for k, v in ndcg.items()},
        failures=tuple(failures),
    )


def render_comparison(reports: Sequence[RetrievalReport], *, ks: Sequence[int] = (1, 5)) -> str:
    """Render several retrievers side by side."""
    if not reports:
        return "no retrievers evaluated"

    columns = [f"R@{k}" for k in ks] + ["MRR", f"nDCG@{max(ks)}"]
    header = f"{'retriever':<22}" + "".join(f"{c:>10}" for c in columns)
    lines = ["=" * len(header), "RETRIEVAL COMPARISON", "=" * len(header), header, "-" * len(header)]

    for report in reports:
        row = f"{report.name:<22}"
        for k in ks:
            row += f"{report.recall[k]:>9.1%} "
        row = row.rstrip() + f"{report.mrr:>10.3f}" + f"{report.ndcg[max(ks)]:>10.3f}"
        lines.append(row)

    best = max(reports, key=lambda r: (r.recall[max(ks)], r.mrr))
    lines.append("-" * len(header))
    lines.append(f"best R@{max(ks)}: {best.name}")
    if best.failures:
        lines.append(f"still zero-recall on {len(best.failures)} query/queries: {', '.join(best.failures[:5])}")
    return "\n".join(lines)
