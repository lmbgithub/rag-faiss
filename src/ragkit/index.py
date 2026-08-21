"""Vector index.

FAISS when it is installed, a NumPy brute-force index when it is not. Both
return identical results on the same vectors — the NumPy path is exact, and
`IndexFlatIP` is exact too, so this is a genuine fallback rather than a
degraded approximation.

The index stores unit vectors and searches by inner product, which for unit
vectors *is* cosine similarity. Doing the normalization once at ingest is
cheaper and less error-prone than dividing by norms on every query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ragkit.embeddings import l2_normalize

try:  # pragma: no cover - import-time branch
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None
    FAISS_AVAILABLE = False


@dataclass(frozen=True)
class Hit:
    """One retrieved item."""

    key: str
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "score": round(float(self.score), 6), "rank": self.rank}


class VectorIndex:
    """Exact inner-product index over unit vectors."""

    def __init__(self, dim: int, *, use_faiss: bool = True) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_faiss = bool(use_faiss and FAISS_AVAILABLE)
        self.backend = "faiss.IndexFlatIP" if self.use_faiss else "numpy.bruteforce"
        self._keys: list[str] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = faiss.IndexFlatIP(dim) if self.use_faiss else None

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        """Add vectors. Keys must be unique and aligned with rows."""
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[0] != len(keys):
            raise ValueError(
                f"got {len(keys)} keys for {vectors.shape[0]} vectors; they must align"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        duplicates = set(keys) & set(self._keys)
        if duplicates:
            raise ValueError(f"duplicate keys: {sorted(duplicates)[:5]}")

        self._keys.extend(keys)
        if self.use_faiss:
            self._faiss_index.add(vectors)
        else:
            self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        """Return the top-k most similar keys."""
        if k < 1:
            raise ValueError("k must be positive")
        if not self._keys:
            return []

        vector = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(k, len(self._keys))

        if self.use_faiss:
            scores, indices = self._faiss_index.search(vector, k)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ vector.T).ravel()
            top = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in top)

        return [
            Hit(key=self._keys[idx], score=float(score), rank=rank)
            for rank, (idx, score) in enumerate(pairs, start=1)
            if idx >= 0
        ]
