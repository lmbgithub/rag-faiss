"""BM25 lexical retrieval.

Dense retrieval fails in a specific, predictable way: it is bad at exact tokens
it never saw in training. Product codes, drug names, error identifiers, version
numbers — the terms users are most likely to paste verbatim are the ones an
embedding averages into noise. BM25 is excellent at exactly those, which is why
hybrid retrieval beats either alone rather than being a hedge.

Implemented directly (Robertson/Sparck-Jones BM25) rather than pulled from a
dependency: it is thirty lines, and the parameters matter enough to be visible.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ragkit.embeddings import tokenize
from ragkit.index import Hit


class BM25:
    """Okapi BM25 over a fixed corpus.

    `k1` controls term-frequency saturation: a term appearing twenty times is
    barely more relevant than one appearing ten. `b` controls length
    normalization: without it, long documents win by accident.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._keys: list[str] = []
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, keys: Sequence[str], texts: Sequence[str]) -> None:
        if len(keys) != len(texts):
            raise ValueError("keys and texts must align")
        for key, text in zip(keys, texts):
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._keys.append(key)
            self._tf.append(counts)
            self._lengths.append(len(tokens))
            self._df.update(counts.keys())
        total = sum(self._lengths)
        self._avg_len = total / len(self._lengths) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +0.5 smoothing.

        The max(..., 0) floor matters: without it, a term appearing in more than
        half the corpus gets a negative weight and actively pushes documents
        that contain the query term *down* the ranking.
        """
        n = len(self._keys)
        df = self._df.get(term, 0)
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        if not self._keys:
            return []

        terms = tokenize(query)
        if not terms:
            return []

        scores = [0.0] * len(self._keys)
        for term in terms:
            if term not in self._df:
                continue
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, counts in enumerate(self._tf):
                freq = counts.get(term, 0)
                if not freq:
                    continue
                norm = 1.0 - self.b + self.b * (self._lengths[i] / self._avg_len if self._avg_len else 1.0)
                scores[i] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)

        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0), key=lambda p: (-p[0], p[1])
        )[:k]
        return [
            Hit(key=self._keys[i], score=float(s), rank=rank)
            for rank, (s, i) in enumerate(ranked, start=1)
        ]
