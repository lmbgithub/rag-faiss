"""Embedding backends.

Two implementations behind one interface, for the same reason the rest of these
projects carry a mock: the package must be runnable and testable on clone, with
no model download and no network.

`HashingEmbedder` is a deterministic bag-of-ngrams projection. It is a real
embedder — it produces stable vectors with meaningful cosine similarity for
lexical overlap — but it has no semantic understanding whatsoever. It exists as
a floor and a test fixture, never as a recommendation.

`SentenceTransformerEmbedder` is the one you would actually deploy.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.

    Everything downstream assumes unit vectors, which is what lets FAISS's inner
    product index return cosine similarity directly. Zero rows are left as zero
    rather than divided by zero; they simply never match anything.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class Embedder(Protocol):
    """Anything that turns text into unit-norm float32 vectors."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class HashingEmbedder:
    """Deterministic hashed character-ngram embedder. No model, no network.

    Uses signed hashing (a +1/-1 sign per feature) so unrelated features cancel
    on collision instead of always adding, which keeps collisions from inflating
    similarity between unrelated texts.
    """

    def __init__(self, dim: int = 256, ngram: int = 4) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        if ngram < 2:
            raise ValueError("ngram must be at least 2")
        self.dim = dim
        self.ngram = ngram
        self.name = f"hashing-{dim}d"

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        padded = f" {' '.join(tokens)} "
        feats.extend(padded[i : i + self.ngram] for i in range(max(0, len(padded) - self.ngram + 1)))
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                out[row, value % self.dim] += 1.0 if (value >> 63) & 1 else -1.0
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. The production path."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed; "
                'install the extra with: pip install "rag-faiss[embeddings]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        # The accessor was renamed across sentence-transformers versions; try the
        # current name first so the package works on both without a version pin.
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))
