"""The retrieval pipeline.

Ties chunking, embedding, indexing and fusion into one object with three
retrieval modes so they can be compared on the same corpus rather than argued
about in the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from ragkit.chunking import Chunk, chunk_corpus
from ragkit.embeddings import Embedder, HashingEmbedder
from ragkit.fusion import reciprocal_rank_fusion, weighted_score_fusion
from ragkit.index import Hit, VectorIndex
from ragkit.lexical import BM25

Mode = Literal["dense", "lexical", "hybrid", "hybrid_weighted"]


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk with its score, ready to put in a prompt."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def citation(self) -> str:
        """A traceable reference: document and character span."""
        return f"{self.chunk.doc_id}:{self.chunk.start}-{self.chunk.end}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "citation": self.citation,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "text": self.chunk.text,
        }


class RagPipeline:
    """Chunk, embed, index, and retrieve."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        use_faiss: bool = True,
        max_chars: int = 800,
        overlap: int = 120,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.max_chars = max_chars
        self.overlap = overlap
        self.chunks: dict[str, Chunk] = {}
        self.index = VectorIndex(self.embedder.dim, use_faiss=use_faiss)
        self.bm25 = BM25()

    @property
    def backend(self) -> str:
        return self.index.backend

    def __len__(self) -> int:
        return len(self.chunks)

    def add_documents(self, documents: Iterable[tuple[str, str]]) -> list[Chunk]:
        """Chunk, embed and index `(doc_id, text)` pairs."""
        chunks = chunk_corpus(documents, max_chars=self.max_chars, overlap=self.overlap)
        if not chunks:
            return []
        return self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        keys = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts)
        self.index.add(keys, vectors)
        self.bm25.add(keys, texts)
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        return list(chunks)

    def _dense(self, query: str, k: int) -> list[Hit]:
        return self.index.search(self.embedder.encode([query])[0], k=k)

    def _lexical(self, query: str, k: int) -> list[Hit]:
        return self.bm25.search(query, k=k)

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        """Retrieve top-k chunk keys in the requested mode.

        Hybrid modes over-fetch before fusing: an item ranked 8th by one
        retriever and 2nd by the other should be able to win, and it cannot if
        both lists were already truncated to k.
        """
        if mode == "dense":
            return self._dense(query, k)
        if mode == "lexical":
            return self._lexical(query, k)

        fetch = max(k * 4, 20)
        dense, lexical = self._dense(query, fetch), self._lexical(query, fetch)
        if mode == "hybrid":
            return reciprocal_rank_fusion([dense, lexical], k=k)
        if mode == "hybrid_weighted":
            return weighted_score_fusion(dense, lexical, k=k)
        raise ValueError(f"unknown mode {mode!r}")

    def retrieve(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Passage]:
        """Retrieve passages with their text and citations."""
        return [
            Passage(chunk=self.chunks[hit.key], score=hit.score, rank=hit.rank)
            for hit in self.search(query, k=k, mode=mode)
            if hit.key in self.chunks
        ]

    def build_context(
        self, query: str, k: int = 5, mode: Mode = "hybrid", *, max_chars: int = 4000
    ) -> tuple[str, list[Passage]]:
        """Assemble a cited context block under a character budget.

        Returns the block and the passages that fit, so a caller can always
        report exactly what the model was shown. Passages are numbered and
        labelled with their source span: a generated answer that cites [2] can
        be traced to a document and offset, which is the difference between a
        RAG system and a plausible-sounding one.
        """
        passages = self.retrieve(query, k=k, mode=mode)
        parts: list[str] = []
        used: list[Passage] = []
        total = 0

        for position, passage in enumerate(passages, start=1):
            block = f"[{position}] ({passage.citation})\n{passage.chunk.text}"
            if total + len(block) > max_chars and used:
                break
            parts.append(block)
            used.append(passage)
            total += len(block) + 2

        return "\n\n".join(parts), used
