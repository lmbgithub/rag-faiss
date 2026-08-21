"""Document chunking.

Chunking is the retrieval decision that gets the least attention and causes the
most damage. Too large and the embedding averages away the specific fact you
need; too small and the fact loses the context that makes it interpretable. The
overlap exists so a fact sitting on a boundary is not cut in half.

Chunks carry their source and character offsets, so any retrieved answer can be
traced back to an exact span of an exact document. A RAG system that cannot cite
its own source is not auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a document."""

    text: str
    doc_id: str
    chunk_id: str
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def chunk_document(
    text: str,
    doc_id: str,
    *,
    max_chars: int = 800,
    overlap: int = 120,
    respect_paragraphs: bool = True,
) -> list[Chunk]:
    """Split a document into overlapping chunks on natural boundaries.

    Splits at paragraph breaks first, then sentence boundaries, and only falls
    back to a hard character cut when a single sentence exceeds the budget.
    Cutting mid-sentence is a last resort because it reliably produces chunks
    that embed poorly: half a clause has no clear meaning to average over.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    text = text or ""
    if not text.strip():
        return []

    units = _split_units(text, respect_paragraphs)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    cursor = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, cursor
        if not buffer:
            return
        body = " ".join(buffer).strip()
        if body:
            start = text.find(buffer[0], cursor) if buffer[0] in text[cursor:] else cursor
            start = start if start >= 0 else cursor
            chunks.append(
                Chunk(
                    text=body,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::{len(chunks):04d}",
                    start=start,
                    end=start + len(body),
                )
            )
            cursor = max(cursor, start + max(1, len(body) - overlap))
        buffer, buffer_len = [], 0

    for unit in units:
        if len(unit) > max_chars:
            flush()
            for piece in _hard_split(unit, max_chars, overlap):
                start = text.find(piece[:40], cursor)
                start = start if start >= 0 else cursor
                chunks.append(
                    Chunk(
                        text=piece,
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}::{len(chunks):04d}",
                        start=start,
                        end=start + len(piece),
                    )
                )
                cursor = max(cursor, start + max(1, len(piece) - overlap))
            continue

        if buffer_len + len(unit) + 1 > max_chars and buffer:
            tail = _overlap_tail(buffer, overlap)
            flush()
            buffer = list(tail)
            buffer_len = sum(len(t) + 1 for t in buffer)

        buffer.append(unit)
        buffer_len += len(unit) + 1

    flush()
    return chunks


def _split_units(text: str, respect_paragraphs: bool) -> list[str]:
    if respect_paragraphs:
        units: list[str] = []
        for para in _PARAGRAPH.split(text):
            para = para.strip()
            if para:
                units.extend(split_sentences(para) or [para])
        return units
    return split_sentences(text) or [text.strip()]


def _overlap_tail(buffer: Sequence[str], overlap: int) -> list[str]:
    """Keep trailing units up to `overlap` characters, so context carries over."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(buffer):
        if total + len(unit) > overlap and tail:
            break
        tail.insert(0, unit)
        total += len(unit) + 1
    return tail


def _hard_split(unit: str, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    return [unit[i : i + max_chars].strip() for i in range(0, len(unit), step) if unit[i : i + max_chars].strip()]


def chunk_corpus(
    documents: Iterable[tuple[str, str]], **kwargs: Any
) -> list[Chunk]:
    """Chunk `(doc_id, text)` pairs into one flat list."""
    out: list[Chunk] = []
    for doc_id, text in documents:
        out.extend(chunk_document(text, doc_id, **kwargs))
    return out
