"""A small, measurable RAG pipeline.

Chunking, embeddings, FAISS, hybrid retrieval and the metrics to judge them.
"""

from ragkit.chunking import Chunk, chunk_corpus, chunk_document
from ragkit.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    l2_normalize,
)
from ragkit.evaluate import (
    Query,
    RetrievalReport,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    render_comparison,
)
from ragkit.fusion import reciprocal_rank_fusion, weighted_score_fusion
from ragkit.index import FAISS_AVAILABLE, Hit, VectorIndex
from ragkit.lexical import BM25
from ragkit.pipeline import Passage, RagPipeline

__version__ = "0.1.0"

__all__ = [
    "BM25",
    "FAISS_AVAILABLE",
    "Chunk",
    "Embedder",
    "HashingEmbedder",
    "Hit",
    "Passage",
    "Query",
    "RagPipeline",
    "RetrievalReport",
    "SentenceTransformerEmbedder",
    "VectorIndex",
    "__version__",
    "chunk_corpus",
    "chunk_document",
    "evaluate",
    "l2_normalize",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_fusion",
    "render_comparison",
    "weighted_score_fusion",
]
