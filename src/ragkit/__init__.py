"""A small, measurable RAG pipeline: chunking, embeddings, FAISS, hybrid retrieval, metrics."""

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
    "Chunk", "chunk_corpus", "chunk_document",
    "Embedder", "HashingEmbedder", "SentenceTransformerEmbedder", "l2_normalize",
    "Query", "RetrievalReport", "evaluate", "ndcg_at_k", "precision_at_k",
    "recall_at_k", "reciprocal_rank", "render_comparison",
    "reciprocal_rank_fusion", "weighted_score_fusion",
    "FAISS_AVAILABLE", "Hit", "VectorIndex", "BM25",
    "Passage", "RagPipeline", "__version__",
]
