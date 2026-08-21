import numpy as np
import pytest

from ragkit.embeddings import HashingEmbedder, l2_normalize, tokenize
from ragkit.index import FAISS_AVAILABLE, VectorIndex


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]
    assert tokenize("") == []


def test_l2_normalize_produces_unit_rows():
    normed = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_l2_normalize_leaves_zero_rows_alone():
    # A zero row has no direction. Dividing by its norm would be a NaN factory.
    normed = l2_normalize(np.zeros((1, 4)))
    assert np.allclose(normed, 0.0)
    assert not np.isnan(normed).any()


def test_l2_normalize_accepts_a_single_vector():
    assert l2_normalize(np.array([3.0, 4.0])).shape == (1, 2)


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    assert np.allclose(e.encode(["chest pain"]), e.encode(["chest pain"]))


def test_hashing_embedder_shape_and_norm():
    e = HashingEmbedder(dim=64)
    vectors = e.encode(["one", "two", "three"])
    assert vectors.shape == (3, 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_hashing_embedder_ranks_lexical_overlap_higher():
    e = HashingEmbedder(dim=512)
    v = e.encode([
        "the patient reported chest pain",
        "the patient reported chest discomfort",
        "quarterly revenue exceeded forecast",
    ])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_hashing_embedder_rejects_bad_config():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)
    with pytest.raises(ValueError):
        HashingEmbedder(ngram=1)


@pytest.mark.parametrize("use_faiss", [True, False])
def test_index_search_returns_ranked_hits(use_faiss):
    e = HashingEmbedder(dim=128)
    texts = ["chest pain and shortness of breath", "quarterly revenue report", "hip fracture after a fall"]
    index = VectorIndex(128, use_faiss=use_faiss)
    index.add([f"c{i}" for i in range(3)], e.encode(texts))

    hits = index.search(e.encode(["chest pain"])[0], k=3)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].key == "c0"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_faiss_and_numpy_backends_agree():
    if not FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    e = HashingEmbedder(dim=128)
    texts = [f"document number {i} about topic {i % 3}" for i in range(20)]
    vectors = e.encode(texts)
    keys = [f"c{i}" for i in range(20)]

    a, b = VectorIndex(128, use_faiss=True), VectorIndex(128, use_faiss=False)
    a.add(keys, vectors)
    b.add(keys, vectors)
    query = e.encode(["topic 1"])[0]
    assert [h.key for h in a.search(query, k=5)] == [h.key for h in b.search(query, k=5)]


def test_empty_index_returns_nothing():
    assert VectorIndex(8).search(np.ones(8), k=3) == []


def test_index_rejects_misaligned_input():
    index = VectorIndex(4)
    with pytest.raises(ValueError, match="align"):
        index.add(["a"], np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        index.add(["a"], np.zeros((1, 8), dtype=np.float32))


def test_index_rejects_duplicate_keys():
    index = VectorIndex(4)
    index.add(["a"], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        index.add(["a"], np.ones((1, 4), dtype=np.float32))


def test_k_larger_than_corpus_is_clamped():
    e = HashingEmbedder(dim=32)
    index = VectorIndex(32)
    index.add(["a", "b"], e.encode(["one", "two"]))
    assert len(index.search(e.encode(["one"])[0], k=99)) == 2
