import pytest

from ragkit.evaluate import (
    Query, evaluate, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, render_comparison,
)
from ragkit.fusion import reciprocal_rank_fusion, weighted_score_fusion
from ragkit.index import Hit
from ragkit.lexical import BM25

CORPUS = [
    ("c0", "the patient reported chest pain radiating to the left arm"),
    ("c1", "quarterly revenue exceeded the forecast by twelve percent"),
    ("c2", "error code XR-4471 indicates a failed authentication handshake"),
    ("c3", "the patient has a history of hypertension and diabetes"),
]


def make_bm25():
    bm = BM25()
    bm.add([k for k, _ in CORPUS], [t for _, t in CORPUS])
    return bm


# -- BM25 ----------------------------------------------------------------


def test_bm25_finds_the_exact_term():
    hits = make_bm25().search("chest pain", k=2)
    assert hits[0].key == "c0"


def test_bm25_excels_at_rare_exact_tokens():
    # The case dense retrieval reliably loses: an identifier the embedder never
    # saw in training and cannot place in vector space.
    hits = make_bm25().search("XR-4471", k=1)
    assert hits[0].key == "c2"


def test_bm25_returns_nothing_for_unseen_terms():
    assert make_bm25().search("kangaroo", k=3) == []
    assert make_bm25().search("", k=3) == []


def test_bm25_empty_corpus():
    assert BM25().search("anything", k=3) == []


def test_bm25_ranks_are_sequential():
    hits = make_bm25().search("patient", k=5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_idf_floor_keeps_common_terms_non_negative():
    # "the" appears in most documents; without the floor its weight goes
    # negative and pushes matching documents down the ranking.
    bm = make_bm25()
    assert bm._idf("the") >= 0.0


def test_bm25_rejects_bad_params():
    with pytest.raises(ValueError):
        BM25(k1=-1)
    with pytest.raises(ValueError):
        BM25(b=2.0)
    with pytest.raises(ValueError):
        make_bm25().search("x", k=0)


def test_bm25_rejects_misaligned_input():
    with pytest.raises(ValueError):
        BM25().add(["a"], ["one", "two"])


# -- fusion --------------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers():
    dense = [Hit("a", 0.9, 1), Hit("b", 0.8, 2)]
    lexical = [Hit("b", 12.0, 1), Hit("c", 3.0, 2)]
    fused = reciprocal_rank_fusion([dense, lexical], k=3)
    # b is 2nd and 1st; a is 1st and absent. Agreement wins.
    assert fused[0].key == "b"
    assert [h.rank for h in fused] == [1, 2, 3]


def test_rrf_ignores_score_scale():
    # BM25 scores are unbounded, cosine sits in [-1,1]. RRF combines ranks, so
    # a huge lexical score cannot swamp the dense ranking.
    dense = [Hit("a", 0.99, 1)]
    lexical = [Hit("b", 9999.0, 1)]
    fused = reciprocal_rank_fusion([dense, lexical], k=2)
    assert {h.key for h in fused} == {"a", "b"}
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([[], []], k=3) == []


def test_rrf_rejects_bad_params():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], rrf_k=0)


def test_weighted_fusion_respects_alpha():
    dense = [Hit("a", 1.0, 1), Hit("b", 0.0, 2)]
    lexical = [Hit("b", 10.0, 1), Hit("a", 0.0, 2)]
    assert weighted_score_fusion(dense, lexical, alpha=1.0, k=2)[0].key == "a"
    assert weighted_score_fusion(dense, lexical, alpha=0.0, k=2)[0].key == "b"


def test_weighted_fusion_handles_a_flat_ranking():
    flat = [Hit("a", 5.0, 1), Hit("b", 5.0, 2)]
    assert len(weighted_score_fusion(flat, [], alpha=0.5, k=2)) == 2


def test_weighted_fusion_rejects_bad_alpha():
    with pytest.raises(ValueError):
        weighted_score_fusion([], [], alpha=1.5)


# -- metrics -------------------------------------------------------------


HITS = [Hit("a", 0.9, 1), Hit("b", 0.8, 2), Hit("c", 0.7, 3)]


def test_recall_at_k():
    assert recall_at_k(HITS, frozenset({"a", "z"}), 3) == 0.5
    assert recall_at_k(HITS, frozenset({"a"}), 1) == 1.0
    assert recall_at_k(HITS, frozenset({"c"}), 2) == 0.0


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k(HITS, frozenset(), 3) == 0.0


def test_precision_at_k():
    assert precision_at_k(HITS, frozenset({"a", "b"}), 2) == 1.0
    assert precision_at_k(HITS, frozenset({"a"}), 2) == 0.5
    assert precision_at_k([], frozenset({"a"}), 2) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_hit():
    assert reciprocal_rank(HITS, frozenset({"b"})) == 0.5
    assert reciprocal_rank(HITS, frozenset({"a", "c"})) == 1.0
    assert reciprocal_rank(HITS, frozenset({"z"})) == 0.0


def test_ndcg_rewards_higher_placement():
    top = ndcg_at_k(HITS, frozenset({"a"}), 3)
    bottom = ndcg_at_k(HITS, frozenset({"c"}), 3)
    assert top == 1.0
    assert top > bottom > 0.0


def test_query_from_dict_normalizes_shapes():
    q = Query.from_dict({"id": "q1", "query": "text", "relevant": "c0"})
    assert q.relevant == frozenset({"c0"})
    assert Query.from_dict({"id": "q2", "query": "t", "relevant": ["a", "b"]}).relevant == frozenset({"a", "b"})


def test_evaluate_aggregates_and_flags_total_misses():
    queries = [
        Query("q1", "chest pain", frozenset({"a"})),
        Query("q2", "nothing matches", frozenset({"zzz"})),
    ]
    report = evaluate("stub", lambda q, k: HITS[:k], queries, ks=(1, 3))
    assert report.queries == 2
    assert report.recall[1] == 0.5
    assert report.mrr == 0.5
    assert report.failures == ("q2",)


def test_evaluate_requires_at_least_one_k():
    with pytest.raises(ValueError):
        evaluate("s", lambda q, k: [], [], ks=())


def test_render_comparison_names_the_best():
    good = evaluate("good", lambda q, k: HITS[:k], [Query("q1", "x", frozenset({"a"}))], ks=(1, 5))
    bad = evaluate("bad", lambda q, k: [], [Query("q1", "x", frozenset({"a"}))], ks=(1, 5))
    out = render_comparison([good, bad], ks=(1, 5))
    assert "best R@5: good" in out
    assert render_comparison([]) == "no retrievers evaluated"
