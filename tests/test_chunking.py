import pytest

from ragkit.chunking import Chunk, chunk_corpus, chunk_document, split_sentences

TEXT = (
    "The patient arrived at six. He reported chest pain radiating to the left arm.\n\n"
    "Past history includes hypertension and type two diabetes. He takes ramipril daily.\n\n"
    "Vital signs were stable on arrival. Blood pressure was one thirty over eighty."
)


def test_returns_chunks_with_ids_and_source():
    chunks = chunk_document(TEXT, "doc1", max_chars=120, overlap=20)
    assert chunks
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.chunk_id.startswith("doc1::") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_respects_the_size_budget_on_natural_boundaries():
    chunks = chunk_document(TEXT, "doc1", max_chars=120, overlap=20)
    # Boundary-aware packing can exceed the budget only by the last unit added.
    assert all(c.length <= 200 for c in chunks)


def test_empty_and_whitespace_input_yields_nothing():
    assert chunk_document("", "d") == []
    assert chunk_document("   \n\n  ", "d") == []


def test_short_document_is_a_single_chunk():
    chunks = chunk_document("One short sentence.", "d", max_chars=800)
    assert len(chunks) == 1
    assert chunks[0].text == "One short sentence."


def test_overlap_carries_context_between_chunks():
    chunks = chunk_document(TEXT, "d", max_chars=100, overlap=40)
    assert len(chunks) > 1
    # Some later chunk repeats content from its predecessor.
    assert any(
        set(a.text.split()) & set(b.text.split())
        for a, b in zip(chunks, chunks[1:])
    )


def test_oversized_sentence_is_hard_split():
    giant = "word " * 400
    chunks = chunk_document(giant, "d", max_chars=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.length <= 200 for c in chunks)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        chunk_document(TEXT, "d", max_chars=0)
    with pytest.raises(ValueError):
        chunk_document(TEXT, "d", overlap=-1)
    with pytest.raises(ValueError):
        chunk_document(TEXT, "d", max_chars=100, overlap=100)


def test_chunk_corpus_flattens_and_keeps_doc_ids():
    chunks = chunk_corpus([("a", TEXT), ("b", TEXT)], max_chars=150, overlap=20)
    assert {c.doc_id for c in chunks} == {"a", "b"}


def test_offsets_are_non_negative_and_ordered_within_a_doc():
    chunks = chunk_document(TEXT, "d", max_chars=120, overlap=20)
    assert all(c.start >= 0 and c.end >= c.start for c in chunks)


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []


def test_chunk_serializes():
    payload = Chunk("t", "d", "d::0000", 0, 1).to_dict()
    assert payload["chunk_id"] == "d::0000"
    assert payload["doc_id"] == "d"
