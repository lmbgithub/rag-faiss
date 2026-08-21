import json
from pathlib import Path

import pytest

from ragkit.cli import EXIT_BELOW_THRESHOLD, EXIT_OK, EXIT_USAGE, main
from ragkit.embeddings import HashingEmbedder
from ragkit.pipeline import RagPipeline

DOCS = [
    ("clinical", "The patient reported chest pain radiating to the left arm at six in the morning. "
                 "Vital signs were stable with blood pressure of one thirty over eighty."),
    ("finance", "Quarterly revenue exceeded the forecast by twelve percent. "
                "Operating margin improved across all three business units."),
    ("support", "Error code XR-4471 indicates a failed authentication handshake. "
                "Rotate the client secret and retry the connection."),
]


@pytest.fixture
def pipeline():
    p = RagPipeline(HashingEmbedder(dim=512), max_chars=200, overlap=40)
    p.add_documents(DOCS)
    return p


def test_pipeline_indexes_chunks(pipeline):
    assert len(pipeline) > 0
    assert len(pipeline.index) == len(pipeline)
    assert len(pipeline.bm25) == len(pipeline)


def test_adding_no_documents_is_safe():
    p = RagPipeline(HashingEmbedder(dim=32))
    assert p.add_documents([]) == []
    assert p.search("anything", k=3) == []


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid", "hybrid_weighted"])
def test_every_mode_retrieves(pipeline, mode):
    hits = pipeline.search("chest pain", k=3, mode=mode)
    assert hits
    assert hits[0].key.startswith("clinical")


def test_unknown_mode_is_rejected(pipeline):
    with pytest.raises(ValueError, match="unknown mode"):
        pipeline.search("x", mode="telepathy")


def test_lexical_mode_wins_on_an_exact_identifier(pipeline):
    hits = pipeline.search("XR-4471", k=1, mode="lexical")
    assert hits[0].key.startswith("support")


def test_retrieve_returns_passages_with_citations(pipeline):
    passages = pipeline.retrieve("chest pain", k=2)
    assert passages
    assert all(":" in p.citation and "-" in p.citation for p in passages)
    assert passages[0].chunk.text


def test_build_context_is_cited_and_budgeted(pipeline):
    context, used = pipeline.build_context("chest pain", k=3, max_chars=250)
    assert used
    assert context.startswith("[1] (")
    assert len(context) <= 400
    # Every passage the model would see is reported back to the caller.
    assert all(f"[{i}]" in context for i in range(1, len(used) + 1))


def test_build_context_always_includes_at_least_one_passage(pipeline):
    _, used = pipeline.build_context("chest pain", k=3, max_chars=1)
    assert len(used) == 1


# -- CLI -----------------------------------------------------------------


def write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def corpus_file(tmp_path):
    return write(tmp_path / "docs.jsonl", [{"id": d, "text": t} for d, t in DOCS])


def test_cli_query(corpus_file, capsys):
    assert main(["query", str(corpus_file), "chest pain", "-k", "2"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "clinical" in out
    assert "index:" in out


def test_cli_query_context_mode(corpus_file, capsys):
    assert main(["query", str(corpus_file), "chest pain", "--context"]) == EXIT_OK
    assert "[1] (" in capsys.readouterr().out


def test_cli_evaluate_and_gate(corpus_file, tmp_path, capsys):
    p = RagPipeline(HashingEmbedder(), max_chars=800, overlap=120)
    p.add_documents(DOCS)
    clinical = next(k for k in p.chunks if k.startswith("clinical"))

    queries = write(tmp_path / "q.jsonl", [
        {"id": "q1", "query": "chest pain left arm", "relevant": [clinical]},
    ])
    out_json = tmp_path / "r.json"
    assert main(["evaluate", str(corpus_file), str(queries), "--json", str(out_json),
                 "--min-recall", "0.5"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "RETRIEVAL COMPARISON" in out
    assert "PASS" in out
    assert len(json.loads(out_json.read_text())) == 4


def test_cli_evaluate_gate_fails_on_impossible_labels(corpus_file, tmp_path, capsys):
    queries = write(tmp_path / "q.jsonl", [
        {"id": "q1", "query": "chest pain", "relevant": ["does-not-exist"]},
    ])
    assert main(["evaluate", str(corpus_file), str(queries), "--min-recall", "0.9"]) == EXIT_BELOW_THRESHOLD
    assert "FAIL" in capsys.readouterr().out


def test_cli_info(capsys):
    assert main(["info"]) == EXIT_OK
    assert "faiss available" in capsys.readouterr().out


def test_cli_missing_file(tmp_path, capsys):
    assert main(["query", str(tmp_path / "nope.jsonl"), "x"]) == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_cli_malformed_corpus(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "text": "ok"}\nnope\n', encoding="utf-8")
    assert main(["query", str(bad), "x"]) == EXIT_USAGE
    assert ":2:" in capsys.readouterr().err
