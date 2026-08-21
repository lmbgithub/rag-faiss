"""Command-line interface: index a corpus, query it, or evaluate retrievers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ragkit import __version__
from ragkit.embeddings import HashingEmbedder
from ragkit.evaluate import Query, evaluate, render_comparison
from ragkit.index import FAISS_AVAILABLE
from ragkit.pipeline import RagPipeline

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_USAGE = 2

MODES = ("dense", "lexical", "hybrid", "hybrid_weighted")


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    rows = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{line_no}: invalid JSON ({exc.msg})") from exc
        rows.append(payload)
    return rows


def build_pipeline(
    docs_path: str, embedder_name: str, *, max_chars: int = 400, overlap: int = 80
) -> RagPipeline:
    if embedder_name == "hashing":
        embedder = HashingEmbedder()
    else:
        from ragkit.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(embedder_name)

    rows = load_jsonl(docs_path)
    pipeline = RagPipeline(embedder, max_chars=max_chars, overlap=overlap)
    pipeline.add_documents(
        [(str(r.get("id", f"doc-{i:04d}")), str(r.get("text", ""))) for i, r in enumerate(rows)]
    )
    return pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragkit", description="Small RAG pipeline with measurable retrieval."
    )
    parser.add_argument("--version", action="version", version=f"rag-faiss {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = {"--embedder": "hashing"}

    q = sub.add_parser("query", help="retrieve passages for a query")
    q.add_argument("docs", help="JSONL corpus with 'id' and 'text'")
    q.add_argument("query", help="the query text")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--mode", choices=MODES, default="hybrid")
    q.add_argument("--embedder", default="hashing", help="'hashing' or a sentence-transformers model name")
    q.add_argument("--context", action="store_true", help="print the assembled cited context block")
    q.add_argument("--max-chars", type=int, default=400, help="chunk size budget")
    q.add_argument("--overlap", type=int, default=80, help="chunk overlap")

    e = sub.add_parser("evaluate", help="compare retrieval modes against labelled queries")
    e.add_argument("docs", help="JSONL corpus with 'id' and 'text'")
    e.add_argument("queries", help="JSONL with 'id', 'query', and 'relevant' chunk ids")
    e.add_argument("--embedder", default="hashing")
    e.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    e.add_argument("--min-recall", type=float, default=None, help="fail if best R@5 is below this")
    e.add_argument("--max-chars", type=int, default=400, help="chunk size budget")
    e.add_argument("--overlap", type=int, default=80, help="chunk overlap")
    e.add_argument("--json", dest="json_out")

    sub.add_parser("info", help="show which backends are available")
    return parser


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args.docs, args.embedder, max_chars=args.max_chars, overlap=args.overlap)
    print(f"corpus: {len(pipeline)} chunks | index: {pipeline.backend} | embedder: {pipeline.embedder.name}")
    print(f"mode: {args.mode}\n")

    if args.context:
        context, passages = pipeline.build_context(args.query, k=args.k, mode=args.mode)
        print(context)
        print(f"\n{len(passages)} passage(s) in context")
        return EXIT_OK

    for passage in pipeline.retrieve(args.query, k=args.k, mode=args.mode):
        snippet = passage.chunk.text[:160].replace("\n", " ")
        print(f"{passage.rank}. [{passage.score:.4f}] {passage.citation}")
        print(f"   {snippet}{'...' if len(passage.chunk.text) > 160 else ''}\n")
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args.docs, args.embedder, max_chars=args.max_chars, overlap=args.overlap)
    queries = [Query.from_dict(r) for r in load_jsonl(args.queries)]

    print(f"corpus: {len(pipeline)} chunks | index: {pipeline.backend} | embedder: {pipeline.embedder.name}")
    print(f"queries: {len(queries)}\n")

    reports = [
        evaluate(mode, lambda q, k, m=mode: pipeline.search(q, k=k, mode=m), queries)
        for mode in args.modes
    ]
    print(render_comparison(reports))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([r.to_dict() for r in reports], indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")

    if args.min_recall is not None:
        best = max(r.recall[5] for r in reports)
        passed = best >= args.min_recall
        print(f"\ngate: best R@5 {best:.1%} vs threshold {args.min_recall:.1%} -> {'PASS' if passed else 'FAIL'}")
        return EXIT_OK if passed else EXIT_BELOW_THRESHOLD
    return EXIT_OK


def _cmd_info(_: argparse.Namespace) -> int:
    # Probe with find_spec rather than importing. Importing
    # sentence-transformers pulls in torch, which costs seconds and, on some
    # platforms, conflicts with an already-loaded faiss over OpenMP runtimes.
    # Asking "is it installed?" should never risk loading it.
    from importlib.util import find_spec

    st = find_spec("sentence_transformers") is not None
    print(f"rag-faiss {__version__}")
    print(f"faiss available:                 {FAISS_AVAILABLE}")
    print(f"sentence-transformers available: {st}")
    print(f"index backend in use:            {'faiss.IndexFlatIP' if FAISS_AVAILABLE else 'numpy.bruteforce'}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "query":
            return _cmd_query(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "info":
            return _cmd_info(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser.error(f"unknown command: {args.command}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
