# rag-faiss

A small RAG retrieval pipeline you can actually measure: chunking, embeddings,
FAISS, BM25, hybrid fusion, and the retrieval metrics that tell you whether any
of it is working.

Runs on clone with no model download and no network. FAISS and
sentence-transformers are optional extras, and the package degrades to an exact
NumPy index when FAISS is absent.

```
$ ragkit evaluate examples/corpus.jsonl examples/queries.jsonl
corpus: 19 chunks | index: faiss.IndexFlatIP | embedder: hashing-256d
queries: 20

==============================================================
RETRIEVAL COMPARISON
==============================================================
retriever                    R@1       R@5       MRR    nDCG@5
--------------------------------------------------------------
dense                     47.5%     62.5%     0.587     0.559
lexical                   45.0%     72.5%     0.632     0.625
hybrid                    55.0%     77.5%     0.691     0.678
hybrid_weighted           65.0%     75.0%     0.753     0.724
--------------------------------------------------------------
best R@5: hybrid
still zero-recall on 1 query/queries: q16
```

## Why this exists

Most RAG systems are evaluated by reading a few generated answers and deciding
they look reasonable. That measures the generator, not the retriever, and it
cannot distinguish _the model wrote a plausible answer_ from _the right chunk
was actually retrieved_.

If the correct chunk is not in the context window, no prompt can fix it.
Retrieval recall is the ceiling on everything downstream, so this package
measures it first and separately.

## The result that makes the point

Run the same corpus and queries twice, changing only the embedder:

| Retriever                       | R@1       | R@5        | MRR       | nDCG@5    |
| ------------------------------- | --------- | ---------- | --------- | --------- |
| **hashing embedder (no model)** |           |            |           |           |
| dense                           | 47.5%     | 62.5%      | 0.587     | 0.559     |
| lexical (BM25)                  | 45.0%     | 72.5%      | 0.632     | 0.625     |
| hybrid (RRF)                    | 55.0%     | **77.5%**  | 0.691     | 0.678     |
| hybrid (weighted)               | **65.0%** | 75.0%      | **0.753** | **0.724** |
| **all-MiniLM-L6-v2**            |           |            |           |           |
| dense                           | **77.5%** | **100.0%** | **0.925** | **0.941** |
| lexical (BM25)                  | 45.0%     | 72.5%      | 0.632     | 0.625     |
| hybrid (RRF)                    | 75.0%     | 85.0%      | 0.863     | 0.832     |
| hybrid (weighted)               | 65.0%     | 95.0%      | 0.817     | 0.840     |

With a weak embedder, hybrid retrieval is the clear winner — the usual advice.
With a real semantic embedder, **hybrid makes things worse**: dense alone hits
100% R@5 and fusing it with a much weaker BM25 drags it down to 85%.

That is the whole argument for having an evaluation harness. Rank fusion helps
when the two retrievers are of comparable strength and hurts when one dominates,
and no blog post can tell you which case you are in — only a measurement on your
corpus can. The default in this package is hybrid because it is the safer choice
under an unknown embedder, and `ragkit evaluate` exists so you do not have to
keep that default on faith.

(Both runs use the same 12-document synthetic knowledge base and the same 20
labelled queries in `examples/`. Reproduce with `--embedder
sentence-transformers/all-MiniLM-L6-v2`.)

## Install

```bash
git clone https://github.com/<your-username>/rag-faiss.git
cd rag-faiss

pip install -e ".[dev]"     # + faiss
pip install -e ".[all]"     # + faiss + sentence-transformers
pip install -e .            # NumPy only; exact fallback index
```

```
$ ragkit info
rag-faiss 0.1.0
faiss available:                 True
sentence-transformers available: True
index backend in use:            faiss.IndexFlatIP
```

## Usage

Corpus is JSONL with `id` and `text`:

```bash
ragkit query examples/corpus.jsonl "how do I undo a bad release" -k 3
ragkit query examples/corpus.jsonl "XR-4471" --mode lexical
ragkit query examples/corpus.jsonl "is my data encrypted" --context
```

`--context` prints the assembled, cited block exactly as a model would receive
it:

```
[1] (kb-storage:0-198)
Objects are stored with server-side encryption enabled by default using keys
managed by the platform. Customer-managed keys are available on request.
```

Evaluation needs labelled queries — `id`, `query`, and the `relevant` chunk ids:

```bash
ragkit evaluate corpus.jsonl queries.jsonl --min-recall 0.75   # exits 1 below
ragkit evaluate corpus.jsonl queries.jsonl --embedder sentence-transformers/all-MiniLM-L6-v2
```

### As a library

```python
from ragkit import RagPipeline, SentenceTransformerEmbedder

pipeline = RagPipeline(SentenceTransformerEmbedder(), max_chars=400, overlap=80)
pipeline.add_documents([("kb-auth", auth_text), ("kb-billing", billing_text)])

context, passages = pipeline.build_context("why did authentication fail?", k=5)
answer = my_llm(f"Answer using only the context.\n\n{context}\n\nQ: {question}")

for p in passages:
    print(p.rank, p.citation)   # kb-auth:198-421
```

## Design notes

**Chunks carry their source and character offsets.** A retrieved answer can
always be traced to an exact span of an exact document. A RAG system that cannot
cite its own source is not auditable, and `build_context` returns the passages it
used so you can always report exactly what the model was shown.

**Chunking splits on natural boundaries.** Paragraphs first, then sentences, and
only a hard character cut when a single sentence exceeds the budget. Half a
clause has no clear meaning to average over, so it embeds poorly.

**Vectors are normalized once at ingest.** Inner product over unit vectors _is_
cosine similarity, so `IndexFlatIP` returns cosine directly — cheaper and less
error-prone than dividing by norms on every query.

**BM25 exists for the queries embeddings lose.** Product codes, error
identifiers, drug names, version numbers — the terms users paste verbatim are
exactly the ones an embedder never saw in training. In the table above, dense
retrieval with the hashing embedder misses `XR-4471` and `SSO-8802`; BM25 finds
them first every time.

**Fusion combines ranks, not scores.** BM25 scores are unbounded and
corpus-dependent; cosine sits in [-1, 1]. Any weighted sum silently becomes
"whatever BM25 said" on some corpora and "whatever the embedder said" on others,
with no warning either way. Reciprocal Rank Fusion combines ranks, which are
comparable by construction. `weighted_score_fusion` is included so the harness
can show you which one wins on your data.

**Hybrid modes over-fetch before fusing.** An item ranked 8th by one retriever
and 2nd by the other should be able to win, and it cannot if both lists were
already truncated to k.

**The NumPy fallback is exact, not approximate.** `IndexFlatIP` is a brute-force
exact index too, so both paths return identical results — verified by a test.
CI runs the full suite with FAISS absent.

## Metrics

| Metric              | What it answers                                                                 |
| ------------------- | ------------------------------------------------------------------------------- |
| Recall@k            | Is the right chunk in the context at all? The ceiling on everything downstream. |
| Precision@k         | How much of the context window is wasted?                                       |
| MRR                 | How far down is the first good chunk? Matters when only a few chunks fit.       |
| nDCG@k              | Rank-weighted quality across all relevant chunks.                               |
| zero-recall queries | Which queries retrieve nothing useful. The list to actually go fix.             |

`evaluate` calls the retriever once at the largest k and truncates for smaller
cutoffs, rather than re-querying per k for identical results.

## Tests

```bash
pytest -q     # 68 tests
```

Covers chunk-boundary and overlap behaviour, oversized-sentence hard splitting,
zero-vector normalization, FAISS/NumPy backend agreement, duplicate-key and
dimension-mismatch rejection, the BM25 IDF floor (without which common terms get
negative weight and push matching documents _down_), RRF's scale invariance,
every metric's edge cases, context budgeting, and CLI exit codes.

## Roadmap

- Cross-encoder reranking over the fused candidate list
- Approximate indexes (IVF, HNSW) with a recall-vs-latency curve against exact
- Query expansion and multi-query retrieval
- Chunk-size sweep as a first-class experiment
- MLflow logging for the evaluation runs

## License

MIT — see [LICENSE](LICENSE).
