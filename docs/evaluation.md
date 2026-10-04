# Evaluation

`rag eval` answers "did that change make retrieval better?" with data instead of a feeling. It runs a
labelled question set through the **real pipeline** (your collections, models and settings), scores every
query, and reports each metric with a confidence interval. Two configurations are compared on the *same*
queries with a paired bootstrap, so a reported difference comes with an interval.

```bash
rag eval run cases.jsonl -c legal --out base.json                    # one configuration
rag eval sweep cases.jsonl -c legal \
    -v base:reranker=identity -v precise:reranker=precise \
    -v noexp:query_expander=identity,hybrid_alpha=0.7                # several, side by side (first = baseline)
rag eval compare base.json precise.json                              # saved reports, same dataset
rag eval check precise.json --min ndcg@10=0.55 --baseline base.json  # exit 1 on regression: for CI
rag eval run cases.jsonl --answers --judge-model claude              # also judge generated answers
rag eval generate -c legal -n 100 -o cases.jsonl                     # draft a question set (synthetic)
```

## The dataset

One JSON object per line:

```json
{"id": "q17", "query": "How did cloud revenue change in Q2?",
 "relevant": [{"source": "report.pdf", "pages": [3, 4]}, {"source": "press-release.txt", "grade": 2}],
 "reference_answer": "It grew twelve percent.", "tags": ["finance"]}
```

* `relevant` is the *evidence* a good retrieval must surface. List **distinct** pieces of evidence: each label is
  one item to find, and nDCG's ideal ranking assumes one result per label, so two labels that one chunk satisfies
  (say `source` + `pages` *and* a `contains` for the same passage) make a perfect nDCG unreachable. A label matches a retrieved chunk when **every**
  key it sets matches: `source` (full path, file name or a trailing path), `pages`, `chunk_id`, `contains`
  (case-insensitive substring of the chunk text). `grade` (default 1) weights it in nDCG.
  Prefer labels that survive re-chunking (`source` + `pages`, or `contains`) over `chunk_id`.
* `answerable: false` (and no labels) marks a question the corpus cannot answer. It is skipped by the retrieval
  metrics and, with `--answers`, checks that the system *declines* rather than invents an answer.
* `collection`, `kinds` pick the scope for a case; `tags` give per-tag breakdowns in the report.
* Parsing is strict - an unknown key, a bad type or a duplicate id is an error naming the line, so a typo
  cannot quietly weaken the set.

## Metrics

Per query, over the ranked results (`--k 1,3,5,10`):

| Metric | Meaning |
|---|---|
| `hit@k` | 1 if any relevant evidence is in the top k |
| `recall@k` | share of the query's labels found in the top k (a query with more labels than k cannot reach 1) |
| `precision@k` | share of the top k that match a label (denominator is k, even if fewer came back) |
| `mrr@K` | 1 / rank of the first relevant result within the largest k |
| `ndcg@k` | gain-discounted ranking quality, using label grades; 1.0 = best possible order |

Each label can be found once: further chunks that match only already-found labels earn no gain, so splitting a
page into five chunks cannot inflate recall or nDCG. `--granularity document` first collapses chunks to their
document (best chunk wins) - use it when relevance is document-level, as in most public benchmarks.

With `--answers` the answer service writes an answer per question and:

| Metric | How |
|---|---|
| `faithfulness` | an LLM splits the answer into claims and checks each against the retrieved context |
| `correctness` | an LLM compares the answer with `reference_answer` (correct 1 / partial 0.5 / incorrect 0) |
| `abstained` | for `answerable: false` questions: did the answer decline? |
| `citation_valid`, `citation_rate` | deterministic: do the `[Source: file, Page: n]` citations point at chunks that were actually retrieved, and does the answer cite at all? |

**An LLM judge is a model with its own errors.** Use a strong judge that is *not* the model being judged, treat
small differences as noise, and read some verdicts in the saved report (`cases[].answer.claims`). A judge reply
that cannot be parsed is recorded under `judge_errors` and the case is *left out* of that metric (its `n` is
smaller) - it is never scored as 0.

## Statistics

* Each metric is a mean over queries with a 95% **bootstrap** interval (2 000 resamples, fixed seed). The
  interval is about *which questions you happened to ask*; with 300 queries expect roughly ±0.04 on nDCG.
* Comparisons resample the **paired differences** (same queries in both runs), which is far tighter than
  comparing two intervals: query difficulty cancels. `*` marks an interval that excludes 0.
* Pick the metric you care about *before* looking (nDCG@10 for ranking, recall@k for "is it in the context
  window"). A table of 17 metrics x several variants will contain false positives; treat starred rows on a
  metric you did not pre-choose as leads. Small query sets (< ~100) cannot resolve differences below ~0.05.
* Latency in a report is wall time per query. With `--concurrency` above 1 (the default is 4) it includes
  queueing behind other queries - fine for quality runs, but use `--concurrency 1` when you want latency itself.
* Failed queries are never dropped: they score 0, are counted in `errors`, and fail the run (exit 1) unless
  `--allow-errors`. Every run uses its own in-memory cache, so repeated queries are not answered from a warm
  shared cache.

## Synthetic questions

`rag eval generate` samples chunks and has a chat model write a question each one answers; the chunk's source and
page become the label. It is a fast way to get a regression set. But the questions are written *from* the
passage and tend to reuse its wording, which flatters lexical retrieval, so absolute scores are optimistic.
Review a sample, and for decisions that matter use hand-labelled questions or real query logs.

## Rerankers

The pipeline retrieves with BM25 + vectors, fuses the lists, merges the candidates of all query variants, then
asks one **reranker** to order the best `RERANK_CANDIDATES` (default 30) against *your* query:

| `provider` | What | Notes |
|---|---|---|
| `identity` (default) | keep the fused order | |
| `heuristic` | hand-weighted keyword/page/kind boosts | opt-in; see the result below |
| `cross-encoder` | a local relevance model (`pip install 'ai-rag-info[local]'`) | CPU or GPU, no API key; `model = "cross-encoder/ms-marco-MiniLM-L6-v2"`, `BAAI/bge-reranker-base`, ... |
| `cohere`, `jina` | hosted rerank API (`COHERE_API_KEY` / `JINA_API_KEY`) | `base_url` points at any compatible server |

```toml
reranker = "precise"
[reranker_models.precise]
provider = "cross-encoder"
model = "BAAI/bge-reranker-base"
batch_size = 16
[reranker_models.precise.options]
device = "auto"       # auto | cpu | cuda | mps
```

**Searching several collections at once** is where a model reranker matters most: first-stage scores (BM25
statistics, similarities from different embedding models) are per index and not comparable across collections,
so with `identity` the collections are interleaved by rank - each contributes its best results, but the overall
best match is not guaranteed to come first. A reranker scores every candidate on one scale.

A reranker that fails raises a typed error (HTTP 502); the pipeline never falls back to the unreranked list.
Reranking runs once per cache miss, so its cost is bounded by `RERANK_CANDIDATES`, not by how many query variants
were generated. **Whether it helps depends on the model and the corpus - measure it** (next section).

## Worked example: BEIR SciFact, real models

A public benchmark with human relevance judgements: 5 183 scientific abstracts, 300 test claims.
Everything below runs locally with no API key:

```bash
pip install -e '.[dev,local,worker]'
python scripts/beir_prepare.py --name scifact --work /tmp/beir          # download, ingest (~12 min on 4 vCPU), write cases.jsonl + rag.toml
export RAG_CONFIG=/tmp/beir/scifact/rag.toml OPENAI_API_KEY=unused ES_HOST=http://localhost:9200
rag eval sweep /tmp/beir/scifact/cases.jsonl -c scifact --granularity document --k 1,3,5,10 \
    --set rerank_candidates=50 \
    -v hybrid:reranker=identity -v bm25:reranker=identity,hybrid_alpha=1.0 \
    -v dense:reranker=identity,hybrid_alpha=0.0 -v heuristic:reranker=heuristic \
    -v ce:reranker=ce -v bge:reranker=bge -v bm25ce:reranker=ce,hybrid_alpha=1.0
```

Setup: 17 881 chunks (default 800/200-character chunker, ~3.5 per abstract) in one Elasticsearch 9.5.4 node;
embeddings `BAAI/bge-small-en-v1.5` (local, CPU); query expansion off; results collapsed to documents (the run
requests 30 chunks so that ten distinct documents are usually available, and reranks the best 50 fused chunks).

| configuration | nDCG@10 [95% CI] | Δ vs hybrid [95% CI] | recall@10 | hit@1 | MRR@10 | latency p50 ms |
|---|---|---|---|---|---|---|
| hybrid (BM25 + dense, no reranker) | 0.718 [0.674, 0.759] | — | 0.844 | 0.600 | 0.685 | 84 |
| BM25 only | 0.626 [0.581, 0.671] | -0.093 [-0.118, -0.068] * | 0.771 | 0.497 | 0.586 | |
| dense only (bge-small) | 0.712 [0.667, 0.754] | -0.006 [-0.031, +0.017] | 0.823 | 0.617 | 0.686 | |
| hybrid + heuristic reranker | 0.671 [0.628, 0.711] | -0.048 [-0.066, -0.030] * | 0.825 | 0.513 | 0.630 | |
| hybrid + cross-encoder `ms-marco-MiniLM-L6` | 0.696 [0.654, 0.736] | -0.022 [-0.048, +0.003] | 0.832 | 0.563 | 0.662 | 951 |
| hybrid + cross-encoder `bge-reranker-base` | 0.670 [0.629, 0.709] | -0.048 [-0.078, -0.018] * | 0.846 | 0.510 | 0.622 | 6 107 |
| BM25 only + cross-encoder `MiniLM-L6` | 0.690 [0.647, 0.730] | | 0.811 | 0.567 | 0.661 | |

`*` = the paired interval excludes 0. nDCG@10 was the metric chosen in advance. Latency: p50 of the first 40
queries run one at a time (`--concurrency 1`), CPU only, 50 candidates reranked. Neither cross-encoder changed
recall@10 over hybrid measurably (the candidate pool is the same): a reranker can only reorder it.

Reranking on top of the weak first stage is the one place it paid off: BM25 only -> BM25 + MiniLM is
**+0.064 nDCG@10** [+0.035, +0.094], `hit@1` +0.070 [+0.020, +0.123].

Reading it:

* **The harness reproduces published numbers.** Dense-only (0.712) is where bge-small is published (about 0.71 on
  the MTEB leaderboard), and BM25 + the MiniLM cross-encoder (0.690) is where the BEIR paper puts the same
  combination (0.688). Our BM25 (0.626) is a little below the paper's 0.665 - ours is Elasticsearch `multi_match`
  over chunks - but its interval includes it. Those external figures are from memory: check them before quoting.
* **Hybrid beats BM25 by a wide margin** (+0.093 nDCG@10) and is statistically level with dense-only on this corpus.
* **A reranker is not automatically a win.** On a weak first stage (BM25) the cross-encoder is a clear gain. On top of
  a strong hybrid first stage neither cross-encoder helped: MiniLM -0.022 (interval includes 0), and the larger
  `bge-reranker-base` -0.048 (significant) at 6 s per query on CPU. Both left recall@10 where it was and only
  reshuffled the top, mostly for the worse (`hit@1` fell from 0.600 to 0.563 / 0.510). Bigger was not better.
* **The hand-weighted `heuristic` reranker made results worse** (-0.048 nDCG@10, interval [-0.066, -0.030]). Its
  additive boosts are large next to the tiny spread of fused scores, so it effectively re-sorts by keyword overlap and
  ignores the fusion. That is why it is no longer the default.
* **Cost:** hybrid retrieval answers in ~84 ms; the small cross-encoder adds ~0.9 s on this CPU, the base one ~6 s.

What this does *not* tell you: why. One untested hypothesis is that the cross-encoders judge an 800-character chunk
rather than the whole abstract (the pipeline scores chunks; relevance here is per document); another is that
these models were trained on web-search questions and SciFact queries are scientific claims. Both are cheap to test
with `rag eval` on your own corpus, which is the point.

### Limits of this example

* One benchmark, one language, text only: scientific claims against abstracts. Tables, scans, multilingual
  corpora and conversational queries were not measured. Another corpus can rank these differently.
* 300 queries: differences under ~0.03 are not resolvable. All numbers are single runs.
* Latency is for CPU inference on a shared 4-vCPU machine with four queries in flight sharing one model
  (so it includes queueing). A GPU, a hosted API or fewer candidates changes it a lot; the quality column does not
  depend on it.
* Query expansion with a real LLM and the LLM-judged answer metrics were exercised only against scripted fake models
  in tests, not against a live provider.
