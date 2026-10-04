# Benchmark

Measured with [`scripts/benchmark.py`](../scripts/benchmark.py). Read the **limits** section before using any
number here to size a deployment.

```bash
pip install -e '.[dev]'                      # psutil is a dev dependency
python scripts/benchmark.py --docs 150 --duration 8 --loadgen-processes 3 --json results.json
python scripts/benchmark.py --docs 120 --concurrency 16 --skip-warm --no-expansion --fuzziness 0
```

## Setup

| | |
|---|---|
| Machine | 4 vCPU Intel Xeon @ 2.1 GHz, 15 GiB RAM |
| Elasticsearch | 9.5.4, one node, 1 shard, 0 replicas, default heap, **same machine** as everything else |
| Server | one `python -m src.api.main` process (uvicorn + uvloop), in-process job queue, in-memory cache |
| Load generator | 3 processes (httpx), same machine |
| Models | deterministic fakes from `tests/fake_plugin.py` - **no LLM / embedding provider is called** |
| Corpus | 150 synthetic text documents, 20 chunks each = 3 000 chunks (120 documents / 2 400 chunks for the knob table) |
| Retrieval | defaults: query expansion to 4 variants, `ES_BM25_FUZZINESS=AUTO`, text/table/image indices - 24 Elasticsearch searches per request, sent as one `_msearch` |

The load generator is a separate process from the server and CPU use of the server, Elasticsearch and the
load generator is sampled during every run (`cpu_cores` = mean cores busy), so each result says what saturated.

## Ingestion

| | docs/s | chunks/s | server cores | Elasticsearch cores |
|---|---|---|---|---|
| 1.x design, as committed before this round (refresh waits inside every write) | 3.9 | 79 | 0.17 | 0.11 |
| current, `INGEST_REFRESH=each` (default) | 23.1 | 461 | 0.95 | 0.48 |
| current, `INGEST_REFRESH=interval` (earlier run, 150 docs) | 20.2 | 405 | 0.91 | 0.32 |

The old numbers are a **waiting** problem, not a work problem: server and Elasticsearch were both idle while
every document paid several 1-second refresh waits (delete-by-query `refresh=true`, a `wait_for` on the ledger
write, an explicit refresh). Removing the waits that are not needed for correctness (the stale-generation sweep
and the ledger write no longer force a refresh; one explicit refresh per document remains in `each` mode) made
ingestion ~6x faster on this workload. It is now bound by one Python process (server at ~0.95 cores:
parsing, chunking, hashing, JSON), so the next step up is more worker processes, not tuning.

`each` vs `interval` is a wash on throughput here (a single node refreshing a few small indices is cheap);
`interval` roughly cuts Elasticsearch CPU by a third. It may matter more on a large cluster; this machine can't
show that. In `interval` mode, a search made before Elasticsearch's own refresh would find nothing - the corpus is
marked *settling* for `ES_REFRESH_INTERVAL + 1s` after each ingest and results computed in that window are not
cached (without this a too-early empty answer was cached for an hour; see
`test_ingest_refresh_mode_decides_who_makes_new_chunks_searchable`). Re-ingesting a changed file does one refresh
before its stale-generation sweep, because the sweep only sees searchable chunks.

## Retrieval, cache-cold (every request is a new query)

| clients | QPS | p50 ms | p95 ms | p99 ms | server cores | Elasticsearch cores |
|---|---|---|---|---|---|---|
| 1 | 14.4 | 67 | 96 | 117 | 0.20 | 2.17 |
| 8 | 21.0 | 380 | 494 | 547 | 0.29 | 3.28 |
| 32 | 21.4 | 1 557 | 1 667 | 1 692 | 0.29 | 2.98 |
| 64 | 23.7 | 2 753 | 3 534 | 3 666 | 0.34 | 3.27 |

**Elasticsearch is the bottleneck, not the application**: it uses ~3 of the 4 cores while the server stays near
0.3. Each retrieval costs 24 searches; at ~21 QPS that is ~500 searches/s on this node. Beyond ~8 clients only
latency grows (queueing), which is what you want from a saturated dependency - no errors, no collapse.
This is the number that scales with Elasticsearch shards/replicas/nodes; the server tier does not need to.

### What moves it (16 clients, cache-cold, 2 400 chunks)

| setting | QPS | p50 ms | p95 ms | Elasticsearch cores |
|---|---|---|---|---|
| defaults | 21.5 | 765 | 870 | 3.12 |
| `query_expander = "identity"` (1 variant instead of 4) | 77.3 | 210 | 331 | 2.73 |
| `ES_BM25_FUZZINESS=0` | 56.2 | 283 | 421 | 1.56 |
| both | 181.7 | 87 | 202 | 1.38 |

Query expansion and fuzzy matching are the two big levers, **and both trade recall for speed**. This benchmark
has no relevance judgements, so it cannot say what the recall cost is on your data; measure that before turning
either off. Also note that with a real LLM expansion adds a model call (cached for `QUERY_EXPANSION_CACHE_TTL_SECONDS`)
that the fake model here does not.

## Retrieval, cache-warm (20 distinct queries, so almost every request is a cache hit)

| clients | QPS | p50 ms | p95 ms | p99 ms | server cores | load-gen cores |
|---|---|---|---|---|---|---|
| 1 | 436 | 1.8 | 2.6 | 4.7 | 0.47 | 0.41 |
| 8 | 1 012 | 6.6 | 10.9 | 15.8 | 0.85 | 0.94 |
| 32 | 1 055 | 24 | 39 | 158 | 0.84 | 1.02 |
| 64 | 1 109 | 48 | 66 | 194 | 0.85 | 1.13 |

One server process serves ~1 000-1 100 cached requests per second and then stops scaling because it is a single
Python process using one core (GIL). The remaining cores belong to the load generator and Elasticsearch on this
machine. Scale it by adding uvicorn workers (`WEB_CONCURRENCY`) or replicas behind a load balancer, with
`CACHE_BACKEND=tiered` (or `redis`) so replicas share cached results and the corpus version.

## Batching

One retrieval as 24 separate Elasticsearch calls (variants x indices x lexical/vector) took a median of 55.6 ms;
as the single `_msearch` it is sent as, 29.9 ms (**1.9x**, 24 round trips -> 1). On loopback the round trip is
nearly free; across a real network the serial form pays 24 round trips of latency, so the gap grows.

## Limits - what these numbers are not

* **Fake models.** Real embedding and chat calls add their own latency and provider rate limits; those scale
  independently of this code (and are what `MODEL_MAX_CONCURRENCY` bounds).
* **One 4-core machine shares everything.** Elasticsearch, the server and the load generator compete for
  the same cores. A real cluster separates them, so the cold-retrieval ceiling here is *lower* than a real
  deployment's, while the warm-path and ingestion ceilings (one process each) carry over.
* **Small corpus.** 2 400-3 000 chunks fit in memory. Query cost grows with corpus size, shard count and vector
  dimensions (64 here; real models use 768-3 072). Nothing here measures a million-chunk index.
* **Synthetic text.** Chunk sizes, vocabulary and OCR cost are not representative; OCR (the dominant ingestion
  cost for scans) is off in these runs.
* **Single run each, no confidence intervals.** Differences of roughly 10 % are inside the noise of a shared VM.
* **Not measured at all:** multi-process servers under Redis, a real Redis network hop, GPU OCR, upload throughput of
  large PDFs, the MCP endpoint, long-running soak behaviour.

## Harness notes

Earlier numbers from this project's benchmark were wrong for two reasons that are fixed now and worth knowing
if you extend it: the first version ran the server and the load generator in one process (they shared a GIL), and a
single load-generator process saturates a core at roughly 1 000 requests/s - which looked like the *server*
slowing down as concurrency rose (server CPU fell while "QPS" fell). The harness now runs the server as its own
process, spreads clients over `--loadgen-processes`, reports the load generator's CPU and gives every
cache-cold run query strings the cache has never seen.
