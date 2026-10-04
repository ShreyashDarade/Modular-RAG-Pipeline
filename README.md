# Modular RAG Pipeline

Retrieval-Augmented Generation with **multilingual OCR** (English, Hindi, Marathi), built to be
**modular** (every moving part sits behind a small interface and is chosen by name) and
**horizontally scalable** (stateless API replicas, a separate ingestion-worker tier, shared
Elasticsearch and Redis).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142-green.svg)](https://fastapi.tiangolo.com/)
[![Elasticsearch 9](https://img.shields.io/badge/Elasticsearch-9.x-yellow.svg)](https://www.elastic.co/)
[![MCP](https://img.shields.io/badge/MCP-server-purple.svg)](https://modelcontextprotocol.io/)

## What it does

| | |
|---|---|
| **Many file types** | PDF (text, tables, embedded images), images and multi-page TIFF (OCR), text/Markdown, HTML, CSV/TSV, DOCX, XLSX — one parser per type, add more with a plug-in |
| **Collections** | Independent corpora, each with its own embedding model, chunker, accepted file types and indices. Ingest into, and search across, any selection of them |
| **Selective indexing** | Per request choose collections, content *kinds* (`text`, `table`, `image`) and even individual documents. Skipping `image` skips OCR entirely |
| **Hybrid retrieval** | BM25 + vector search fused with Reciprocal Rank Fusion, cross-reference expansion (same-page / adjacent-page chunks), re-ranking, diversity |
| **Multiple models** | Named chat and embedding profiles: OpenAI, Azure OpenAI, Anthropic, Google, Ollama (or your own). Pick the chat model per request |
| **Chat** | Multi-turn conversations with question condensing, per-turn model choice, server-sent-event streaming, Redis-backed history |
| **MCP server** | `search_documents`, `ask_question`, `list_*` tools over stateless Streamable HTTP or stdio — connect Claude Desktop, IDE agents, … |
| **Scales out** | Stateless API replicas, Redis-Streams job queue with crash recovery, shared cache / rate limits / conversations, load shedding, Prometheus metrics |

> **No silent fallbacks.** A failed dependency, an unknown model name, a missing optional package or
> an unsupported OCR language is an *error*, reported with a typed code — never quietly replaced
> by a default or a degraded answer. Optional behaviour (query expansion, re-ranking, OCR, a cache
> backend) is switched on or off in configuration, explicitly.

---

## Architecture

```
            clients ──► load balancer ──► API replicas (stateless, small image: no OCR / torch)
                                              │  REST · SSE chat · MCP · /metrics
         ┌────────────────────────────────────┼──────────────────────────────┐
         ▼                                    ▼                              ▼
  Elasticsearch                      Redis                          Ingestion workers (N)
  chunk indices per collection       job queue (Streams) ·          parse → chunk → OCR →
  + ingestion ledger                 cache · rate limits ·          embed → bulk index
                                     conversations · locks          (GPU/CPU, scale independently)
```

Code is organised as **ports and adapters**. Orchestrators depend only on the small protocols in
`src/ports/`; `src/core/container.py` is the one place that picks concrete classes.

```
src/
  ports/        Embedder · ChatModel · Parser · OcrEngine · Chunker · IndexWriter · Searcher ·
                DocumentRegistry · QueryExpander · Reranker · Cache · RateLimiter ·
                ConversationStore · JobBackend           (interfaces only)
  core/         config (env) · specs (models + collections) · errors · registry · container
  models/       provider factories (openai, azure_openai, anthropic, google, ollama) + adapters
  parsing/      pdf · image · text · html · csv/xlsx · docx   +   ocr/ (EasyOCR, deskew)
  chunking/     recursive chunker · TF-IDF keywords · content-addressed chunk ids
  indexing/     Elasticsearch adapter (writer, searcher, ledger) · mappings
  retrieval/    hybrid retriever · query expanders · rerankers · cached pipeline
  ingestion/    ingestion service · document service · safe file storage · directory watcher
  jobs/         in-process queue · Redis Streams queue (workers, take-over, locks)
  chat/         answer service · chat service · conversation stores
  runtime/      caches · rate limiters · bulkhead / single-flight · Prometheus metrics
  api/ cli/ mcp_server/ worker.py                        (entry points)
```

How that maps to SOLID, concretely:

* **Single responsibility** – one class per file type, per provider, per backend; the ingestion
  service orchestrates, it does not parse, embed or talk to Elasticsearch itself.
* **Open/closed** – parsers, providers, chunkers, rerankers, caches, queues are looked up by name
  in registries. Adding one never edits an existing module (see [Extending](#extending)).
* **Liskov** – each port has one contract test suite that runs against *every* implementation
  (`tests/integration/test_contracts.py`, `test_jobs.py`).
* **Interface segregation** – writers, searchers and the ingestion ledger are separate ports; so
  are embedders and chat models. A read-only API replica needs none of the write side.
* **Dependency inversion** – no module-level singletons; constructors receive their collaborators.
  Tests swap in fakes (an in-memory `Searcher`, a hash-based `Embedder`) without patching.

---

## Quick start

### Docker Compose (Elasticsearch + Redis + API replicas + worker)

```bash
cp .env.example .env            # set OPENAI_API_KEY (the compose file supplies ES and Redis)
docker compose up --build
docker compose up --scale api=3 --scale worker=2      # scale out: replicas share only ES + Redis
```

### Local development

```bash
python -m venv .venv && source .venv/bin/activate       # Python 3.12+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # or a CUDA index
pip install -e ".[all,dev]"                              # or: pip install -r requirements-dev.txt
cp .env.example .env                                     # ES_CLOUD_ID / ES_HOST, OPENAI_API_KEY
rag-api                                                  # http://localhost:8000/docs
```

Install only what a process needs:

| Image / process | Install | Contains |
|---|---|---|
| API replica | `pip install ".[api,mcp]"` | FastAPI, retrieval, chat, MCP — **~150 MB, no OCR / torch** |
| Ingestion worker | `pip install ".[worker,docx,xlsx]"` | PyMuPDF, OCR, parsers, keyword extraction |
| Everything | `pip install ".[all]"` | the above + CLI + all providers |

Optional provider extras: `anthropic`, `google`, `ollama`. `requirements.txt` is a fully pinned,
universal lock of the tested versions.

---

## Configuration

Two layers:

1. **Environment / `.env`** – infrastructure and limits: Elasticsearch, Redis, credentials,
   concurrency, timeouts. Every variable is documented in [`.env.example`](.env.example).
2. **`RAG_CONFIG` (TOML)** – *what the system is made of*: named chat models, embedding models and
   collections. Without it, one OpenAI chat model, one OpenAI embedding model and one `default`
   collection are built from the classic `OPENAI_*` / `CHUNK_*` / `ES_INDEX_*` variables, so
   existing `.env` files keep working.

```toml
default_chat_model = "fast"
default_collection = "general"

[chat_models.fast]
provider = "openai"
model = "gpt-4o-mini"

[chat_models.claude]
provider = "anthropic"                 # pip install 'turinton-rag[anthropic]'
model = "claude-sonnet-5-5"

[embedding_models.small]
provider = "openai"
model = "text-embedding-3-small"       # dimensions are known; for other models set `dimensions`

[collections.general]
embedding_model = "small"

[collections.legal]                    # its own embedding model, chunking and accepted types
embedding_model = "small"
kinds = ["text", "table"]              # scanned images are not indexed here
parsers = ["pdf", "docx"]
[collections.legal.chunker]
chunk_size = 1200
chunk_overlap = 150
```

See [`config/rag.example.toml`](config/rag.example.toml). **All profiles are built and validated at
start-up**: a bad provider name, a missing API key or a missing optional package stops the process
immediately rather than failing on the first request that uses it. A collection is tied to one
embedding model (its index's vector size depends on it); pointing an existing index at a model of a
different size is refused.

---

## API

Base URL `http://localhost:8000` · interactive docs at `/docs`. Errors are
`{"detail": "...", "code": "not_found"}` with the matching HTTP status.

```bash
# Ingest (default collection). Waits for the result; add ?wait=false for 202 + a job to poll.
curl -F file=@report.pdf http://localhost:8000/api/v1/ingest
curl -F file=@scan.png -F image_language=hi -F collection=legal -F kinds=text,table \
     "http://localhost:8000/api/v1/ingest?force=true"
curl http://localhost:8000/api/v1/jobs/<job_id>

# Retrieve / ask — select collections, kinds, documents, and the chat model per request
curl -X POST http://localhost:8000/api/v1/retrieve -H 'Content-Type: application/json' \
     -d '{"query": "payment terms", "collections": ["legal"], "kinds": ["text", "table"]}'
curl -X POST http://localhost:8000/api/v1/ask -H 'Content-Type: application/json' \
     -d '{"query": "What is the notice period?", "model": "claude"}'

# Chat: omit conversation_id to start; the server returns it
curl -X POST http://localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
     -d '{"message": "Summarise the Q3 results"}'
curl -N -X POST http://localhost:8000/api/v1/chat/stream -H 'Content-Type: application/json' \
     -d '{"message": "and the margin?", "conversation_id": "<id>"}'      # SSE: start / delta / end

# Catalog and housekeeping
curl http://localhost:8000/api/v1/collections          curl http://localhost:8000/api/v1/models
curl "http://localhost:8000/api/v1/documents?collection=legal"
curl -X DELETE "http://localhost:8000/api/v1/documents?source=/data/legal/report.pdf&collection=legal"
curl http://localhost:8000/health    # liveness      curl http://localhost:8000/ready   # dependencies
curl http://localhost:8000/metrics   # Prometheus
```

CLI (`pip install ".[cli]"`): `rag ingest FILE -c legal`, `rag retrieve "…"`, `rag ask "…" -m claude`,
`rag chat`, `rag collections`, `rag models`, `rag documents`, `rag delete SOURCE`.

### MCP

Set `MCP_ENABLED=true` to serve MCP at `/mcp` (stateless Streamable HTTP — any replica can answer
any request). Behind a real domain also set `MCP_ALLOWED_HOSTS=["rag.example.com"]`
(DNS-rebinding protection is on). Tools, all read-only: `list_collections`, `list_models`,
`list_documents`, `search_documents`, `ask_question`.

```jsonc
// Claude Desktop / any stdio MCP client
{ "mcpServers": { "rag": { "command": "rag-mcp", "env": { "ES_HOST": "...", "OPENAI_API_KEY": "..." } } } }
// remote clients: point them at  https://rag.example.com/mcp
```

---

## Scaling

What makes it scale, and where each knob lives:

| Mechanism | Effect | Setting |
|---|---|---|
| Stateless API tier | add replicas freely; no sticky sessions (REST, SSE chat, MCP) | `INGEST_EMBEDDED_WORKER=false` |
| Slim API image | no OCR / torch in replicas → small, fast cold start | `.[api,mcp]` |
| Separate worker tier | OCR/embedding scale independently of query traffic, on GPU or CPU | `rag-worker`, `INGEST_CONCURRENCY` |
| Redis Streams queue | exactly-once hand-out, crash take-over (XAUTOCLAIM), heartbeats, retry cap, back-pressure (`503`/`429` when full) | `INGEST_BACKEND=redis` |
| One search round trip | all query variants × collections × kinds in a single `_msearch`; one embedding call per model | — |
| Async everywhere | Elasticsearch, Redis and model calls never block the event loop; bounded per-dependency concurrency | `ES_MAX_CONCURRENCY`, `MODEL_MAX_CONCURRENCY` |
| Shared caching | retrieval results, query expansions and query embeddings (L1 memory + L2 Redis) with stampede protection; invalidated across replicas on ingest/delete | `CACHE_BACKEND=tiered` |
| Shared limits | one rate limit for the whole fleet, atomic in Redis | `RATE_LIMIT_BACKEND=redis` |
| Load shedding | above N in-flight requests new ones get `503 + Retry-After` instead of queueing into timeouts | `MAX_CONCURRENT_REQUESTS` |
| Bounded everything | Redis connect/read timeouts, per-dependency concurrency caps, an overall deadline on requests *and* streamed answers, uploads refused with `413` before they are spooled, chat history trimmed to a character budget | `REDIS_SOCKET_TIMEOUT_SECONDS`, `REQUEST_TIMEOUT_SECONDS`, `MAX_UPLOAD_MB`, `CHAT_HISTORY_MAX_CHARS` |
| Idempotent indexing | content-addressed chunk ids + write → sweep → ledger commit order: retries and re-ingestion never duplicate, a crash never loses the old version | — |
| Cheap ingestion | streamed parsing, image dedupe (by content), tiny-image skip, deskew-first OCR with early exit, bounded in-flight vectors, no per-request index refresh | `OCR_*`, `INGEST_PIPELINE_DEPTH` |
| Compact vectors | `int8_hnsw` by default; `bbq_hnsw` (~32× smaller) and shortened OpenAI embeddings available | `ES_VECTOR_INDEX_TYPE`, `OPENAI_EMBEDDING_DIMENSIONS` |

Topologies:

* **One process** (development, small teams): `rag-api` with the defaults. Ingestion runs inside
  the API process (`INGEST_BACKEND=inprocess`); run a single server process.
* **Scaled**: Elasticsearch cluster + Redis; N API replicas (`INGEST_EMBEDDED_WORKER=false`,
  `INGEST_BACKEND=redis`, `CACHE_BACKEND=tiered`, `RATE_LIMIT_BACKEND=redis`, `CHAT_STORE=redis`)
  and M `rag-worker` processes. Uploads are written by the API and read by the worker, so
  `DATA_DIR` must be a volume both can see (a shared filesystem; object storage is not built in).
  Scale API replicas on request latency/CPU and workers on `rag_ingest_queue_depth`.
  One server process per container is the scaling unit; use `WEB_CONCURRENCY` > 1 only with the Redis backends.
* **Behind a proxy**: set `FORWARDED_ALLOW_IPS` to the load balancer's addresses so `X-Forwarded-For` is trusted
  *only* from them and rate limits key on the real client (the default trusts localhost only; `*` would let any
  client spoof its IP). Compress responses at the proxy rather than in the app.

Elasticsearch sizing is yours to set (`ES_NUMBER_OF_SHARDS`, `ES_NUMBER_OF_REPLICAS`, or per
collection `shards` / `replicas` / `vector_index_type`). Tested against Elasticsearch **9.5** with the 9.x Python client.

### Measured

`python scripts/benchmark.py` starts the real HTTP stack against a local Elasticsearch with deterministic fake models
(so it measures this system, not an LLM provider) and reports ingestion throughput, retrieval latency/throughput at several
concurrency levels (cache-cold and cache-warm), what each CPU is doing, and the cost of one batched search against the
serial calls of the 1.x design. On a shared 4-core VM with Elasticsearch 9.5 on the same machine:

* **Ingestion** 23 documents/s (460 chunks/s) with one server process, up from 3.9 documents/s before the per-write
  refresh waits were removed; now bound by that one process.
* **Cache-cold retrieval** ~21 requests/s at 24 Elasticsearch searches each - Elasticsearch, not the application,
  is saturated (~3 of 4 cores vs ~0.3). Query expansion and fuzzy matching are the levers: both off gives ~180 requests/s,
  at a recall cost this benchmark cannot measure.
* **Cache-warm retrieval** ~1 000-1 100 requests/s per server process, p50 7-48 ms from 8 to 64 clients.

Details, the limits of these numbers (fake models, tiny corpus, shared machine) and how to reproduce them are in
[`docs/benchmark.md`](docs/benchmark.md). Run it on your own hardware before sizing anything.

---

## Extending

Everything is registered by name. A **plug-in** is any importable module with a
`register(registries)` function, enabled with `PLUGINS=["my_company.rag_plugin"]`:

```python
# my_company/rag_plugin.py
from pathlib import Path
from src.ports.parsing import ParsedUnit, TextBlock

class EmlParser:                                   # a new file type
    name = "eml"
    extensions = frozenset({".eml"})
    def __init__(self, settings): ...              # keep construction cheap; import heavy libs in iter_units
    def iter_units(self, path: Path):
        yield ParsedUnit(unit=1, texts=[TextBlock(path.read_text(), "en")])

def register(registries):
    registries.parsers.register("eml", EmlParser)
    # also: chat_providers, embedding_providers, chunkers, ocr_engines, query_expanders, rerankers,
    #       caches, rate_limiters, conversation_stores, job_backends
```

Then list `"eml"` in a collection's `parsers`, or leave it open to all. No core file changes.
Custom providers receive `(model_id, spec, settings)` and return an object satisfying
`ChatModel` / `Embedder` (`src/ports/models.py`).

---

---

## Operations

* **Probes**: `/health` (liveness, no dependency checks) and `/ready` (Elasticsearch, Redis-backed
  components; `503` listing what failed). Workers expose `/metrics` on `WORKER_METRICS_PORT` (9100).
* **Metrics**: request count/latency by route template, in-flight, shed and rate-limited requests,
  cache hit/miss, upstream latency and errors per dependency, ingestion jobs, queue depth.
* **Logs**: `LOG_FORMAT=json` for structured logs; every line of a request carries its
  `X-Request-ID`. Error responses for dependency failures say only "a backing service failed" — details stay in the log.
* **Shutdown**: SIGTERM stops taking jobs and drains running ones (`SHUTDOWN_GRACE_SECONDS`); a
  killed worker's job is taken over by another after `JOB_VISIBILITY_TIMEOUT_SECONDS`.
* **Ledger**: `doc-registry` records which files are fully indexed (path, checksum, kinds). Unchanged
  files are skipped (`reindexed: false`); `force=true` re-runs. Changing a document replaces its old chunks.

---

## Upgrading from 1.x

**Breaking changes**

* Python **3.12+** (numpy 2.5 needs it). Elasticsearch Python client **9.x** (tested against a 9.5 server).
* Modules moved: `src.services.*`, `src.pipelines.*`, `src.utils.*` are gone; the entry points
  `uvicorn src.api.server:app`, `python -m src.cli`, and the REST paths are unchanged. LangChain
  1.x is used only for model clients and text splitting; the unused `langchain`,
  `langchain-community`, `langchain-elasticsearch`, `unstructured`, `spacy`, `indic-nlp-library`,
  `pdfminer.six`, `sentence-transformers`, `rapidfuzz` and `gunicorn` dependencies are dropped (they were never imported), and
  `slowapi` / `cachetools` are replaced by async in-repo equivalents.
* `POST /ingest` response gained `job_id`, `status`, `status_url`, `collection`, `warnings` and may
  return `202`; `/retrieve` and `/ask` documents gained `collection` and `kind`.
* Failures are no longer disguised: an LLM error is now a `502` (it used to be a `200` whose
  `answer` contained the error text); an unsupported OCR language or file type is rejected.
* The file watcher is **off** by default (`WATCH_DATA_DIR=true` to enable); it used to re-ingest every
  upload a second time, concurrently. `ALLOWED_FILE_EXTENSIONS`, `RETRIEVER_BM25_K1/B` and
  `MAX_WORKERS` no longer exist. `license` metadata now matches the repository's MPL-2.0 `LICENSE`.

**Migrating data**: existing `doc-text` / `doc-tables` / `doc-images` indices keep working (they are
the `default` collection). The first ingest of each file after upgrading re-indexes it (the new
ledger has not seen it) and removes its old chunks. Existing indices gain the new `kind` and
`file_checksum` fields additively; an index whose vector size differs from the embedding model is refused.

**Bugs fixed on the way**: re-ingesting duplicated every chunk; cross-reference chunks always
outranked real hits (score 0.35 vs ~0.01); the "max 2 chunks per page" filter never filtered; the
embedding cache returned the wrong vector for texts sharing a 500-character prefix; failed bulk
writes were logged but reported as success; OCR confidence was always 0.0 (EasyOCR's
`paragraph=True` drops it) so the "pick the better pass" logic never ran; deskew never corrected real
scans; one-chunk pages got no keywords; Devanagari words were shredded in keyword extraction;
chunks started with a stray `.` / `।`; uploads trusted client file names and had no size limit.
The OCR "clean-up" rewrote *correct* text (`learn` → `leam`, `class` → `dass`, `2020` → `2०2०` in Hindi, URLs split into
`www. example. com`, `&`/`%`/`-` deleted, and a Devanagari word-final nukta glued to the next word); CSV cells
containing newlines were merged; XLSX rows after a long blank gap were silently dropped; chat history was sent to
the model without any size limit.

---

## Testing

```bash
pip install -e ".[all,dev]"
pytest tests/unit                      # no services needed
pytest tests/integration               # needs Elasticsearch (RAG_TEST_ES_URL) and Redis (RAG_TEST_REDIS_URL)
ruff check src tests && ruff format --check src tests && mypy src
```

The integration suite runs the real pipeline end to end against live Elasticsearch/Redis with
deterministic fake models (registered through the real plug-in hook): multi-format ingestion,
idempotent re-ingestion, collection isolation, the HTTP API, the job queues including crash
take-over and heartbeats, a multi-replica + separate-worker topology, MCP over HTTP with the SDK's
own client, and the CLI. OCR tests use real EasyOCR weights when available (`RAG_TEST_OCR_MODELS`), including a correctly shaped Hindi
rendering that goes through OCR, ingestion and lexical search. Around 93% of `src/` is covered.

---

## License

Mozilla Public License 2.0 — see [LICENSE](LICENSE).
