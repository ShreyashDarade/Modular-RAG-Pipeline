# RAG Pipeline

RAG is a retrieval-augmented generation stack that ingests PDFs and images, stores multi-modal embeddings in Elasticsearch, expands user queries, and streams grounded answers through Ollama-hosted large language models. The project exposes both a FastAPI service and a Typer-powered CLI so you can automate ingestion or run question-answering locally.

- Hybrid search that fuses BM25 and vector (kNN) scoring across dedicated text, table, and image indexes.
- Automated OCR for English and Marathi sources (EasyOCR) with dynamic preprocessing and language heuristics.
- Query expansion and answer generation via ChatOllama, configurable through environment variables.
- File-change watcher that reindexes documents stored in the shared `data/` directory.
- Reusable ingestion, retrieval, and ask pipelines for embedding, storing, and querying your knowledge base.

## Project Description

RAG is a production-ready retrieval augmented generation platform designed to help enterprises unlock value hidden in scattered PDF manuals, scanned documents, and multilingual archives. The system ingests files, converts every relevant signal into structured representations, and delivers grounded answers through a unified interface.

During ingestion each document travels through a multi-stage preparation workflow. PDFs are parsed page by page, table structures become Markdown, and inline images route through a resilient OCR stack that balances EasyOCR accuracy with custom preprocessing heuristics. The outputs are chunked, enriched with TF-IDF keywords, vectorized with Sentence Transformers, and stored in dedicated Elasticsearch indices for text, table, and image modalities. Checksum tracking ensures that only genuinely modified files are reprocessed, while a watchdog monitors the data directory and asynchronously triggers reindexing when new evidence appears.

Retrieval blends semantic and lexical relevance to improve recall without sacrificing precision. Queries are first expanded by an Ollama-hosted language model to cover alternate phrasings or language switches; then the HybridRetriever fuses BM25 and kNN scores per index and normalizes them into a single ranking. The Ask pipeline layers a prompt-engineered LLM on top, formatting citations and guaranteeing that generated answers stay grounded within the retrieved context. Each response includes the expanded queries and ordered snippets so downstream consumers can trace the reasoning.

Operations teams orchestrate RAG through multiple touch points. A FastAPI service exposes REST endpoints for ingestion, retrieval, and question answering, making integration with existing portals or chat assistants straightforward. Power users can turn to the Typer CLI for quick experiments, one-off indexing jobs, or scripted batch loads. Configuration lives in environment variables and the `Settings` model, enabling deployments ranging from a single laptop to a containerized cluster. With Ollama and Elasticsearch running locally, teams prototype data copilots within minutes while retaining full control over sensitive documents.

## Repository Layout

```
.
├── src/
│   ├── api/            # FastAPI app, pydantic schemas, and server bootstrap
│   ├── cli/            # Typer CLI (`python -m src.cli`) for ingest/retrieve/ask
│   ├── pipelines/      # Ingestion, retrieval, and ask orchestration layers
│   ├── services/       # Embedding, OCR, Elasticsearch, query expansion, watcher
│   └── utils/          # PDF parsing, language detection, image preprocessing, I/O helpers
├── data/               # Local document staging area, `.index_state.json` checksum cache
├── models/             # EasyOCR model cache (created automatically)
├── requirements.txt    # Runtime dependencies
├── pyproject.toml      # Packaging metadata
└── README.md
```

> The `devanagari_finetuned.pth` weight file ships with the repository so Marathi OCR can initialize without an extra download.

## Prerequisites

- Python >= 3.10
- pip (or pipx) for dependency management
- [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) 8.x running on `http://localhost:9200`
- [Ollama](https://ollama.com/download) with a chat model that matches `OLLAMA_MODEL` (defaults to `gpt-oss:20b`)
- System packages for OCR: `libgl1`, `ffmpeg`, and `tesseract-ocr` are typically required when installing OpenCV/EasyOCR on Linux

## Installation

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # Optional extras for CLI or API entrypoints
   pip install .[cli,api]
   ```

3. **Configure environment**

   Copy the sample snippet below into `.env` (or export the variables before running the app):

   ```ini
   # .env
   DATA_DIR=./data
   ES_HOST=http://localhost:9200
   ES_USERNAME=elastic            # leave blank if security is disabled
   ES_PASSWORD=changeme
   OLLAMA_MODEL=gpt-oss:20b
   OLLAMA_BASE_URL=http://localhost:11434
   RETRIEVER_TOP_K=6
   HYBRID_ALPHA=0.6
   ```

   Any variable omitted falls back to the default declared in `src/core/config.py`.

## Running the Stack

- **Elasticsearch** (single node for local testing):

  ```bash
  docker run -d --name -es \
    -p 9200:9200 -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.12.2
  ```

- **Ollama**: install from <https://ollama.com/download>, run `ollama serve`, then pull the configured model. Example:

  ```bash
  ollama pull gpt-oss:20b
  ```

Ensure both services are reachable before attempting ingestion or question answering.

## Ingesting Documents

You can ingest PDFs or supported images (PNG, JPG, TIFF) either from disk or through the API. The ingestion pipeline splits content into text/table/image chunks, extracts TF-IDF keywords, embeds each chunk, and writes to separate Elasticsearch indexes.

```bash
python -m src.cli ingest path/to/document.pdf
python -m src.cli ingest path/to/scan.jpg --image-language mr   # optional OCR hint
python -m src.cli ingest path/to/document.pdf --force           # skip checksum guard
```

Successful ingestion prints chunk counts. If nothing changed, the CLI exits early using `.index_state.json` to compare file checksums.

### OCR Notes

- English (`en`) and Marathi (`mr`) recognition use EasyOCR with custom preprocessing (`src/utils/image_ops.py`).
- Leaving `--image-language` unspecified triggers automatic language detection and script heuristics.
- OCR output is chunked like text pages and stored in the image index so downstream retrieval can surface visual snippets.

## Retrieving Knowledge via CLI

The hybrid retriever expands the query through ChatOllama, performs BM25 and kNN searches per index, fuses the scores, and returns the top K results with metadata.

```bash
python -m src.cli retrieve "What is the warranty policy?"
python -m src.cli retrieve "लायसन्स फी" --limit 3
```

## Asking Questions (RAG)

```bash
python -m src.cli ask "Summarize the onboarding checklist."
```

This command:

1. Expands the original query (3 variants by default).
2. Runs the hybrid retriever and selects cross-index hits.
3. Formats context snippets with source metadata.
4. Calls the configured Ollama chat model using the grounded prompt in `src/pipelines/ask.py`.

Answers are streamed back with the context items that support them.

## FastAPI Service

Start the HTTP server:

```bash
python -m src.api.main
# or uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints (see `src/api/schemas.py` for payloads):

| Method | Path                 | Description                                                         |
|--------|----------------------|---------------------------------------------------------------------|
| POST   | `/api/v1/ingest`     | Upload and index a file (`multipart/form-data`, optional `image_language`) |
| POST   | `/api/v1/retrieve`   | Return expanded queries and the ranked hybrid hits                  |
| POST   | `/api/v1/ask`        | Produce a full RAG answer with supporting context                   |

On startup the server also launches a `DataDirectoryWatcher` (`watchdog`) that reindexes any file dropped into `data/`.

## Configuration Reference

Key settings from `src/core/config.py` (all overrideable via env vars):

| Variable               | Default                     | Purpose |
|------------------------|-----------------------------|---------|
| `DATA_DIR`             | `./data`                    | Directory where ingested files and index state live |
| `ES_HOST`              | `http://localhost:9200`     | Elasticsearch connection string |
| `ES_INDEX_TEXT`        | `doc-text`                  | Index for PDF text chunks |
| `ES_INDEX_TABLES`      | `doc-tables`                | Index for table markdown chunks |
| `ES_INDEX_IMAGES`      | `doc-images`                | Index for OCR image text |
| `EMBEDDING_MODEL`      | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding backbone |
| `CHUNK_SIZE`           | `800`                       | Character window for recursive chunking |
| `CHUNK_OVERLAP`        | `80`                        | Overlap between chunks |
| `RETRIEVER_TOP_K`      | `6`                         | Number of hits returned per query |
| `HYBRID_ALPHA`         | `0.6`                       | Blend ratio between BM25 and kNN scores |
| `OLLAMA_MODEL`         | `gpt-oss:20b`               | Chat model name used for expansion & answering |
| `OLLAMA_BASE_URL`      | `http://localhost:11434`    | Ollama HTTP endpoint |
| `ALLOWED_FILE_EXTENSIONS` | `.pdf,.png,.jpg,.jpeg,.tif,.tiff` | Controlled by ingestion guard |

## Development Tips

- Logs are configured in `src/core/logger.py` and default to INFO level. Set `LOG_LEVEL=DEBUG` before running to enable verbose output.
- Unit tests are not bundled; consider adding pytest suites around `src/pipelines/` for production deployments.
- The ingestion pipeline uses the first document to probe embedding dimensionality. If you swap the embedding model, clear your indexes to avoid dimension mismatches.

## Troubleshooting

- **`ValueError: The current process has no default GPU...`** — EasyOCR defaults to CPU; ensure `gpu=False` stays configured or install CUDA dependencies if you want GPU acceleration.
- **`elasticsearch.ApiError: illegal_argument_exception`** — Your cluster does not support `index.knn`. The indexer automatically retries without kNN; confirm the fallback created the index.
- **No new chunks after re-ingest** — Delete the cached checksum in `data/.index_state.json` or use the `--force` flag.
- **Slow OCR on large images** — Pre-resize or compress scans; the pipeline already applies normalization, but very large TIFFs still take time.

Once ingestion succeeds, you can iterate quickly by editing files inside `data/`; the watcher will push updates without restarting the app.
