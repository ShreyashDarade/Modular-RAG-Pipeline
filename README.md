# 🚀 RAG-OCR Pipeline

**Retrieval-Augmented Generation (RAG)** pipeline with **multilingual OCR** support for **Hindi, Marathi, and English**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple.svg)](https://openai.com/)

---

## ✨ Features

### 🔍 **Advanced OCR**

- **Multilingual Support**: Hindi (हिंदी), Marathi (मराठी), English
- **GPU Acceleration**: Auto-detects CUDA for faster processing
- **Image Preprocessing**: Deskewing, denoising, contrast enhancement
- **NLP Post-processing**: Script-aware text cleaning

### 📄 **Document Intelligence**

- **PDF Processing**: Text, tables, and embedded images
- **Table Extraction**: Preserves structure as Markdown
- **Cross-References**: Links text ↔ tables ↔ images on same page

### 🔎 **Hybrid Retrieval**

- **BM25 + Vector Search**: Reciprocal Rank Fusion (RRF)
- **Re-Ranking**: Multi-signal relevance scoring
- **Cross-Reference Expansion**: Fetches related chunks automatically
- **Diversity Selection**: Ensures varied results

### 🤖 **OpenAI Integration**

- **LLM**: GPT-4o-mini for Q&A
- **Embeddings**: text-embedding-3-small (1536 dimensions)
- **Context-Aware Answers**: Cites sources with page numbers

---

## 📋 Prerequisites

- **Python 3.10+**
- **Elasticsearch 8.x**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **CUDA GPU** (optional, for faster OCR)

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ShreyashDarade/RAG-OCR-Pipeline.git
cd RAG-OCR-Pipeline
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 5. Setup Elasticsearch

**Option A: Elasticsearch Cloud (Recommended for Production)**

1. Sign up at [cloud.elastic.co](https://cloud.elastic.co/)
2. Create a deployment
3. Get your **Cloud ID** and **API Key**
4. Add to `.env`:

```env
ES_CLOUD_ID=my-deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGFiYzEyMyQ...
ES_API_KEY=your-api-key-here
```

**Option B: Local Docker (Development)**

```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

**Option C: Self-hosted with Authentication**

```env
ES_HOST=https://your-elasticsearch-server:9200
ES_USERNAME=elastic
ES_PASSWORD=your-password
```

### 6. Create Required Directories

```bash
mkdir data models
```

### 7. Run the Application

```bash
# Development
uvicorn src.api.server:app --reload --port 8000

# Production
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🌐 API Endpoints

Base URL: `http://localhost:8000`

### Health Check

```bash
curl http://localhost:8000/health
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Ingest Document

```bash
# Auto-detect language
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@document.pdf"

# Specify Hindi
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@document.pdf" \
  -F "image_language=hi"

# Specify Marathi
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@document.png" \
  -F "image_language=mr"
```

### Retrieve Documents

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query"}'
```

### Ask Question

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the document about?"}'
```

---

## 🗂️ Project Structure

```
RAG-OCR-Pipeline/
├── src/
│   ├── api/
│   │   ├── server.py          # FastAPI application with rate limiting
│   │   └── schemas.py         # Request/response models
│   ├── core/
│   │   ├── config.py          # Configuration (OpenAI, ES, etc.)
│   │   └── logger.py          # Logging setup
│   ├── pipelines/
│   │   ├── ingestion.py       # Document ingestion with cross-refs
│   │   ├── retrieval.py       # Query processing
│   │   └── ask.py             # Q&A with OpenAI
│   ├── services/
│   │   ├── ocr.py             # Multilingual OCR (EasyOCR)
│   │   ├── embedding.py       # OpenAI embeddings
│   │   ├── elastic.py         # Elasticsearch client
│   │   └── retrieval.py       # Hybrid retriever with RRF
│   └── utils/
│       ├── image_ops.py       # Image preprocessing + deskewing
│       ├── language.py        # Language detection
│       ├── nlp_processing.py  # NLP post-processing
│       └── pdf_parser.py      # PDF extraction
├── data/                      # Uploaded documents (gitignored)
├── models/                    # OCR model weights (gitignored)
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .gitignore
└── README.md
```

---

## 🌍 Supported Languages

| Code | Language | Script     | OCR Support |
| ---- | -------- | ---------- | ----------- |
| `en` | English  | Latin      | ✅ Full     |
| `hi` | Hindi    | Devanagari | ✅ Full     |
| `mr` | Marathi  | Devanagari | ✅ Full     |

---

## ⚙️ Configuration

### Required Environment Variables

| Variable         | Description    | Example       |
| ---------------- | -------------- | ------------- |
| `OPENAI_API_KEY` | OpenAI API key | `sk-xxxxxxxx` |

**Elasticsearch (choose one):**

| Variable      | Description            | Example                      |
| ------------- | ---------------------- | ---------------------------- |
| `ES_CLOUD_ID` | Elasticsearch Cloud ID | `deployment:base64string...` |
| `ES_API_KEY`  | Elasticsearch API key  | `your-api-key`               |
| `ES_HOST`     | Self-hosted ES URL     | `http://localhost:9200`      |

### Optional Configuration

| Variable                  | Default                  | Description              |
| ------------------------- | ------------------------ | ------------------------ |
| `OPENAI_MODEL`            | `gpt-4o-mini`            | LLM model for Q&A        |
| `OPENAI_EMBEDDING_MODEL`  | `text-embedding-3-small` | Embedding model          |
| `CHUNK_SIZE`              | `800`                    | Text chunk size          |
| `CHUNK_OVERLAP`           | `200`                    | Overlap between chunks   |
| `ENABLE_CROSS_REFERENCES` | `true`                   | Link text/tables/images  |
| `RERANK_ENABLED`          | `true`                   | Enable result re-ranking |
| `RERANK_TOP_K`            | `6`                      | Final results count      |
| `OCR_GPU_ENABLED`         | `true`                   | Use GPU for OCR          |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ /ingest  │  │/retrieve │  │  /ask    │  │ /health /ready   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────────┘ │
└───────┼─────────────┼─────────────┼─────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌───────────────┐ ┌─────────────────────┐ ┌───────────────────────┐
│   Ingestion   │ │   Hybrid Retriever  │ │     Ask Pipeline      │
│   Pipeline    │ │                     │ │                       │
│ ┌───────────┐ │ │ ┌───────┐ ┌───────┐ │ │ ┌─────────────────┐   │
│ │ PDF Parse │ │ │ │ BM25  │ │  KNN  │ │ │ │  OpenAI GPT-4o  │   │
│ │ OCR (GPU) │ │ │ └───┬───┘ └───┬───┘ │ │ └────────┬────────┘   │
│ │ Chunking  │ │ │     │   RRF   │     │ │          │            │
│ │ Cross-Ref │ │ │     └────┬────┘     │ │          ▼            │
│ └─────┬─────┘ │ │     ┌────▼────┐     │ │   Context + Answer    │
│       │       │ │     │ Re-Rank │     │ │                       │
└───────┼───────┘ │     │Diversify│     │ └───────────────────────┘
        │         │     └─────────┘     │
        ▼         └──────────┬──────────┘
┌───────────────────────────┐│
│      Elasticsearch        ││
│  ┌────────┐ ┌────────┐    ││
│  │doc-text│ │doc-tbl │    │◄───── Query Vector
│  └────────┘ └────────┘    ││
│  ┌────────┐               ││
│  │doc-img │ (with vectors)││
│  └────────┘               ││
└───────────────────────────┘│
        ▲                    │
        │  OpenAI Embeddings │
        └────────────────────┘
```

---

## 🔧 Troubleshooting

| Issue                          | Solution                                            |
| ------------------------------ | --------------------------------------------------- |
| `OPENAI_API_KEY is required`   | Add your API key to `.env`                          |
| Elasticsearch connection error | Check if ES is running: `curl localhost:9200`       |
| GPU not detected               | Install CUDA toolkit or set `OCR_GPU_ENABLED=false` |
| Rate limit exceeded            | Wait 1 minute or increase `RATE_LIMIT_PER_MINUTE`   |
| Hindi/Marathi OCR poor quality | Use high-resolution images (300+ DPI)               |

---

## 📈 Performance Tips

1. **Use GPU**: 5-10x faster OCR with CUDA
2. **Increase Workers**: Set `MAX_WORKERS=8` for more concurrency
3. **Enable Caching**: Set `USE_REDIS_CACHE=true` with Redis
4. **Tune Chunk Size**: Smaller chunks = more precise, larger = more context
5. **Pre-warm Models**: First request loads models; subsequent are faster

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Multilingual OCR
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Elasticsearch](https://www.elastic.co/) - Vector search engine
