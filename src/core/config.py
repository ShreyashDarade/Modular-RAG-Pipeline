from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    app_name: str = "OCR-rag"
    environment: str = "local"
    data_dir: Path = Path("data").resolve()

    # ElasticSearch
    es_host: str = "http://localhost:9200"
    es_username: str | None = None
    es_password: str | None = None
    es_index_text: str = "doc-text"
    es_index_tables: str = "doc-tables"
    es_index_images: str = "doc-images"

    # Embedding / chunking
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 800
    chunk_overlap: int = 80
    keyword_top_k: int = 20

    # Retrieval settings
    retriever_top_k: int = 6
    retriever_bm25_k1: float = 1.5
    retriever_bm25_b: float = 0.75
    hybrid_alpha: float = 0.6

    # Ollama
    ollama_model: str = "gpt-oss:20b"
    ollama_base_url: str = "http://localhost:11434"

    allowed_file_extensions: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
