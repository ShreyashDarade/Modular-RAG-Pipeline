from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.config import settings


@dataclass
class ChunkedDocument:
    document: Document
    keywords: List[str]


class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def split(self, text: str, metadata: dict | None = None) -> List[Document]:
        return self.splitter.create_documents([text], metadatas=[metadata or {}])


class KeywordExtractor:
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.keyword_top_k

    def extract(self, texts: Sequence[str]) -> List[List[str]]:
        if not texts:
            return [[]]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.top_k)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        keywords_per_doc: List[List[str]] = []
        for doc_index in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(doc_index)
            sorted_indices = row.toarray().flatten().argsort()[::-1]
            keywords = [feature_names[idx] for idx in sorted_indices if row[0, idx] > 0]
            keywords_per_doc.append(keywords[: self.top_k])
        return keywords_per_doc


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.embedder = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_documents(self, documents: Sequence[Document]) -> List[List[float]]:
        texts = [doc.page_content for doc in documents]
        return self.embedder.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)


__all__ = ["TextChunker", "KeywordExtractor", "EmbeddingService", "ChunkedDocument"]
