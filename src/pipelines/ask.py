from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.core.config import settings
from src.pipelines.retrieval import RetrievalPipeline, RetrievalResult


@dataclass
class AskResponse:
    query: str
    expanded_queries: List[str]
    answer: str
    context: List[dict]


ASK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for enterprise knowledge retrieval. Use only the provided context. Cite page numbers or image labels when possible. If unsure, say you do not have enough information.",
        ),
        (
            "user",
            "Original question: {query}\nExpanded queries: {expanded}\n\nContext:\n{context}\n\nReturn a grounded answer in the user's language.",
        ),
    ]
)


class AskPipeline:
    def __init__(self) -> None:
        self.retrieval = RetrievalPipeline()
        self.chat = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)

    def ask(self, query: str) -> AskResponse:
        retrieval: RetrievalResult = self.retrieval.retrieve(query)
        documents = retrieval.documents[: settings.retriever_top_k]
        context_blocks: List[str] = []
        structured: List[dict] = []
        for idx, hit in enumerate(documents, start=1):
            meta = hit.document.metadata
            snippet = hit.document.page_content.strip().replace("\n", " ")
            block = f"[{idx}] Source: {meta.get('source')} | Page: {meta.get('page')} | Type: {meta.get('type')} | Score: {hit.score:.3f}\n{snippet}"
            context_blocks.append(block)
            structured.append(
                {
                    "rank": idx,
                    "score": hit.score,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "keywords": meta.get("keywords"),
                    "type": meta.get("type"),
                    "content": hit.document.page_content,
                }
            )
        messages = ASK_PROMPT.format_messages(
            query=query,
            expanded=", ".join(retrieval.expanded_queries),
            context="\n\n".join(context_blocks) or "No context available.",
        )
        result = self.chat.invoke(messages)
        return AskResponse(
            query=query,
            expanded_queries=retrieval.expanded_queries,
            answer=result.content,
            context=structured,
        )


__all__ = ["AskPipeline", "AskResponse"]
