from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import settings
from src.core.logger import logger


@dataclass
class QueryExpansion:
    original: str
    expanded: List[str]


class QueryExpansionService:
    def __init__(self) -> None:
        self._client = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
        self._system_prompt = (
            "You rewrite search queries to increase recall. Provide 3 diverse short variations."
        )

    def expand(self, query: str) -> QueryExpansion:
        try:
            response = self._client.invoke(
                [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=f"Query: {query}\nReturn: bullet list of variations."),
                ]
            )
            text = response.content
            candidates = []
            for line in text.splitlines():
                line = line.strip(" -")
                if not line:
                    continue
                candidates.append(line)
            deduped = []
            seen = set()
            for item in candidates:
                normalized = item.lower()
                if normalized not in seen:
                    deduped.append(item)
                    seen.add(normalized)
            if query not in deduped:
                deduped.insert(0, query)
            return QueryExpansion(original=query, expanded=deduped[:4])
        except Exception as exc:  # pragma: no cover
            logger.warning("Query expansion failed via LLM, fallback applied: %s", exc)
            return QueryExpansion(original=query, expanded=[query])


__all__ = ["QueryExpansionService", "QueryExpansion"]
