from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import settings
from src.core.logger import logger


@dataclass
class QueryExpansion:
    original: str
    expanded: List[str]


class QueryExpansionService:
    """Query expansion using OpenAI to improve retrieval recall."""
    
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for query expansion")
        
        self._client = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.7,  # Some creativity for variations
            max_tokens=200,
        )
        self._system_prompt = (
            "You are a search query rewriter. Given a user query, generate 3 diverse "
            "alternative phrasings to improve search recall. Keep variations concise. "
            "Return only the variations as a bullet list, one per line."
        )

    def expand(self, query: str) -> QueryExpansion:
        """Expand a query into multiple variations for better retrieval."""
        try:
            response = self._client.invoke(
                [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=f"Query: {query}"),
                ]
            )
            text = response.content
            
            # Parse bullet points
            candidates = []
            for line in text.splitlines():
                line = line.strip(" -•*")
                if not line or len(line) < 3:
                    continue
                candidates.append(line)
            
            # Deduplicate
            deduped = []
            seen = set()
            for item in candidates:
                normalized = item.lower().strip()
                if normalized not in seen and normalized != query.lower().strip():
                    deduped.append(item)
                    seen.add(normalized)
            
            # Always include original query first
            result = [query] + deduped[:3]
            return QueryExpansion(original=query, expanded=result)
            
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
            return QueryExpansion(original=query, expanded=[query])


__all__ = ["QueryExpansionService", "QueryExpansion"]
