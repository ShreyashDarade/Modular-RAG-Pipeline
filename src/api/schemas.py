from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    source: str
    text_chunks: int = Field(ge=0)
    table_chunks: int = Field(ge=0)
    image_chunks: int = Field(ge=0)
    skipped_reason: Optional[str] = None
    reindexed: bool = True


class RetrieveRequest(BaseModel):
    query: str


class RetrievedDocumentSchema(BaseModel):
    content: str
    score: float
    source: str | None = None
    page: int | None = None
    type: str | None = None
    keywords: List[str] | None = None


class RetrieveResponse(BaseModel):
    query: str
    expanded_queries: List[str]
    documents: List[RetrievedDocumentSchema]


class AskRequest(BaseModel):
    query: str


class AskContextItem(BaseModel):
    rank: int
    score: float
    source: str | None = None
    page: int | None = None
    type: str | None = None
    keywords: List[str] | None = None
    content: str


class AskResponseSchema(BaseModel):
    query: str
    expanded_queries: List[str]
    answer: str
    context: List[AskContextItem]

