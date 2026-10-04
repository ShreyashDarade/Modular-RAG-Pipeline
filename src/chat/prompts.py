from __future__ import annotations

from collections.abc import Sequence

from src.core.types import ChatMessage, RetrievedDocument

ANSWER_SYSTEM = """You are a helpful assistant for enterprise knowledge retrieval.

Instructions:
- Use ONLY the provided context to answer questions
- Cite sources using [Source: filename, Page: X] format
- If the context contains tables, reference the table data specifically
- If information comes from an image/OCR, mention that
- Support Hindi (हिंदी) and Marathi (मराठी) - respond in the same language as the question
- If you cannot find the answer in the context, say so clearly
- Be concise but thorough"""

CONDENSE_SYSTEM = (
    "Rewrite the user's latest message as a single standalone search query, using the conversation "
    "for any missing context (pronouns, 'it', 'that one', ...). Keep the language of the message. "
    "Return only the query, nothing else."
)


def format_context(documents: Sequence[RetrievedDocument]) -> str:
    if not documents:
        return "No relevant context found."
    blocks = []
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata
        header = (
            f"[{index}] Source: {meta.get('source', 'Unknown')} | Page: {meta.get('page', 'N/A')} | "
            f"Type: {meta.get('type') or meta.get('content_type') or doc.kind} | Collection: {doc.collection}"
        )
        extra = ""
        if meta.get("table_summary"):
            extra += f"\nTable Info: {meta['table_summary']}"
        if meta.get("ocr_confidence"):
            extra += f"\nOCR Confidence: {meta['ocr_confidence']:.2f}"
        blocks.append(f"{header}{extra}\n{doc.content.strip()}")
    return "\n\n".join(blocks)


def answer_messages(
    query: str,
    expanded: Sequence[str],
    documents: Sequence[RetrievedDocument],
    history: Sequence[ChatMessage] = (),
) -> list[ChatMessage]:
    user = (
        f"Question: {query}\n\n"
        f"Related queries considered: {', '.join(expanded) if expanded else 'None'}\n\n"
        f"Retrieved Context:\n{format_context(documents)}\n\n"
        "Please provide a comprehensive answer based on the context above."
    )
    return [ChatMessage("system", ANSWER_SYSTEM), *history, ChatMessage("user", user)]


def condense_messages(history: Sequence[ChatMessage], message: str, turns: int = 6) -> list[ChatMessage]:
    transcript = "\n".join(f"{m.role}: {m.content[:600]}" for m in history[-turns:])
    return [
        ChatMessage("system", CONDENSE_SYSTEM),
        ChatMessage("user", f"Conversation:\n{transcript}\n\nLatest message: {message}"),
    ]
