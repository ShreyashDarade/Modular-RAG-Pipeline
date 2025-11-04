from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Optional

import typer

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _ensure_utf8_stream(stream: Optional[io.TextIOBase]) -> None:
    if stream is None:
        return
    encoding = getattr(stream, "encoding", None)
    if encoding and encoding.lower() == "utf-8":
        return
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            wrapper = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")
            wrapper.flush()
            if stream is sys.stdout:
                sys.stdout = wrapper
            elif stream is sys.stderr:
                sys.stderr = wrapper


_ensure_utf8_stream(sys.stdout)
_ensure_utf8_stream(sys.stderr)

from src.pipelines.ask import AskPipeline
from src.pipelines.ingestion import IngestionPipeline
from src.pipelines.retrieval import RetrievalPipeline
from src.utils.language import SUPPORTED_LANGS

app = typer.Typer(help="CLI tools for RAG pipeline.")

_ingestion = IngestionPipeline()
_retrieval = RetrievalPipeline()
_ask = AskPipeline()


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    force: bool = typer.Option(False, "--force", help="Force reindex even if unchanged."),
    image_language: Optional[str] = typer.Option(
        None,
        "--image-language",
        "-l",
        help="Optional OCR language hint for images (en, mr, auto).",
    ),
) -> None:
    """Ingest a PDF or image into ElasticSearch."""
    normalized_lang: Optional[str] = None
    if image_language:
        candidate = image_language.strip().lower()
        if candidate not in {"", "auto", "none"} and candidate not in SUPPORTED_LANGS:
            raise typer.BadParameter(f"Unsupported language '{image_language}'. Use one of {', '.join(SUPPORTED_LANGS)} or 'auto'.")
        if candidate not in {"", "auto", "none"}:
            normalized_lang = candidate
    summary = _ingestion.ingest_path(path, force=force, image_language=normalized_lang)
    if not summary.reindexed:
        typer.echo(f"No changes detected for {path}; skipped.")
        return
    typer.echo(
        f"Indexed {path} -> text: {summary.text_chunks}, tables: {summary.table_chunks}, images: {summary.image_chunks}"
    )


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Natural language query."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of documents returned."),
) -> None:
    """Retrieve documents for a query using hybrid BM25 + kNN search."""
    result = _retrieval.retrieve(query)
    documents = result.documents if limit is None else result.documents[:limit]
    typer.echo(f"Expanded queries: {', '.join(result.expanded_queries)}")
    for idx, hit in enumerate(documents, start=1):
        meta = hit.document.metadata
        typer.echo(
            f"[{idx}] score={hit.score:.3f} source={meta.get('source')} page={meta.get('page')} type={meta.get('type')}\n{hit.document.page_content[:400]}\n"
        )


@app.command()
def ask(query: str = typer.Argument(..., help="Question to ask over indexed knowledge.")) -> None:
    """Run the full RAG ask pipeline with LLM generation."""
    response = _ask.ask(query)
    typer.echo(f"Expanded queries: {', '.join(response.expanded_queries)}")
    typer.echo("\nAnswer:\n" + response.answer + "\n")
    typer.echo("Context snippets:")
    for item in response.context:
        typer.echo(
            f"[{item['rank']}] score={item['score']:.3f} source={item['source']} page={item['page']} type={item['type']}\n{item['content'][:400]}\n"
        )


if __name__ == "__main__":
    app()
