from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from src.cli.evaluate import eval_app
from src.core.config import get_settings
from src.core.container import Container, Role
from src.core.errors import RagError
from src.core.kinds import parse_kinds
from src.core.logger import configure_logging
from src.core.types import CONTENT_KINDS, JobSpec

app = typer.Typer(help="Command line tools for the RAG pipeline.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

Collection = Annotated[
    str | None, typer.Option("--collection", "-c", help="Collection (default: the default collection)")
]
Kinds = Annotated[
    list[str] | None,
    typer.Option("--kind", "-k", help=f"Restrict to content kind: {', '.join(CONTENT_KINDS)}"),
]
Model = Annotated[str | None, typer.Option("--model", "-m", help="Named chat model")]


def _run(coroutine_factory, *, role: Role = "cli", with_ingestion: bool = False):
    """Build the container, run one command, always close it; typed errors become exit code 1."""

    async def main():
        settings = get_settings()
        configure_logging("WARNING")
        container = await Container.build(settings, role=role, with_ingestion=with_ingestion)
        try:
            await container.start()
            return await coroutine_factory(container)
        finally:
            await container.close()

    try:
        return asyncio.run(main())
    except RagError as exc:
        typer.secho(f"{exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    collection: Collection = None,
    force: Annotated[bool, typer.Option("--force", help="Re-index even if unchanged.")] = False,
    image_language: Annotated[
        str | None, typer.Option("--image-language", "-l", help="OCR language: en, mr, hi or auto")
    ] = None,
    kind: Kinds = None,
) -> None:
    """Ingest a file (PDF, image, text, HTML, CSV, DOCX, XLSX, ...) into a collection."""

    async def go(container: Container) -> None:
        name = collection or container.config.default_collection
        spec = JobSpec(
            collection=name,
            path=str(path),
            force=force,
            image_language=image_language,
            kinds=parse_kinds(kind),
        )
        summary = await container.ingestion.ingest(spec)  # type: ignore[union-attr]
        if not summary.reindexed:
            typer.echo(f"No changes detected for {path}; skipped.")
            return
        typer.echo(
            f"Indexed {path} into '{name}' -> text: {summary.text_chunks}, tables: {summary.table_chunks}, "
            f"images: {summary.image_chunks}"
        )
        for warning in summary.warnings:
            typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)

    _run(go, with_ingestion=True)


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="Natural language query.")],
    collection: Annotated[list[str] | None, typer.Option("--collection", "-c")] = None,
    kind: Kinds = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Limit number of documents returned.")] = None,
) -> None:
    """Hybrid BM25 + vector search."""

    async def go(container: Container) -> None:
        result = await container.retrieval.retrieve(query, container.retrieval.scope(collection, kind))
        typer.echo(f"Expanded queries: {', '.join(result.expanded_queries)}")
        for index, doc in enumerate(result.documents[:limit], start=1):
            meta = doc.metadata
            typer.echo(
                f"[{index}] score={doc.final_score:.4f} collection={doc.collection} source={meta.get('source')} "
                f"page={meta.get('page')} type={meta.get('type')}\n{doc.content[:400]}\n"
            )

    _run(go)


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Question to ask over indexed knowledge.")],
    collection: Annotated[list[str] | None, typer.Option("--collection", "-c")] = None,
    kind: Kinds = None,
    model: Model = None,
) -> None:
    """Answer a question from retrieved context."""

    async def go(container: Container) -> None:
        result = await container.answers.ask(query, container.retrieval.scope(collection, kind), model)
        typer.echo(f"Model: {result.model}\nExpanded queries: {', '.join(result.expanded_queries)}")
        typer.echo("\nAnswer:\n" + result.answer + "\n\nSources:")
        for index, doc in enumerate(result.documents, start=1):
            meta = doc.metadata
            typer.echo(f"[{index}] {meta.get('source')} p.{meta.get('page')} ({doc.collection}/{doc.kind})")

    _run(go)


@app.command()
def chat(
    collection: Annotated[list[str] | None, typer.Option("--collection", "-c")] = None,
    kind: Kinds = None,
    model: Model = None,
) -> None:
    """Interactive multi-turn chat (empty line or Ctrl-D to quit)."""

    async def go(container: Container) -> None:
        scope = container.retrieval.scope(collection, kind)
        conversation_id: str | None = None
        while True:
            try:
                message = await asyncio.to_thread(input, "you> ")
            except EOFError:
                return
            if not message.strip():
                return
            result = await container.chat.chat(message, scope, conversation_id=conversation_id, model=model)
            conversation_id = result.conversation_id
            typer.echo(f"\n{result.answer}\n")

    _run(go)


@app.command()
def collections() -> None:
    """List collections."""

    async def go(container: Container) -> None:
        config = container.config
        for name, spec in sorted(config.collections.items()):
            marker = "*" if name == config.default_collection else " "
            typer.echo(
                f"{marker} {name:20} model={spec.embedding_model:16} kinds={','.join(spec.kinds)}  {spec.description}"
            )

    _run(go)


@app.command()
def models() -> None:
    """List chat and embedding models."""

    async def go(container: Container) -> None:
        config = container.config
        for name, spec in sorted(config.chat_models.items()):
            typer.echo(
                f"chat       {'*' if name == config.default_chat_model else ' '} {name:20} {spec.provider}/{spec.model}"
            )
        for name, embedding_spec in sorted(config.embedding_models.items()):
            dims = container.models.embedder(name).dimensions
            typer.echo(
                f"embedding    {name:20} {embedding_spec.provider}/{embedding_spec.model} ({dims} dims)"
            )

    _run(go)


@app.command()
def documents(collection: Collection = None, limit: int = 50, offset: int = 0) -> None:
    """List documents ingested into a collection."""

    async def go(container: Container) -> None:
        name = collection or container.config.default_collection
        records, total = await container.documents.list(name, limit=limit, offset=offset)
        typer.echo(f"{total} document(s) in '{name}'")
        for r in records:
            typer.echo(
                json.dumps(
                    {
                        "source": r.source,
                        "parser": r.parser,
                        "kinds": r.kinds,
                        "chunks": r.text_chunks + r.table_chunks + r.image_chunks,
                    }
                )
            )

    _run(go)


@app.command()
def delete(
    source: Annotated[str, typer.Argument(help="Source path as shown by `documents`")],
    collection: Collection = None,
) -> None:
    """Remove a document's chunks from a collection."""

    async def go(container: Container) -> None:
        name = collection or container.config.default_collection
        typer.echo(
            f"Deleted {await container.documents.delete(name, source)} chunk(s) of {source} from '{name}'"
        )

    _run(go)


if __name__ == "__main__":
    app()
