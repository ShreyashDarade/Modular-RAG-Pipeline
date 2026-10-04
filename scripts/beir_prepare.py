#!/usr/bin/env python
"""Turn a BEIR retrieval benchmark (SciFact, NFCorpus, ...) into an evaluation setup for this pipeline:
documents ingested into a collection, an evaluation dataset (`cases.jsonl`) built from the benchmark's
relevance judgements, and a `rag.toml` using local models (no API key, no network once the models are
cached).

    pip install -e '.[dev,local,worker]'
    python scripts/beir_prepare.py --name scifact --work /tmp/beir
    export RAG_CONFIG=/tmp/beir/scifact/rag.toml OPENAI_API_KEY=unused    # the chat model is never called
    rag eval sweep /tmp/beir/scifact/cases.jsonl --granularity document -c scifact \\
        -v hybrid -v ce:reranker=ce

Relevance is document-level, so evaluate with ``--granularity document``. The published BEIR numbers
(nDCG@10 for SciFact: BM25 0.665, BM25 + ms-marco MiniLM cross-encoder 0.688) make a useful sanity check
that the harness and the pipeline behave as expected.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

TOML = """\
default_chat_model = "unused"
default_collection = "{name}"
query_expander = "identity"
reranker = "identity"

# never called by retrieval evaluation (no query expansion, no answers)
[chat_models.unused]
provider = "openai"
model = "gpt-4o-mini"

[embedding_models.local]
provider = "huggingface"
model = "{embedding}"
dimensions = {dims}
batch_size = 32
[embedding_models.local.options]
pooling = "{pooling}"
query_prefix = "{query_prefix}"

[reranker_models.ce]
provider = "cross-encoder"
model = "cross-encoder/ms-marco-MiniLM-L6-v2"

[reranker_models.bge]
provider = "cross-encoder"
model = "BAAI/bge-reranker-base"

[collections.{name}]
embedding_model = "local"
index_prefix = "beir-{name}"
kinds = ["text"]
parsers = ["text"]
"""


def download(name: str, work: Path) -> Path:
    target = work / name
    if (target / "corpus.jsonl").exists():
        return target
    work.mkdir(parents=True, exist_ok=True)
    archive = work / f"{name}.zip"
    print(f"downloading {BASE_URL.format(name=name)} ...", flush=True)
    urllib.request.urlretrieve(BASE_URL.format(name=name), archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(work)
    return target


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_corpus(dataset: Path, docs_dir: Path) -> int:
    docs_dir.mkdir(parents=True, exist_ok=True)
    corpus = read_jsonl(dataset / "corpus.jsonl")
    for doc in corpus:
        title = (doc.get("title") or "").strip()
        body = (doc.get("text") or "").strip()
        (docs_dir / f"{doc['_id']}.txt").write_text(f"{title}\n\n{body}" if title else body, encoding="utf-8")
    return len(corpus)


def write_cases(dataset: Path, split: str, collection: str, out: Path) -> int:
    queries = {q["_id"]: q["text"] for q in read_jsonl(dataset / "queries.jsonl")}
    relevant: dict[str, list[tuple[str, int]]] = {}
    with (dataset / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["score"]) > 0:
                relevant.setdefault(row["query-id"], []).append((row["corpus-id"], int(row["score"])))
    lines = []
    for query_id, labels in relevant.items():
        if query_id in queries:
            case = {
                "id": f"{collection}-{query_id}",
                "query": queries[query_id],
                "relevant": [{"source": f"{doc_id}.txt", "grade": grade} for doc_id, grade in labels],
                "collection": collection,
            }
            lines.append(json.dumps(case, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


async def ingest(files: list[Path], config_path: Path, collection: str, concurrency: int) -> None:
    from src.core.config import Settings, load_rag_config
    from src.core.container import Container
    from src.core.types import JobSpec

    settings = Settings(
        rag_config=config_path, ocr_enabled=False, data_dir=config_path.parent / "data", log_level="WARNING"
    )
    container = await Container.build(
        settings, role="cli", with_ingestion=True, config=load_rag_config(settings)
    )
    await container.start()
    try:
        gate = asyncio.Semaphore(concurrency)
        done = 0
        started = time.perf_counter()

        async def one(path: Path) -> None:
            nonlocal done
            async with gate:
                await container.ingestion.ingest(JobSpec(collection=collection, path=str(path)))  # type: ignore[union-attr]
            done += 1
            if done % 500 == 0 or done == len(files):
                print(f"  ingested {done}/{len(files)} ({time.perf_counter() - started:.0f}s)", flush=True)

        await asyncio.gather(*(one(p) for p in files))
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="scifact", help="BEIR dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--work", type=Path, default=Path("/tmp/beir"))
    parser.add_argument("--embedding", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dims", type=int, default=384)
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--query-prefix", default="Represent this sentence for searching relevant passages: ")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    dataset = download(args.name, args.work)
    docs_dir = dataset / "docs"
    count = write_corpus(dataset, docs_dir)
    cases = write_cases(dataset, args.split, args.name, dataset / "cases.jsonl")
    config_path = dataset / "rag.toml"
    config_path.write_text(
        TOML.format(
            name=args.name,
            embedding=args.embedding,
            dims=args.dims,
            pooling=args.pooling,
            query_prefix=args.query_prefix,
        ),
        encoding="utf-8",
    )
    print(f"{count} documents, {cases} labelled queries ({args.split}) in {dataset}")
    if not args.skip_ingest:
        asyncio.run(ingest(sorted(docs_dir.glob("*.txt")), config_path, args.name, args.concurrency))
    print(f"\nexport RAG_CONFIG={config_path} OPENAI_API_KEY=unused")
    print(f"rag eval run {dataset / 'cases.jsonl'} --granularity document -c {args.name}")


if __name__ == "__main__":
    main()
