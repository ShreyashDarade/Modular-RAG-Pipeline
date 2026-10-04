#!/usr/bin/env python
"""Benchmark the pipeline against a local Elasticsearch.

Uses the deterministic fake models from tests/fake_plugin.py, so the numbers measure *this system*
(HTTP stack, fusion, caching, Elasticsearch round trips, ingestion) and not an LLM / embedding
provider's latency. A real deployment adds those, but they scale independently of this code.

The server runs as its own OS process (as in production) so the load generator does not share its
GIL; CPU utilisation of the server and of Elasticsearch is sampled during each run, which shows what
is saturated. Needs: pip install psutil.

    python scripts/benchmark.py --es http://localhost:9200 --docs 150 --duration 8
    python scripts/benchmark.py --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import psutil  # noqa: E402
from src.core.config import Settings  # noqa: E402

WORDS = [
    "revenue",
    "margin",
    "growth",
    "cloud",
    "subscription",
    "retention",
    "churn",
    "headcount",
    "infrastructure",
    "forecast",
    "quarter",
    "region",
    "enterprise",
    "customer",
    "pricing",
    "contract",
    "renewal",
    "support",
    "latency",
    "throughput",
    "deployment",
    "security",
    "compliance",
    "audit",
    "inventory",
    "logistics",
    "supplier",
    "warehouse",
    "shipment",
    "demand",
    "supply",
    "procurement",
    "budget",
    "expense",
    "profit",
    "loss",
    "asset",
    "liability",
    "equity",
    "dividend",
    "capital",
    "investment",
    "portfolio",
    "risk",
    "hedge",
    "currency",
    "inflation",
    "interest",
    "tax",
    "policy",
]


def make_docs(directory: Path, count: int, paragraphs: int) -> list[Path]:
    rng = random.Random(7)
    paths = []
    for i in range(count):
        body = "\n\n".join(
            " ".join(rng.choice(WORDS) for _ in range(120)).capitalize() + "." for _ in range(paragraphs)
        )
        path = directory / f"doc{i:04d}.txt"
        path.write_text(body)
        paths.append(path)
    return paths


class CpuMeter:
    """Mean CPU use (in cores) of some processes over a window."""

    def __init__(self, **procs: psutil.Process) -> None:
        self.procs = procs
        self.start: dict[str, float] = {}
        self.t0 = 0.0

    @staticmethod
    def _cpu(proc: psutil.Process) -> float:
        total = proc.cpu_times()
        reaped = getattr(total, "children_user", 0.0) + getattr(
            total, "children_system", 0.0
        )  # e.g. a finished pool
        return (
            total.user
            + total.system
            + reaped
            + sum(c.user + c.system for c in (x.cpu_times() for x in proc.children(recursive=True)))
        )

    def __enter__(self) -> CpuMeter:
        self.start = {name: self._cpu(p) for name, p in self.procs.items()}
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = time.perf_counter() - self.t0
        self.cores = {
            name: round((self._cpu(p) - self.start[name]) / elapsed, 2) for name, p in self.procs.items()
        }
        if "loadgen" in self.cores and "server" in self.cores:  # loadgen's children include the server
            self.cores["loadgen"] = round(max(0.0, self.cores["loadgen"] - self.cores["server"]), 2)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


async def ingest_bench(base: str, paths: list[Path], concurrency: int) -> dict:
    gate = asyncio.Semaphore(concurrency)
    chunks = 0
    failures = 0

    async def one(client: httpx.AsyncClient, path: Path) -> None:
        nonlocal chunks, failures
        async with gate:
            with path.open("rb") as handle:
                response = await client.post(
                    f"{base}/api/v1/ingest", files={"file": (path.name, handle)}, timeout=300
                )
        if response.status_code == 200:
            chunks += response.json()["text_chunks"]
        else:
            failures += 1

    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, p) for p in paths))
    elapsed = time.perf_counter() - started
    return {
        "docs": len(paths),
        "chunks": chunks,
        "seconds": round(elapsed, 2),
        "docs_per_s": round(len(paths) / elapsed, 1),
        "chunks_per_s": round(chunks / elapsed, 1),
        "failures": failures,
    }


async def _retrieve_samples(
    base: str, concurrency: int, duration: float, distinct_queries: int, seed: int, nonce: int
):
    rng = random.Random(11)
    # the nonce makes every run's queries new to the cache, so "cold" runs really are cold
    queries = [" ".join(rng.sample(WORDS, 3)) + f" {nonce}-{i}" for i in range(distinct_queries)]
    latencies: list[float] = []
    errors = 0
    stop_at = time.perf_counter() + duration

    async def worker(client: httpx.AsyncClient, local: random.Random) -> None:
        nonlocal errors
        while time.perf_counter() < stop_at:
            started = time.perf_counter()
            try:
                response = await client.post(f"{base}/api/v1/retrieve", json={"query": local.choice(queries)})
                if response.status_code != 200:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            latencies.append((time.perf_counter() - started) * 1000)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=60) as client:
        await asyncio.gather(*(worker(client, random.Random(seed * 1000 + i)) for i in range(concurrency)))
    return latencies, errors


def _load_process(args: tuple[str, int, float, int, int, int]) -> tuple[list[float], int]:
    return asyncio.run(_retrieve_samples(*args))


def retrieve_bench(
    base: str, concurrency: int, duration: float, distinct_queries: int, processes: int
) -> dict:
    """``concurrency`` clients spread over ``processes`` load-generator processes: one Python process
    saturates a core long before a healthy server does, which would make the *client* the limit."""
    processes = max(1, min(processes, concurrency))
    shares = [concurrency // processes + (1 if i < concurrency % processes else 0) for i in range(processes)]
    nonce = time.time_ns()
    started = time.perf_counter()
    if processes == 1:
        samples = [_load_process((base, concurrency, duration, distinct_queries, 1, nonce))]
    else:
        with ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("fork")) as pool:
            samples = list(
                pool.map(
                    _load_process,
                    [
                        (base, share, duration, distinct_queries, i + 1, nonce)
                        for i, share in enumerate(shares)
                    ],
                )
            )
    elapsed = time.perf_counter() - started
    latencies = [x for lat, _ in samples for x in lat]
    errors = sum(e for _, e in samples)
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "errors": errors,
        "qps": round(len(latencies) / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
    }


async def batching_bench(settings: Settings, repeat: int = 30) -> dict:
    """What one retrieval costs in Elasticsearch calls: the old design (serial: per query variant x index x
    lexical/vector) against the current one (a single _msearch). Latency on loopback understates the gap:
    over a real network the serial form pays one round trip per call."""
    from src.core.container import Container
    from src.core.types import SearchRequest

    container = await Container.build(settings, role="cli")
    await container.start()
    try:
        es = container.elastic.client
        collection = container.config.collection(container.config.default_collection)
        dims = container.models.embedder(collection.embedding_model).dimensions
        vector = tuple([0.1] * dims)
        requests = [
            SearchRequest(index, "revenue growth", vector, 30)
            for _ in range(4)
            for index in collection.index_names().values()
        ]  # 4 query variants x 3 content kinds
        searcher = container.retrieval._retriever._searcher

        async def serial() -> None:
            for r in requests:
                await es.search(index=r.index, query={"match": {"content": r.text}}, size=r.size)
                await es.search(
                    index=r.index,
                    size=r.size,
                    knn={
                        "field": "content_vector",
                        "query_vector": list(r.vector or ()),
                        "k": r.size,
                        "num_candidates": 60,
                    },
                )

        async def batched() -> None:
            await searcher.search(requests)

        result: dict = {}
        for name, fn in (("serial_calls", serial), ("one_msearch", batched)):
            await fn()  # warm up
            samples = []
            for _ in range(repeat):
                started = time.perf_counter()
                await fn()
                samples.append((time.perf_counter() - started) * 1000)
            result[name] = {
                "es_requests": len(requests) * 2 if name == "serial_calls" else 1,
                "median_ms": round(statistics.median(samples), 1),
            }
        result["speedup_on_loopback"] = round(
            result["serial_calls"]["median_ms"] / result["one_msearch"]["median_ms"], 1
        )
        return result
    finally:
        await container.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--es", default="http://localhost:9200")
    parser.add_argument("--docs", type=int, default=150)
    parser.add_argument("--paragraphs", type=int, default=10, help="paragraphs per document (~1 chunk each)")
    parser.add_argument("--ingest-concurrency", type=int, default=8)
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--duration", type=float, default=8.0, help="seconds per retrieval run")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--ingest-refresh", choices=["each", "interval"], default="each")
    parser.add_argument("--skip-retrieval", action="store_true", help="measure ingestion only")
    parser.add_argument(
        "--server-processes", type=int, default=1, help="uvicorn processes (>1 shares state through Redis)"
    )
    parser.add_argument(
        "--no-expansion", action="store_true", help="query_expander = identity (1 query variant instead of 4)"
    )
    parser.add_argument("--fuzziness", default="AUTO", help="ES_BM25_FUZZINESS (AUTO or 0)")
    parser.add_argument("--skip-warm", action="store_true", help="skip the cache-warm runs")
    parser.add_argument("--skip-cold", action="store_true", help="skip the cache-cold runs")
    parser.add_argument("--loadgen-processes", type=int, default=2, help="client processes sharing the load")
    parser.add_argument("--redis", default="redis://localhost:6379", help="used when --server-processes > 1")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="rag-bench-"))
    run_id = f"bench{int(time.time()) % 100000}"
    config_path = work / "rag.toml"
    config_path.write_text(
        f'''default_chat_model = "fast"
default_collection = "bench"
query_expander = "{"identity" if args.no_expansion else "llm"}"
[chat_models.fast]
provider = "fake"
model = "fast"
[embedding_models.h]
provider = "fake"
model = "hash"
dimensions = 64
[collections.bench]
embedding_model = "h"
index_prefix = "{run_id}"
[collections.bench.chunker]
chunk_size = 800
chunk_overlap = 100
min_chunk_size = 50
'''
    )
    settings = Settings(
        _env_file=None,
        es_host=args.es,
        es_number_of_replicas=0,
        es_refresh_interval="1s",
        es_index_registry=f"{run_id}-registry",
        rag_config=config_path,
        plugins=["tests.fake_plugin"],
        data_dir=work / "data",
        ocr_enabled=False,
        rate_limit_per_minute=0,
        max_concurrent_requests=0,
        ingest_concurrency=4,
        log_level="ERROR",
        cache_ttl_seconds=600,
    )
    port = free_port()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "ES_HOST": args.es,
        "ES_NUMBER_OF_REPLICAS": "0",
        "ES_REFRESH_INTERVAL": "1s",
        "ES_INDEX_REGISTRY": f"{run_id}-registry",
        "RAG_CONFIG": str(config_path),
        "PLUGINS": '["tests.fake_plugin"]',
        "DATA_DIR": str(work / "data"),
        "OCR_ENABLED": "false",
        "RATE_LIMIT_PER_MINUTE": "0",
        "MAX_CONCURRENT_REQUESTS": "0",
        "INGEST_CONCURRENCY": "4",
        "LOG_LEVEL": "ERROR",
        "CACHE_TTL_SECONDS": "600",
        "INGEST_REFRESH": args.ingest_refresh,
        "PORT": str(port),
        "HOST": "127.0.0.1",
        "WEB_CONCURRENCY": str(args.server_processes),
        "INGEST_BACKEND": "redis" if args.server_processes > 1 else "inprocess",
        "CACHE_BACKEND": "tiered" if args.server_processes > 1 else "memory",
        "RATE_LIMIT_BACKEND": "memory",
        "CHAT_STORE": "memory",
        "REDIS_NAMESPACE": run_id,
        "REDIS_URL": args.redis,
        "ES_BM25_FUZZINESS": args.fuzziness,
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "src.api.main"],
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    while True:
        try:
            if httpx.get(f"{base}/ready", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if server.poll() is not None or time.time() > deadline:
            raise SystemExit(
                "the server did not start:\n" + (server.stderr.read().decode() if server.stderr else "")
            )
        time.sleep(0.3)
    es_proc = next(
        (
            p
            for p in psutil.process_iter(["cmdline"])
            if p.info["cmdline"]
            and any(
                "org.elasticsearch.server" in a or "elasticsearch" in a.lower() for a in p.info["cmdline"][:3]
            )
            and p.name() == "java"
        ),
        None,
    )
    meters = {
        "server": psutil.Process(server.pid),
        **({"elasticsearch": es_proc} if es_proc else {}),
        "loadgen": psutil.Process(),  # if this is near 1.0 the client, not the server, is the limit
    }

    report: dict = {
        "environment": {
            "docs": args.docs,
            "machine": f"{os.cpu_count()} cores, {psutil.virtual_memory().total // 2**30} GiB",
            "loadgen_processes": args.loadgen_processes,
            "note": "fake models; Elasticsearch, the server and the load generator share this machine",
        }
    }
    try:
        docs = make_docs(work, args.docs, args.paragraphs)
        print(f"ingesting {len(docs)} documents ...", flush=True)
        with CpuMeter(**meters) as cpu:
            report["ingest"] = asyncio.run(ingest_bench(base, docs, args.ingest_concurrency))
        report["ingest"]["cpu_cores"] = cpu.cores
        print(" ", report["ingest"], flush=True)

        report["retrieve_cold"] = []
        report["retrieve_warm"] = []
        levels = [] if args.skip_retrieval else [int(x) for x in args.concurrency.split(",")]
        for c in [] if args.skip_cold else levels:
            print(f"retrieval, concurrency {c} (cache-cold: 5000 distinct queries) ...", flush=True)
            with CpuMeter(**meters) as cpu:
                result = retrieve_bench(base, c, args.duration, 5000, args.loadgen_processes)
            result["cpu_cores"] = cpu.cores
            report["retrieve_cold"].append(result)
            print(" ", report["retrieve_cold"][-1], flush=True)
        for c in [] if args.skip_warm else levels:
            print(f"retrieval, concurrency {c} (cache-warm: 20 distinct queries) ...", flush=True)
            with CpuMeter(**meters) as cpu:
                result = retrieve_bench(base, c, args.duration, 20, args.loadgen_processes)
            result["cpu_cores"] = cpu.cores
            report["retrieve_warm"].append(result)
            print(" ", report["retrieve_warm"][-1], flush=True)

        if not args.skip_retrieval:
            print("batched msearch vs serial calls ...", flush=True)
            report["batching"] = asyncio.run(batching_bench(settings))
            print(" ", report["batching"], flush=True)
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
        import elasticsearch

        es = elasticsearch.Elasticsearch(args.es)
        names = list(es.indices.get(index=f"{run_id}*", ignore_unavailable=True))
        if names:
            es.indices.delete(index=names, ignore_unavailable=True)
        es.close()
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
