"""Prometheus metrics. Module-level collectors on the default registry; one process = one scrape
target, so run one server process per container and scale with replicas."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "rag_http_request_seconds", "HTTP request latency", ["method", "route"], buckets=_LATENCY_BUCKETS
)
HTTP_IN_FLIGHT = Gauge("rag_http_in_flight", "Requests currently being served")
HTTP_SHED = Counter("rag_http_shed_total", "Requests rejected by admission control")
RATE_LIMITED = Counter("rag_rate_limited_total", "Requests rejected by the rate limiter", ["route"])

CACHE_REQUESTS = Counter("rag_cache_requests_total", "Cache lookups", ["cache", "result"])

UPSTREAM_LATENCY = Histogram(
    "rag_upstream_seconds",
    "Latency of calls to dependencies",
    ["service", "operation"],
    buckets=_LATENCY_BUCKETS,
)
UPSTREAM_ERRORS = Counter(
    "rag_upstream_errors_total", "Failed calls to dependencies", ["service", "operation"]
)

INGEST_JOBS = Counter("rag_ingest_jobs_total", "Ingestion jobs by outcome", ["status"])
INGEST_SECONDS = Histogram(
    "rag_ingest_seconds", "Ingestion job duration", buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600)
)
INGEST_QUEUE_DEPTH = Gauge("rag_ingest_queue_depth", "Ingestion jobs waiting or running")
