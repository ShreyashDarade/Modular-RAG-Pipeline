"""Aggregate case results into a report, save/load it, compare reports, and gate on it."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.errors import ConfigError
from src.evaluation.dataset import Dataset
from src.evaluation.metrics import Interval, PairedDelta, mean_ci, paired_delta, percentile
from src.evaluation.runner import CaseResult

#: Shown first, in this order, when present.
HEADLINE = ("ndcg@10", "recall@10", "mrr@10", "hit@5", "recall@5", "ndcg@5")
SCHEMA = 1


@dataclass(slots=True)
class EvalReport:
    name: str
    meta: dict[str, Any]
    #: Retrieval and answer metrics: mean with a bootstrap confidence interval and the number of queries.
    metrics: dict[str, Interval]
    #: tag -> metric -> (mean, n)
    by_tag: dict[str, dict[str, tuple[float, int]]]
    latency_ms: dict[str, float]
    errors: int
    cases: list[dict[str, Any]]

    # --- per-case values, for paired comparison ----------------------------------------------
    def values(self, metric: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for case in self.cases:
            value = case["metrics"].get(metric)
            if value is None:
                value = ((case.get("answer") or {}).get("metrics") or {}).get(metric)
            if value is not None:
                out[case["id"]] = value
        return out

    # --- serialisation ---------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "meta": self.meta,
            "metrics": {
                k: {"mean": v.mean, "low": v.low, "high": v.high, "n": v.n} for k, v in self.metrics.items()
            },
            "by_tag": {
                t: {m: {"mean": v, "n": n} for m, (v, n) in ms.items()} for t, ms in self.by_tag.items()
            },
            "latency_ms": self.latency_ms,
            "errors": self.errors,
            "cases": self.cases,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalReport:
        if raw.get("schema") != SCHEMA:
            raise ConfigError(f"unsupported report schema {raw.get('schema')!r}; expected {SCHEMA}")
        return cls(
            name=raw["name"],
            meta=dict(raw["meta"]),
            metrics={k: Interval(v["mean"], v["low"], v["high"], v["n"]) for k, v in raw["metrics"].items()},
            by_tag={t: {m: (v["mean"], v["n"]) for m, v in ms.items()} for t, ms in raw["by_tag"].items()},
            latency_ms=dict(raw["latency_ms"]),
            errors=int(raw["errors"]),
            cases=list(raw["cases"]),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EvalReport:
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ConfigError(f"cannot read report {path}: {exc}") from exc


def build_report(
    name: str, dataset: Dataset, results: Sequence[CaseResult], meta: Mapping[str, Any]
) -> EvalReport:
    per_metric: dict[str, list[float]] = defaultdict(list)
    per_tag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        values = dict(result.metrics)
        if result.answer:
            values.update(result.answer.get("metrics") or {})
        for metric, value in values.items():
            per_metric[metric].append(value)
            for tag in result.tags:
                per_tag[tag][metric].append(value)
    latencies = [r.latency_ms for r in results if r.error is None]
    failed_answers = sum(1 for r in results if r.answer and (r.answer.get("error") or r.answer.get("failed")))
    return EvalReport(
        name=name,
        meta={
            **meta,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset_fingerprint": dataset.fingerprint,
            "cases": len(dataset),
            "labelled_cases": sum(1 for c in dataset.cases if c.labels),
        },
        metrics={m: mean_ci(v) for m, v in per_metric.items()},
        by_tag={t: {m: (sum(v) / len(v), len(v)) for m, v in ms.items()} for t, ms in per_tag.items()},
        latency_ms=(
            {
                "mean": sum(latencies) / len(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies),
            }
            if latencies
            else {}
        ),
        errors=sum(1 for r in results if r.error is not None) + failed_answers,
        cases=[r.to_dict() for r in results],
    )


def ordered_metrics(names: Sequence[str]) -> list[str]:
    head = [m for m in HEADLINE if m in names]
    rest = sorted(m for m in names if m not in head)
    return [*head, *rest]


def render_report(report: EvalReport) -> str:
    lines = [
        f"{report.name}  -  {report.meta.get('labelled_cases', '?')} labelled / {report.meta['cases']} queries"
    ]
    config = report.meta.get("config")
    if config:
        lines.append("  " + ", ".join(f"{k}={v}" for k, v in config.items()))
    width = max((len(m) for m in report.metrics), default=8)
    lines.append(f"  {'metric':<{width}}   mean    95% CI           n")
    for metric in ordered_metrics(list(report.metrics)):
        v = report.metrics[metric]
        lines.append(f"  {metric:<{width}}  {v.mean:6.3f}  [{v.low:5.3f}, {v.high:5.3f}]  {v.n:4d}")
    if report.latency_ms:
        lat = report.latency_ms
        note = ""
        if report.meta.get("concurrency", 1) > 1:
            note = f"  (concurrency {report.meta['concurrency']}: includes queueing; use --concurrency 1 for clean latency)"
        lines.append(f"  latency ms: p50 {lat['p50']:.0f}  p95 {lat['p95']:.0f}  max {lat['max']:.0f}{note}")
    short = report.meta.get("short_rankings", 0)
    if short:
        lines.append(
            f"  note: {short} quer{'y' if short == 1 else 'ies'} returned fewer than {max(report.meta.get('ks', [0]))} "
            "results, so the largest cutoff is not fully populated (raise retriever_top_k)"
        )
    if report.errors:
        lines.append(
            f"  !! {report.errors} quer{'y' if report.errors == 1 else 'ies'} FAILED "
            "(a failed retrieval scores 0; a failed answer or judgement is left out of the answer metrics)"
        )
    return "\n".join(lines)


# --- comparison -----------------------------------------------------------------------------------
@dataclass(slots=True)
class MetricComparison:
    metric: str
    baseline: float
    values: list[float]  # one per compared report
    deltas: list[PairedDelta]


def compare_reports(
    reports: Sequence[EvalReport], metrics: Sequence[str] | None = None
) -> list[MetricComparison]:
    """The first report is the baseline; every other is compared to it on the same queries."""
    if len(reports) < 2:
        raise ConfigError("compare needs at least two reports")
    fingerprints = {r.meta.get("dataset_fingerprint") for r in reports}
    if len(fingerprints) != 1:
        raise ConfigError(
            "these reports were produced on different datasets and cannot be compared: "
            + ", ".join(f"{r.name}={r.meta.get('dataset_fingerprint')}" for r in reports)
        )
    base, others = reports[0], reports[1:]
    shared = set(base.metrics).intersection(*(r.metrics for r in others))
    chosen = ordered_metrics([m for m in (metrics or shared) if m in shared])
    return [
        MetricComparison(
            metric,
            base.metrics[metric].mean,
            [r.metrics[metric].mean for r in others],
            [paired_delta(base.values(metric), r.values(metric)) for r in others],
        )
        for metric in chosen
    ]


def render_comparison(reports: Sequence[EvalReport], comparisons: Sequence[MetricComparison]) -> str:
    base, others = reports[0], reports[1:]
    lines = [
        f"baseline: {base.name}   ({base.meta['cases']} queries; 95% bootstrap CI of the paired difference)"
    ]
    width = max((len(c.metric) for c in comparisons), default=8)
    header = f"  {'metric':<{width}}  {base.name[:14]:>14}"
    for other in others:
        header += f"  {other.name[:14]:>14}  {'delta [95% CI]':>24}  {'p':>6}"
    lines.append(header)
    for c in comparisons:
        row = f"  {c.metric:<{width}}  {c.baseline:14.3f}"
        for value, d in zip(c.values, c.deltas, strict=True):
            mark = "*" if d.significant else " "
            row += f"  {value:14.3f}  {d.diff:+7.3f} [{d.low:+6.3f},{d.high:+6.3f}]{mark}  {d.p_value:6.3f}"
        lines.append(row)
    lines.append("  * the interval excludes 0. With many metrics and variants some will, by chance: read")
    lines.append("    them as a lead to check, not a verdict. A confident claim needs a larger query set.")
    for other in others:
        if other.latency_ms and base.latency_ms:
            lines.append(
                f"  latency p50/p95 ms: {base.name} {base.latency_ms['p50']:.0f}/{base.latency_ms['p95']:.0f}"
                f"  ->  {other.name} {other.latency_ms['p50']:.0f}/{other.latency_ms['p95']:.0f}"
            )
    return "\n".join(lines)


# --- gating ------------------------------------------------------------------------------------------
def check_report(
    report: EvalReport,
    *,
    minimums: Mapping[str, float] | None = None,
    baseline: EvalReport | None = None,
    max_drop: float = 0.02,
    require_significance: bool = False,
) -> list[str]:
    """Reasons the report fails the gate (empty = pass)."""
    problems = []
    if report.errors:
        problems.append(f"{report.errors} queries failed to run")
    for metric, floor in (minimums or {}).items():
        have = report.metrics.get(metric)
        if have is None:
            problems.append(
                f"{metric}: not in the report (have: {', '.join(ordered_metrics(list(report.metrics)))})"
            )
        elif have.mean < floor:
            problems.append(f"{metric}: {have.mean:.3f} is below the minimum {floor:.3f}")
    if baseline is not None:
        for c in compare_reports([baseline, report]):
            drop = -c.deltas[0].diff
            if drop > max_drop and (not require_significance or c.deltas[0].high < 0):
                problems.append(f"{c.metric}: fell {drop:.3f} from {c.baseline:.3f} to {c.values[0]:.3f}")
    return problems
