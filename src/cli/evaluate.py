"""``rag eval ...``: measure retrieval and answer quality, compare configurations, gate regressions."""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Annotated

import typer

from src.core.config import get_settings, load_rag_config
from src.core.container import Container
from src.core.errors import ConfigError, RagError
from src.core.logger import configure_logging
from src.evaluation.answers import AnswerEvaluator
from src.evaluation.dataset import load_dataset, write_dataset
from src.evaluation.generate import generate_cases
from src.evaluation.judge import LlmJudge
from src.evaluation.report import (
    EvalReport,
    build_report,
    check_report,
    compare_reports,
    render_comparison,
    render_report,
)
from src.evaluation.runner import Granularity, RetrievalEvaluator
from src.evaluation.variants import Variant, build_variant, parse_variant, validate_variant

eval_app = typer.Typer(
    help="Measure retrieval and answer quality; compare configurations.", no_args_is_help=True
)

Dataset = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="JSONL evaluation set")]
CollectionOpt = Annotated[str | None, typer.Option("--collection", "-c", help="Collection to search")]
KsOpt = Annotated[str, typer.Option("--k", help="Comma-separated cutoffs for hit/recall/precision/nDCG@k")]
GranularityOpt = Annotated[
    str,
    typer.Option(
        "--granularity", help="chunk: every chunk is a ranked item | document: best chunk per document"
    ),
]
AnswersOpt = Annotated[
    bool, typer.Option("--answers", help="Also generate answers and judge them (needs LLM calls)")
]
JudgeOpt = Annotated[
    str | None, typer.Option("--judge-model", help="Chat model used as judge (default: utility model)")
]
ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Chat model that writes the answers")]
OutOpt = Annotated[Path | None, typer.Option("--out", "-o", help="Write the JSON report here")]
MinOpt = Annotated[
    list[str] | None,
    typer.Option("--min", help="Fail unless metric >= value, e.g. ndcg@10=0.45 (repeatable)"),
]
ConcurrencyOpt = Annotated[
    int,
    typer.Option(
        "--concurrency",
        min=1,
        help="Queries in flight. Use 1 for clean latency numbers (more includes queueing)",
    ),
]
AllowErrors = Annotated[
    bool, typer.Option("--allow-errors", help="Exit 0 even if some queries failed (they still score 0)")
]


def _parse_ks(text: str) -> list[int]:
    try:
        ks = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError as exc:
        raise ConfigError(f"--k must be comma-separated integers, got '{text}'") from exc
    if not ks or ks[0] < 1:
        raise ConfigError("--k needs at least one positive integer")
    return ks


def _parse_minimums(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise ConfigError(f"--min expects metric=value, got '{item}'") from None
        if not sep:
            raise ConfigError(f"--min expects metric=value, got '{item}'")
    return out


def _granularity(value: str) -> Granularity:
    if value not in ("chunk", "document"):
        raise ConfigError("--granularity must be 'chunk' or 'document'")
    return value  # type: ignore[return-value]


def _progress(label: str):
    def show(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            typer.echo(f"  {label}: {done}/{total}", err=True)

    return show


def _guard(func):
    """Typed errors become a one-line message and exit code 1, as in the other commands."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RagError as exc:
            typer.secho(f"{exc.code}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc

    return functools.wraps(func)(wrapper)


async def evaluate_variant(
    variant: Variant,
    dataset_path: Path,
    *,
    collection: str | None,
    ks: list[int],
    granularity: Granularity,
    answers: bool,
    judge_model: str | None,
    answer_model: str | None,
    concurrency: int = 4,
) -> EvalReport:
    settings, config = get_settings(), load_rag_config(get_settings())
    dataset = load_dataset(dataset_path)
    container: Container = await build_variant(settings, config, variant, ks, granularity)
    try:
        s, c = container.settings, container.config
        evaluator = RetrievalEvaluator(
            container.retrieval,
            ks=ks,
            granularity=granularity,
            default_collection=collection,
            concurrency=concurrency,
        )
        typer.echo(f"[{variant.name}] retrieval over {len(dataset)} queries", err=True)
        results = await evaluator.run(dataset, _progress("retrieval"))
        if answers:
            judge = LlmJudge(container.models.chat(judge_model or c.utility_model))
            typer.echo(
                f"[{variant.name}] answers + judging (model {answer_model or c.default_chat_model}, judge {judge.model_id})",
                err=True,
            )
            await AnswerEvaluator(container.answers, judge, evaluator, answer_model=answer_model).run(
                dataset, results, _progress("answers")
            )
        spec = c.reranker_spec()
        meta = {
            "collection": collection or c.default_collection,
            "ks": ks,
            "granularity": granularity,
            "concurrency": concurrency,
            "short_rankings": sum(1 for r in results if r.error is None and len(r.ranked) < max(ks)),
            "config": {
                "reranker": f"{spec.provider}:{spec.model}" if spec.model else spec.provider,
                "query_expander": c.query_expander,
                "hybrid_alpha": s.hybrid_alpha,
                "candidates": s.rerank_candidates,
                "top_k": s.retriever_top_k,
                "fuzziness": s.es_bm25_fuzziness,
                **({"overrides": dict(variant.overrides)} if variant.overrides else {}),
            },
            **({"judge_model": judge.model_id} if answers else {}),
        }
        return build_report(variant.name, dataset, results, meta)
    finally:
        await container.close()


@eval_app.command("run")
@_guard
def run(
    dataset: Dataset,
    collection: CollectionOpt = None,
    k: KsOpt = "1,3,5,10",
    granularity: GranularityOpt = "chunk",
    name: Annotated[str, typer.Option("--name", help="Label for this run")] = "run",
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="Override, e.g. reranker=precise or hybrid_alpha=0.7")
    ] = None,
    answers: AnswersOpt = False,
    judge_model: JudgeOpt = None,
    model: ModelOpt = None,
    concurrency: ConcurrencyOpt = 4,
    out: OutOpt = None,
    minimum: MinOpt = None,
    baseline: Annotated[
        Path | None, typer.Option("--baseline", exists=True, help="Fail if a metric fell versus this report")
    ] = None,
    max_drop: Annotated[
        float, typer.Option("--max-drop", help="Allowed absolute drop versus --baseline")
    ] = 0.02,
    allow_errors: AllowErrors = False,
) -> None:
    """Score one configuration against a dataset; optionally gate on thresholds or a baseline report."""
    configure_logging("WARNING")
    variant = parse_variant(name + (":" + ",".join(set_) if set_ else ""))
    report = asyncio.run(
        evaluate_variant(
            variant,
            dataset,
            collection=collection,
            ks=_parse_ks(k),
            granularity=_granularity(granularity),
            answers=answers,
            judge_model=judge_model,
            answer_model=model,
            concurrency=concurrency,
        )
    )
    typer.echo(render_report(report))
    if out:
        report.save(out)
        typer.echo(f"report written to {out}", err=True)
    problems = check_report(
        report,
        minimums=_parse_minimums(minimum),
        baseline=EvalReport.load(baseline) if baseline else None,
        max_drop=max_drop,
    )
    if allow_errors:
        problems = [p for p in problems if "failed to run" not in p]
    if problems:
        for problem in problems:
            typer.secho(f"FAIL: {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@eval_app.command("sweep")
@_guard
def sweep(
    dataset: Dataset,
    variant: Annotated[
        list[str],
        typer.Option(
            "--variant", "-v", help="name:key=value,key=value (repeatable). The first is the baseline."
        ),
    ],
    collection: CollectionOpt = None,
    k: KsOpt = "1,3,5,10",
    granularity: GranularityOpt = "chunk",
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="key=value applied to every variant (a variant's own value wins)"),
    ] = None,
    answers: AnswersOpt = False,
    judge_model: JudgeOpt = None,
    model: ModelOpt = None,
    concurrency: ConcurrencyOpt = 4,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Write one JSON report per variant here, as each finishes"),
    ] = None,
) -> None:
    """Run several configurations on the same dataset and print them side by side with paired CIs."""
    configure_logging("WARNING")
    common = parse_variant("common:" + ",".join(set_ or [])).overrides
    variants = [Variant(v.name, {**common, **v.overrides}) for v in map(parse_variant, variant)]
    if len(variants) < 2:
        raise ConfigError("a sweep needs at least two --variant values (the first is the baseline)")
    if len({v.name for v in variants}) != len(variants):
        raise ConfigError("variant names must be unique")
    ks, mode = _parse_ks(k), _granularity(granularity)
    settings, config = get_settings(), load_rag_config(get_settings())
    for v in (
        variants
    ):  # every variant is checked before the first one starts: a typo in the last must not cost the rest
        validate_variant(settings, config, v, ks, mode)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    async def go() -> list[EvalReport]:
        reports = []
        for v in variants:  # sequential: each variant owns models, caches and (for a local reranker) the CPU
            report = await evaluate_variant(
                v,
                dataset,
                collection=collection,
                ks=ks,
                granularity=mode,
                answers=answers,
                judge_model=judge_model,
                answer_model=model,
                concurrency=concurrency,
            )
            reports.append(report)
            if out_dir:  # saved as soon as it exists, so a later failure cannot lose it
                report.save(out_dir / f"{report.name}.json")
        return reports

    reports = asyncio.run(go())
    for report in reports:
        typer.echo(render_report(report) + "\n")
    typer.echo(render_comparison(reports, compare_reports(reports)))
    if any(r.errors for r in reports):
        typer.secho("some queries failed - see the reports", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@eval_app.command("compare")
@_guard
def compare(
    reports: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Report files; the first is the baseline"),
    ],
) -> None:
    """Compare saved reports (same dataset) with paired bootstrap intervals."""
    loaded = [EvalReport.load(p) for p in reports]
    typer.echo(render_comparison(loaded, compare_reports(loaded)))


@eval_app.command("check")
@_guard
def check(
    report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    minimum: MinOpt = None,
    baseline: Annotated[Path | None, typer.Option("--baseline", exists=True)] = None,
    max_drop: Annotated[float, typer.Option("--max-drop")] = 0.02,
    significant: Annotated[
        bool, typer.Option("--significant", help="With --baseline: only fail on drops whose CI excludes 0")
    ] = False,
) -> None:
    """Exit 1 if a saved report is below the thresholds or regressed against a baseline (for CI)."""
    problems = check_report(
        EvalReport.load(report),
        minimums=_parse_minimums(minimum),
        baseline=EvalReport.load(baseline) if baseline else None,
        max_drop=max_drop,
        require_significance=significant,
    )
    for problem in problems:
        typer.secho(f"FAIL: {problem}", fg=typer.colors.RED, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo("ok")


@eval_app.command("generate")
@_guard
def generate(
    collection: CollectionOpt = None,
    count: Annotated[int, typer.Option("--count", "-n", help="Questions to draft")] = 50,
    model: ModelOpt = None,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    out: Annotated[Path, typer.Option("--out", "-o", help="JSONL file to write")] = Path("eval-cases.jsonl"),
) -> None:
    """Draft questions from the indexed corpus with a chat model (synthetic: review before trusting)."""
    configure_logging("WARNING")

    async def go():
        settings = get_settings()
        container = await Container.build(settings, role="cli")
        try:
            await container.start()
            name = collection or container.config.default_collection
            return await generate_cases(
                container.searcher,
                container.config,
                container.models.chat(model or container.config.utility_model),
                collection=name,
                count=count,
                seed=seed,
                progress=_progress("questions"),
            )
        finally:
            await container.close()

    generated = asyncio.run(go())
    if not generated.cases:
        raise ConfigError(
            f"no usable questions were drafted (skipped: {generated.skipped or 'nothing sampled'})"
        )
    write_dataset(generated.cases, out)
    typer.echo(f"wrote {len(generated.cases)} synthetic questions to {out}")
    if generated.skipped:
        typer.echo(f"skipped: {generated.skipped}", err=True)
    typer.echo(
        "These questions were written from the passages and reuse their wording; review a sample "
        "before relying on the scores.",
        err=True,
    )
