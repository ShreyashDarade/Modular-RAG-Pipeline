"""`rag eval ...` end to end against live Elasticsearch: ingest, score, sweep, compare, gate, generate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.cli.app import app
from src.core.config import get_settings
from typer.testing import CliRunner

from tests.conftest import ES_URL

pytestmark = pytest.mark.integration

DOCS = {
    "finance.txt": "Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions. "
    "Operating margin improved to eighteen percent after cost reductions in the second half of the year. "
    "Management expects subscription revenue to keep growing as enterprise customers renew their contracts.",
    "hr.txt": "Employee headcount increased modestly while hiring focused on engineering and support roles. "
    "Retention remained strong and voluntary attrition fell below two percent for the full fiscal year. "
    "The company introduced a new parental leave policy and expanded its internal training programme.",
    "ops.txt": "The data centre migration finished two weeks early and reduced infrastructure spending by a fifth. "
    "Latency across all regions improved after the network upgrade and the new load balancing configuration. "
    "Availability targets were met every month and no major incidents were recorded during the migration.",
}
CASES = [
    {
        "id": "fin",
        "query": "how did subscription revenue grow",
        "relevant": [{"source": "finance.txt"}],
        "tags": ["finance"],
    },
    {
        "id": "hr",
        "query": "employee attrition and retention",
        "relevant": [{"source": "hr.txt"}],
        "tags": ["people"],
    },
    {
        "id": "ops",
        "query": "data centre migration latency",
        "relevant": [{"source": "ops.txt"}],
        "tags": ["infra"],
    },
    {"id": "none", "query": "what is the airspeed of a swallow", "answerable": False},
]


@pytest.fixture
def cli(monkeypatch, make_settings, rag_toml: Path, run_id: str, tmp_path: Path):
    toml = tmp_path / "eval.toml"
    toml.write_text(
        rag_toml.read_text() + '\n[reranker_models.overlap]\nprovider = "overlap"\nmodel = "words"\n'
    )
    for key, value in {
        "ES_HOST": ES_URL,
        "ES_NUMBER_OF_REPLICAS": "0",
        "ES_REFRESH_INTERVAL": "1s",
        "ES_INDEX_REGISTRY": f"t{run_id}-registry",
        "RAG_CONFIG": str(toml),
        "PLUGINS": '["tests.fake_plugin"]',
        "DATA_DIR": str(tmp_path / "data"),
        "OCR_ENABLED": "false",
        "CACHE_BACKEND": "memory",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    runner = CliRunner()
    for name, text in DOCS.items():
        path = tmp_path / name
        path.write_text(text)
        result = runner.invoke(app, ["ingest", str(path)])
        assert result.exit_code == 0, result.output
    yield runner
    get_settings.cache_clear()
    import elasticsearch

    es = elasticsearch.Elasticsearch(make_settings().es_host)
    names = list(es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
    if names:
        es.indices.delete(index=names, ignore_unavailable=True)
    es.close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in CASES))
    return path


def test_run_scores_a_configuration_writes_a_report_and_gates_on_it(
    cli: CliRunner, dataset: Path, tmp_path: Path
):
    out = tmp_path / "base.json"
    result = cli.invoke(app, ["eval", "run", str(dataset), "--k", "1,3", "--name", "base", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "ndcg@3" in result.output and "recall@3" in result.output and "mrr@3" in result.output
    report = json.loads(out.read_text())
    assert report["meta"]["config"]["reranker"] == "identity" and report["meta"]["labelled_cases"] == 3
    assert report["errors"] == 0 and report["metrics"]["hit@3"]["n"] == 3
    assert report["metrics"]["hit@3"]["mean"] == 1.0, (
        "each question has one on-topic document in a three-document corpus"
    )
    assert {c["id"] for c in report["cases"]} == {"fin", "hr", "ops", "none"}

    assert cli.invoke(app, ["eval", "run", str(dataset), "--k", "3", "--min", "hit@3=1.0"]).exit_code == 0
    failing = cli.invoke(app, ["eval", "run", str(dataset), "--k", "3", "--min", "mrr@3=1.01"])
    assert failing.exit_code == 1 and "below the minimum" in failing.output
    assert cli.invoke(app, ["eval", "check", str(out), "--min", "hit@3=0.9"]).exit_code == 0
    assert cli.invoke(app, ["eval", "check", str(out), "--min", "hit@3=1.1"]).exit_code == 1
    assert cli.invoke(app, ["eval", "check", str(out), "--baseline", str(out)]).exit_code == 0


def test_sweep_runs_variants_on_the_same_queries_and_compares_them(
    cli: CliRunner, dataset: Path, tmp_path: Path
):
    out = tmp_path / "reports"
    result = cli.invoke(
        app,
        [
            "eval", "sweep", str(dataset), "--k", "1,3",
            "-v", "baseline:reranker=identity",
            "-v", "overlap:reranker=overlap",
            "-v", "noexp:query_expander=identity,hybrid_alpha=0.9",
            "--out-dir", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "baseline: baseline" in result.output and "overlap" in result.output and "noexp" in result.output
    assert "overlap:words" in result.output, "the report records which reranker model ran"
    assert sorted(p.name for p in out.glob("*.json")) == ["baseline.json", "noexp.json", "overlap.json"]
    compared = cli.invoke(app, ["eval", "compare", str(out / "baseline.json"), str(out / "overlap.json")])
    assert compared.exit_code == 0 and "ndcg@3" in compared.output


def test_a_variant_with_a_typo_or_bad_value_is_an_error_not_a_silent_no_op(cli: CliRunner, dataset: Path):
    typo = cli.invoke(app, ["eval", "run", str(dataset), "--set", "hybird_alpha=0.9"])
    assert typo.exit_code == 1 and "unknown override" in typo.output
    unknown = cli.invoke(app, ["eval", "run", str(dataset), "--set", "reranker=nonexistent"])
    assert unknown.exit_code == 1 and "Unknown reranker" in unknown.output
    assert cli.invoke(app, ["eval", "sweep", str(dataset), "-v", "only"]).exit_code == 1


def test_a_typo_in_the_last_variant_is_caught_before_any_variant_runs(
    cli: CliRunner, dataset: Path, tmp_path: Path
):
    out = tmp_path / "reports"
    result = cli.invoke(
        app,
        [
            "eval",
            "sweep",
            str(dataset),
            "-v",
            "base:reranker=identity",
            "-v",
            "bad:reranker=typo",
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 1 and "Unknown reranker 'typo'" in result.output
    assert "retrieval over" not in result.output, "nothing ran"
    assert not list(out.glob("*.json")) if out.exists() else True


def test_reports_are_saved_as_each_variant_finishes(
    cli: CliRunner, dataset: Path, tmp_path: Path, monkeypatch
):
    """A failure in a later variant must not lose the earlier variants' results."""
    import src.cli.evaluate as evaluate_module

    real = evaluate_module.evaluate_variant

    async def failing_second(variant, *args, **kwargs):
        if variant.name == "second":
            raise RuntimeError("simulated crash")
        return await real(variant, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "evaluate_variant", failing_second)
    out = tmp_path / "reports"
    result = cli.invoke(
        app, ["eval", "sweep", str(dataset), "-v", "first", "-v", "second", "--out-dir", str(out)]
    )
    assert result.exit_code != 0
    assert (out / "first.json").exists(), "the finished variant survived the crash"


def test_a_query_that_fails_is_reported_and_fails_the_run(cli: CliRunner, tmp_path: Path):
    path = tmp_path / "too-long.jsonl"
    path.write_text(
        json.dumps({"id": "long", "query": "word " * 1000, "relevant": [{"source": "finance.txt"}]}) + "\n"
    )
    result = cli.invoke(app, ["eval", "run", str(path), "--k", "3"])
    assert result.exit_code == 1 and "FAILED" in result.output and "failed to run" in result.output
    assert cli.invoke(app, ["eval", "run", str(path), "--k", "3", "--allow-errors"]).exit_code == 0


def test_answers_mode_generates_and_judges_with_the_configured_models(
    cli: CliRunner, dataset: Path, tmp_path: Path
):
    out = tmp_path / "answers.json"
    result = cli.invoke(
        app,
        ["eval", "run", str(dataset), "--k", "3", "--answers", "--judge-model", "smart", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "faithfulness" in result.output and "citation_valid" in result.output
    report = json.loads(out.read_text())
    assert report["meta"]["judge_model"] == "smart"
    by_id = {c["id"]: c["answer"] for c in report["cases"]}
    assert by_id["fin"]["metrics"]["citation_valid"] == 1.0, (
        "the scripted answer cites only retrieved sources"
    )
    assert "abstained" in by_id["none"]["metrics"]


def test_generate_drafts_a_dataset_that_run_can_then_score(cli: CliRunner, tmp_path: Path):
    out = tmp_path / "drafted.jsonl"
    result = cli.invoke(app, ["eval", "generate", "-n", "3", "--out", str(out), "--seed", "3"])
    assert result.exit_code == 0 and "wrote 3 synthetic questions" in result.output
    cases = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(cases) == 3 and all(c["tags"] == ["synthetic"] and c["relevant"] for c in cases)
    scored = cli.invoke(app, ["eval", "run", str(out), "--k", "3"])
    assert scored.exit_code == 0 and "recall@3" in scored.output
