"""The Typer CLI against live services (fake model plug-in, TOML config)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.cli.app import app
from src.core.config import get_settings
from typer.testing import CliRunner

from tests.conftest import ES_URL
from tests.helpers import make_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
def cli(monkeypatch, make_settings, rag_toml: Path, run_id: str, tmp_path: Path):
    for key, value in {
        "ES_HOST": ES_URL,
        "ES_NUMBER_OF_REPLICAS": "0",
        "ES_REFRESH_INTERVAL": "1s",
        "ES_INDEX_REGISTRY": f"t{run_id}-registry",
        "RAG_CONFIG": str(rag_toml),
        "PLUGINS": '["tests.fake_plugin"]',
        "DATA_DIR": str(tmp_path / "data"),
        "OCR_ENABLED": "false",
        "CACHE_BACKEND": "memory",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield CliRunner()
    get_settings.cache_clear()
    import elasticsearch

    settings = make_settings()
    es = elasticsearch.Elasticsearch(settings.es_host)
    names = list(es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
    if names:
        es.indices.delete(index=names, ignore_unavailable=True)
    es.close()


def test_cli_end_to_end(cli: CliRunner, tmp_path: Path):
    pdf = make_pdf(
        tmp_path / "finance.pdf",
        ["Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions."],
    )

    result = cli.invoke(app, ["collections"])
    assert result.exit_code == 0 and "alpha" in result.output and "beta" in result.output
    result = cli.invoke(app, ["models"])
    assert (
        result.exit_code == 0
        and "fast" in result.output
        and "hash64" in result.output
        and "64 dims" in result.output
    )

    result = cli.invoke(app, ["ingest", str(pdf)])
    assert result.exit_code == 0 and "Indexed" in result.output and "into 'alpha'" in result.output
    assert "No changes detected" in cli.invoke(app, ["ingest", str(pdf)]).output

    result = cli.invoke(app, ["retrieve", "quarterly revenue growth", "--limit", "2"])
    assert result.exit_code == 0 and "finance.pdf" in result.output and "collection=alpha" in result.output
    result = cli.invoke(app, ["ask", "what happened to revenue", "--model", "smart"])
    assert result.exit_code == 0 and "Model: smart" in result.output and "finance.pdf" in result.output
    result = cli.invoke(app, ["documents"])
    assert result.exit_code == 0 and "1 document(s)" in result.output
    result = cli.invoke(app, ["delete", str(pdf)])
    assert result.exit_code == 0 and "Deleted" in result.output
    assert "0 document(s)" in cli.invoke(app, ["documents"]).output


def test_cli_reports_typed_errors_with_a_non_zero_exit(cli: CliRunner, tmp_path: Path):
    result = cli.invoke(app, ["retrieve", "x", "--collection", "nope"])
    assert result.exit_code == 1 and "not_found" in result.output and "unknown collection" in result.output
    bad = tmp_path / "program.exe"
    bad.write_text("MZ")
    result = cli.invoke(app, ["ingest", str(bad)])
    assert result.exit_code == 1 and "unsupported_type" in result.output
