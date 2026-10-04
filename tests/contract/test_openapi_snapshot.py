"""The wire contract is checked in; a change to it must be a deliberate, reviewable diff."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _exporter():
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_the_app_still_produces_the_checked_in_openapi_document():
    exporter = _exporter()
    expected = (ROOT / "docs" / "openapi.json").read_text(encoding="utf-8")
    assert exporter.render(exporter.current_document()) == expected, (
        "the REST API's OpenAPI document changed. If that is intended (additive changes only within /api/v1), "
        "run `python scripts/export_openapi.py` and commit docs/openapi.json"
    )


def test_every_sdk_operation_has_a_route_in_the_contract():
    """The SDK calls these paths; if one disappears from the contract the SDK is broken."""
    document = json.loads((ROOT / "docs" / "openapi.json").read_text(encoding="utf-8"))
    paths = document["paths"]
    for method, path in [
        ("post", "/api/v1/retrieve"),
        ("post", "/api/v1/ask"),
        ("post", "/api/v1/chat"),
        ("post", "/api/v1/chat/stream"),
        ("get", "/api/v1/chat/{conversation_id}"),
        ("delete", "/api/v1/chat/{conversation_id}"),
        ("post", "/api/v1/ingest"),
        ("get", "/api/v1/jobs/{job_id}"),
        ("get", "/api/v1/documents"),
        ("delete", "/api/v1/documents"),
        ("get", "/api/v1/collections"),
        ("get", "/api/v1/models"),
    ]:
        assert method in paths.get(path, {}), f"{method.upper()} {path} is missing from docs/openapi.json"
