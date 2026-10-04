#!/usr/bin/env python
"""Write the OpenAPI document of the REST API to ``docs/openapi.json``.

The file is checked in and ``tests/contract/test_openapi_snapshot.py`` fails when the app no longer produces
it, so every change to the wire contract shows up as a diff in review (docs/framework.md, section 4).
Run after an intentional change:  python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET = ROOT / "docs" / "openapi.json"


def current_document() -> dict:
    from src.api.server import create_app
    from src.core.config import Settings

    # defaults only: the document must not depend on the environment it is generated in
    app = create_app(Settings(_env_file=None, mcp_enabled=False, plugins=[]))
    return app.openapi()


def render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    TARGET.write_text(render(current_document()), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
