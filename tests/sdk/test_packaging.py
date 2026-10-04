"""The packaging promise (docs/adr/0005): `pip install turinton-rag` is a thin client.

Builds the wheel, installs it into a brand-new virtualenv *without extras*, and checks from the outside that
the client works, that nothing of the engine is installed or imported, and that asking for the in-process SDK
gives an install hint rather than a stack trace. Slow (builds and installs), so it is marked `packaging`;
CI runs it as its own job.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.packaging, pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")]

ROOT = Path(__file__).resolve().parents[2]
#: What the thin install may contain: the client's two dependencies and their own dependencies.
ALLOWED = {
    "turinton-rag", "pydantic", "pydantic-core", "annotated-types", "typing-extensions", "typing-inspection",
    "httpx", "httpcore", "h11", "anyio", "idna", "certifi", "sniffio",
}  # fmt: skip
ENGINE_LIBS = [
    "elasticsearch",
    "redis",
    "fastapi",
    "langchain_core",
    "pydantic_settings",
    "torch",
    "transformers",
]

PROGRAM = r"""
import importlib.util, json, sys
import httpx
from turinton_rag import RagClient, AsyncRagClient
from turinton_rag.errors import NotFoundError, RagError

def handler(request):
    if request.url.path.endswith("/retrieve"):
        return httpx.Response(200, json={"query": "q", "expanded_queries": ["q"], "documents": []})
    return httpx.Response(404, json={"detail": "nope", "code": "not_found"}, headers={"x-request-id": "r-1"})

import asyncio
async def main():
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with AsyncRagClient("http://x", http_client=http) as rag:
        ok = await rag.retrieve("q")
        try:
            await rag.chat.get("c")
        except NotFoundError as e:
            return ok.query, type(e).__name__, e.code, e.request_id
print(json.dumps({
    "result": asyncio.run(main()),
    "engine_imported": [m for m in sys.modules if m.split(".")[0] in ("elasticsearch", "redis", "fastapi", "langchain_core", "pydantic_settings", "torch", "transformers") or m in ("src.core.container", "src.application")],
    "engine_installed": [m for m in __LIBS__ if importlib.util.find_spec(m) is not None],
    "rag_hint": None,
}))
try:
    import turinton_rag; turinton_rag.Rag
except ImportError as exc:
    print("HINT=" + str(exc))
""".replace("__LIBS__", repr(ENGINE_LIBS))


def test_a_bare_install_is_a_working_thin_client(tmp_path: Path):
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), str(ROOT)], check=True, capture_output=True
    )
    wheel = next(dist.glob("turinton_rag-*.whl"))
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True, capture_output=True)
    python = str(venv / "bin" / "python")
    subprocess.run(["uv", "pip", "install", "--python", python, str(wheel)], check=True, capture_output=True)

    installed = {
        line.split()[0].lower()
        for line in subprocess.run(
            ["uv", "pip", "list", "--python", python], check=True, capture_output=True, text=True
        ).stdout.splitlines()[2:]
    }
    assert installed <= ALLOWED, (
        f"a bare install pulled in more than the thin client needs: {sorted(installed - ALLOWED)}"
    )

    out = subprocess.run(
        [python, "-c", PROGRAM], check=True, capture_output=True, text=True, cwd=tmp_path
    ).stdout.splitlines()
    report = json.loads(out[0])
    assert report["result"] == ["q", "NotFoundError", "not_found", "r-1"]
    assert report["engine_imported"] == [] and report["engine_installed"] == []
    hint = next(line for line in out if line.startswith("HINT="))
    assert "pip install 'turinton-rag[engine]'" in hint

    names = subprocess.run(
        [python, "-c", "import importlib.metadata as m; print(m.requires('turinton-rag'))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "extra ==" in names, "the engine is declared as extras, not as base dependencies"


def test_the_wheel_ships_py_typed_and_both_packages(tmp_path: Path):
    import zipfile

    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), str(ROOT)], check=True, capture_output=True
    )
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as z:
        files = set(z.namelist())
    assert (
        "turinton_rag/py.typed" in files
        and "turinton_rag/client.py" in files
        and "src/application/service.py" in files
    )
    assert not any(f.startswith(("tests/", "docs/", "scripts/")) for f in files), (
        "tests and docs do not belong in the wheel"
    )
