"""What is public is a checked-in decision (docs/framework.md, section 6)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

import ai_rag_info
import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBMODULES = ["models", "errors", "extend", "testing"]


def _surface():
    spec = importlib.util.spec_from_file_location("api_surface", ROOT / "scripts" / "api_surface.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_the_public_surface_matches_the_checked_in_description():
    expected = (ROOT / "tests" / "sdk" / "api_surface.txt").read_text(encoding="utf-8")
    assert _surface().describe() == expected, (
        "the public API of ai_rag_info changed. If that is intended, run `python scripts/api_surface.py --write` "
        "and commit tests/sdk/api_surface.txt - a removal or a changed signature also needs a deprecation (framework section 7)"
    )


@pytest.mark.parametrize("module_name", ["ai_rag_info", *[f"ai_rag_info.{m}" for m in SUBMODULES]])
def test_every_exported_name_resolves_is_sorted_and_documented(module_name):
    module = importlib.import_module(module_name)
    names = list(module.__all__)
    assert names == sorted(names, key=lambda n: (not n.isupper(), n)) or names == sorted(names), (
        f"{module_name}.__all__ is not sorted"
    )
    assert len(names) == len(set(names))
    for name in names:
        obj = getattr(module, name)  # lazy exports must resolve too
        if inspect.isclass(obj) or inspect.isfunction(obj):
            assert (inspect.getdoc(obj) or "").strip(), f"{module_name}.{name} has no docstring"


def test_the_package_is_typed_and_the_marker_ships():
    assert (Path(ai_rag_info.__file__).parent / "py.typed").exists()


def test_internal_modules_are_underscore_private_and_nothing_private_is_exported():
    package = Path(ai_rag_info.__file__).parent
    public_modules = {"client", "embedded", "errors", "extend", "models", "testing"}
    for file in package.glob("*.py"):
        stem = file.stem
        if stem != "__init__" and not stem.startswith("_"):
            assert stem in public_modules, (
                f"ai_rag_info/{file.name} looks public but is not a documented submodule"
            )
    for name in ai_rag_info.__all__:
        assert not name.startswith("_") or name == "__version__", f"{name} is private but exported"


def test_the_engine_is_not_imported_until_it_is_asked_for():
    """`import ai_rag_info` must stay thin: the in-process classes load on first use."""
    import subprocess
    import sys

    code = (
        "import sys, ai_rag_info; "
        "heavy = [m for m in ('src.core.container', 'src.application', 'elasticsearch', 'redis', 'fastapi', 'langchain_core') if m in sys.modules]; "
        "print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]", f"`import ai_rag_info` pulled in the engine: {out}"
