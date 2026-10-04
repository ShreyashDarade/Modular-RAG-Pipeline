"""The architecture is a build artefact: these tests run the Import Linter contracts in ``pyproject.toml`` and
check that nothing escapes them (docs/framework.md, section 2)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from importlinter.cli import lint_imports

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["importlinter"]


def test_every_import_linter_contract_holds():
    assert lint_imports(config_filename=str(ROOT / "pyproject.toml"), no_cache=True) == 0, (
        "an architecture contract is broken - run `lint-imports` for the offending import chain"
    )


def _layer_units() -> set[str]:
    layers = next(c for c in CONFIG["contracts"] if c["type"] == "layers")["layers"]
    return {unit.strip() for layer in layers for unit in layer.split("|")}


def _thin_units() -> set[str]:
    contract = next(c for c in CONFIG["contracts"] if c["name"].startswith("The thin client"))
    return set(contract["source_modules"])


def _covered(module: str, units: set[str]) -> bool:
    return any(module == u or module.startswith(u + ".") for u in units)


def _engine_modules() -> list[str]:
    modules = []
    for entry in sorted((ROOT / "src").iterdir()):
        if entry.name.startswith(("_", ".")) or entry.name == "__pycache__":
            continue
        if entry.name == "core":  # core is split into layers of its own: classify each module
            modules += [f"src.core.{p.stem}" for p in sorted(entry.glob("*.py")) if p.stem != "__init__"]
        elif entry.is_dir() and (entry / "__init__.py").exists():
            modules.append(f"src.{entry.name}")
        elif entry.suffix == ".py":
            modules.append(f"src.{entry.stem}")
    return modules


@pytest.mark.parametrize("module", _engine_modules())
def test_every_engine_package_is_assigned_a_layer(module: str):
    """A new top-level package must be placed in the layers contract: Import Linter's exhaustive mode only
    checks direct children of a container, so this closes the gap."""
    assert _covered(module, _layer_units()), (
        f"{module} is not in the layers contract of pyproject.toml - decide which layer it belongs to"
    )


@pytest.mark.parametrize(
    "module",
    [f"ai_rag_info.{p.stem}" for p in sorted((ROOT / "ai_rag_info").glob("*.py")) if p.stem != "__init__"],
)
def test_every_sdk_module_is_classified_as_thin_or_engine_side(module: str):
    assert module in _thin_units() or _covered(module, _layer_units()), (
        f"{module} is neither in the thin-client contract nor in the layers contract: classify it"
    )


def test_the_thin_client_set_is_what_the_client_actually_imports():
    """The modules `ai_rag_info.client` pulls in must all be listed as thin - otherwise the purity contract
    would not be checking them."""
    import grimp

    graph = grimp.build_graph("ai_rag_info", "src")
    imported = set()
    frontier = ["ai_rag_info.client"]
    while frontier:
        current = frontier.pop()
        for dep in graph.find_modules_directly_imported_by(current):
            if dep.startswith("ai_rag_info") and dep not in imported:
                imported.add(dep)
                frontier.append(dep)
    assert imported - {"ai_rag_info"} <= _thin_units(), (
        f"unlisted thin-client modules: {imported - _thin_units()}"
    )


# --- exhaustive checks: they look at EVERY module, so a package added later cannot slip past the static lists ----
def _graph():
    import grimp

    return grimp.build_graph("src", "ai_rag_info", include_external_packages=True)


def _in(module: str, units: list[str]) -> bool:
    return any(module == u or module.startswith(u + ".") for u in units)


def test_every_heavy_third_party_library_is_imported_only_where_it_is_confined():
    """The Import Linter contracts name the modules they police; this iterates all of them."""
    confinement = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["architecture"]["confinement"]
    graph = _graph()
    leaks = []
    for module in sorted(graph.modules):
        if not module.startswith(("src.", "ai_rag_info.")) and module not in ("src", "ai_rag_info"):
            continue
        for imported in graph.find_modules_directly_imported_by(module):
            top = imported.split(".")[0]
            if top in confinement and not _in(module, confinement[top]):
                leaks.append(f"{module} imports {top} (allowed only in {', '.join(confinement[top])})")
    assert not leaks, "third-party confinement broken:\n  " + "\n  ".join(leaks)


def test_the_confinement_table_and_the_import_linter_contracts_name_the_same_libraries():
    confinement = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["architecture"]["confinement"]
    in_contracts = {
        c["forbidden_modules"][0]
        for c in CONFIG["contracts"]
        if c["type"] == "forbidden"
        and len(c["forbidden_modules"]) == 1
        and c["name"].endswith(tuple(", ".join(v) for v in confinement.values()))
    }
    assert set(confinement) == in_contracts, "pyproject's confinement table and its contracts drifted apart"


def test_everything_the_thin_client_can_reach_imports_only_the_standard_library_httpx_and_pydantic():
    """Closes the hole in a forbidden-list contract: a library nobody thought to list."""
    import sys

    graph = _graph()
    seen: set[str] = set()
    frontier = ["ai_rag_info.client", "ai_rag_info.errors", "ai_rag_info.models"]
    external: dict[str, str] = {}
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        for imported in graph.find_modules_directly_imported_by(module):
            top = imported.split(".")[0]
            if top in ("src", "ai_rag_info"):
                frontier.append(imported)
            elif top not in sys.stdlib_module_names and top != "__future__":
                external.setdefault(top, module)
    unexpected = {lib: where for lib, where in external.items() if lib not in {"httpx", "pydantic"}}
    assert not unexpected, (
        f"the thin client reaches third-party libraries beyond httpx and pydantic: {unexpected}"
    )
    assert not any(m.startswith(("src.application", "src.core.container", "src.indexing")) for m in seen)


def test_the_package_root_does_not_import_the_engine_at_import_time():
    """`ai_rag_info/__init__.py` is in no Import Linter contract (the package root contains the in-process engine)."""
    import ast

    tree = ast.parse((ROOT / "ai_rag_info" / "__init__.py").read_text())
    eager = []
    for node in tree.body:  # module level only: TYPE_CHECKING blocks and function bodies are not eager
        if isinstance(node, ast.ImportFrom) and node.module:
            eager.append(node.module)
        elif isinstance(node, ast.Import):
            eager += [alias.name for alias in node.names]
    assert not [m for m in eager if m.startswith(("src", "ai_rag_info.embedded", "ai_rag_info.testing"))], (
        eager
    )
