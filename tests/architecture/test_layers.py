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
    [f"turinton_rag.{p.stem}" for p in sorted((ROOT / "turinton_rag").glob("*.py")) if p.stem != "__init__"],
)
def test_every_sdk_module_is_classified_as_thin_or_engine_side(module: str):
    assert module in _thin_units() or _covered(module, _layer_units()), (
        f"{module} is neither in the thin-client contract nor in the layers contract: classify it"
    )


def test_the_thin_client_set_is_what_the_client_actually_imports():
    """The modules `turinton_rag.client` pulls in must all be listed as thin - otherwise the purity contract
    would not be checking them."""
    import grimp

    graph = grimp.build_graph("turinton_rag", "src")
    imported = set()
    frontier = ["turinton_rag.client"]
    while frontier:
        current = frontier.pop()
        for dep in graph.find_modules_directly_imported_by(current):
            if dep.startswith("turinton_rag") and dep not in imported:
                imported.add(dep)
                frontier.append(dep)
    assert imported - {"turinton_rag"} <= _thin_units(), (
        f"unlisted thin-client modules: {imported - _thin_units()}"
    )
