#!/usr/bin/env python
"""Describe the public API surface of ``turinton_rag`` as text (docs/framework.md, section 6).

``tests/sdk/api_surface.txt`` holds the checked-in description and ``tests/sdk/test_api_surface.py`` fails when
the code no longer matches it, so every change to what is public - a new name, a removed one, a changed
signature, a new model field or error code - is a visible diff. Run after an intentional change:

    python scripts/api_surface.py --write

In CI, ``griffe check`` additionally compares against the latest release tag.
"""

from __future__ import annotations

import enum
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET = ROOT / "tests" / "sdk" / "api_surface.txt"
SUBMODULES = ("turinton_rag.models", "turinton_rag.errors", "turinton_rag.extend", "turinton_rag.testing")
RESOURCES = ("documents", "chat", "jobs", "collections")


def _signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "(...)"


def _methods(cls: type, indent: str) -> list[str]:
    lines = []
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, classmethod | staticmethod):
            lines.append(f"{indent}def {name}{_signature(member.__func__)}  [{type(member).__name__}]")
        elif inspect.isfunction(member):
            marks = " [experimental]" if getattr(member, "__rag_experimental__", False) else ""
            lines.append(
                f"{indent}{'async ' if inspect.iscoroutinefunction(member) else ''}def {name}{_signature(member)}{marks}"
            )
        elif isinstance(member, property):
            lines.append(f"{indent}property {name}")
    return lines


def _describe_class(label: str, cls: type) -> list[str]:
    bases = ", ".join(b.__name__ for b in cls.__bases__ if b is not object)
    lines = [f"class {label}({bases})"]
    model_fields = getattr(cls, "model_fields", None)
    if model_fields:
        for field_name, info in model_fields.items():
            default = "required" if info.is_required() else f"default={info.default!r}"
            lines.append(f"  field {field_name}: {info.annotation} [{default}]")
    for attr in ("code", "status_code"):
        if "code" in vars(cls) and attr in vars(cls):
            lines.append(f"  {attr} = {vars(cls)[attr]!r}")
    if "__init__" in vars(cls) and not model_fields and not getattr(cls, "__rag_internal_init__", False):
        lines.append(f"  def __init__{_signature(vars(cls)['__init__'])}")
    lines += _methods(cls, "  ")
    for res in RESOURCES:
        annotation = getattr(cls, "__annotations__", {}).get(res)
        if annotation is not None:
            lines.append(f"  resource {res}: {annotation}")
    if issubclass(cls, enum.Enum):
        lines += [f"  member {m.name}" for m in cls]
    return lines


def describe() -> str:
    out: list[str] = [
        "# Public API surface of turinton_rag. Regenerate: python scripts/api_surface.py --write",
        "",
    ]
    top = importlib.import_module("turinton_rag")
    out.append("## turinton_rag")
    for name in sorted(top.__all__):
        obj = getattr(top, name)
        if inspect.isclass(obj):
            out += _describe_class(f"turinton_rag.{name}", obj)
        elif callable(obj):
            out.append(f"def turinton_rag.{name}{_signature(obj)}")
        else:
            out.append(f"{name}: {type(obj).__name__}")
    # the resource objects are part of the interface: describe their classes too
    facade = importlib.import_module("turinton_rag._facade")
    sync = importlib.import_module("turinton_rag._sync")
    out += ["", "## resources (reached as rag.documents, rag.chat, rag.jobs, rag.collections)"]
    for res in RESOURCES:
        out += _describe_class(f"Async{res.capitalize()}", getattr(facade, f"Async{res.capitalize()}"))
        out += _describe_class(f"{res.capitalize()} [blocking]", getattr(sync, res.capitalize()))
    for module_name in SUBMODULES:
        module = importlib.import_module(module_name)
        out += ["", f"## {module_name}"]
        for name in sorted(module.__all__):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                out += _describe_class(f"{module_name}.{name}", obj)
            elif callable(obj):
                marks = " [experimental]" if getattr(obj, "__rag_experimental__", False) else ""
                out.append(
                    f"{'async ' if inspect.iscoroutinefunction(obj) else ''}def {module_name}.{name}{_signature(obj)}{marks}"
                )
            else:
                out.append(f"{module_name}.{name}: {type(obj).__name__}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if "--write" in sys.argv:
        TARGET.write_text(describe(), encoding="utf-8")
        print(f"wrote {TARGET.relative_to(ROOT)}")
    else:
        sys.stdout.write(describe())
