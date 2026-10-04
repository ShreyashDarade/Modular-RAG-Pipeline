"""Build a container for one named configuration variant, e.g. ``ce:reranker=precise``.

A variant overrides RagConfig fields (``reranker``, ``query_expander``) and/or any Settings field
(``hybrid_alpha=0.7``, ``rerank_candidates=50``). Every override is validated by the same models
that validate real configuration, so a typo or a bad value is an error, not a silent no-op.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.core.bootstrap import build_registries
from src.core.config import Settings
from src.core.container import Container
from src.core.errors import ConfigError, UnknownComponentError
from src.core.specs import RagConfig

CONFIG_FIELDS = ("reranker", "query_expander")


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    overrides: Mapping[str, str]


def parse_variant(text: str) -> Variant:
    """``name`` or ``name:key=value,key=value``."""
    name, _, rest = text.partition(":")
    name = name.strip()
    if not name:
        raise ConfigError(f"variant '{text}' has no name")
    overrides: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"variant '{name}': '{part}' is not key=value")
        overrides[key.strip()] = value.strip()
    return Variant(name, overrides)


def apply_overrides(
    settings: Settings, config: RagConfig, overrides: Mapping[str, str]
) -> tuple[Settings, RagConfig]:
    config_updates = {k: v for k, v in overrides.items() if k in CONFIG_FIELDS}
    setting_updates: dict[str, Any] = {k: v for k, v in overrides.items() if k not in CONFIG_FIELDS}
    unknown = sorted(set(setting_updates) - set(Settings.model_fields))
    if unknown:
        raise ConfigError(
            f"unknown override(s) {unknown}: use {list(CONFIG_FIELDS)} or a Settings field (e.g. hybrid_alpha)"
        )
    try:
        new_settings = Settings.model_validate({**settings.model_dump(), **setting_updates})
        new_config = RagConfig.model_validate({**config.model_dump(), **config_updates})
    except ValidationError as exc:
        raise ConfigError(f"invalid override: {exc}") from exc
    return new_settings, new_config


#: With document-level scoring, chunks are collapsed to documents, so ask for more chunks than documents.
DOCUMENT_CHUNK_FACTOR = 3


def evaluation_settings(
    settings: Settings, ks: Sequence[int], overrides: Mapping[str, str], granularity: str = "chunk"
) -> Mapping[str, str]:
    """Overrides every evaluation run needs: a private in-memory cache (so repeated queries are
    never answered from a warm shared cache, which would flatter latency and hide reranker cost), and
    a result size and candidate pool wide enough for the largest cutoff - counted in documents when
    scoring documents."""
    forced: dict[str, str] = {"cache_backend": "memory"}
    try:
        wanted = max(ks) * (DOCUMENT_CHUNK_FACTOR if granularity == "document" else 1)
        top_k = (
            int(overrides["retriever_top_k"])
            if "retriever_top_k" in overrides
            else max(settings.retriever_top_k, wanted)
        )
        candidates = (
            int(overrides["rerank_candidates"])
            if "rerank_candidates" in overrides
            else settings.rerank_candidates
        )
    except ValueError as exc:
        raise ConfigError(f"invalid override: {exc}") from exc
    if "retriever_top_k" not in overrides and top_k != settings.retriever_top_k:
        forced["retriever_top_k"] = str(top_k)
    if "rerank_candidates" not in overrides and candidates < top_k:
        forced["rerank_candidates"] = str(
            top_k
        )  # an explicit smaller pool is left for the validator to refuse
    return {**forced, **overrides}


def validate_variant(
    settings: Settings, config: RagConfig, variant: Variant, ks: Sequence[int], granularity: str = "chunk"
) -> tuple[Settings, RagConfig]:
    """Everything that can be checked without connecting to anything: value validity and that the
    named reranker / query expander exist. Run before any variant starts, so a typo in the last
    variant cannot cost the first ones' results."""
    new_settings, new_config = apply_overrides(
        settings, config, evaluation_settings(settings, ks, variant.overrides, granularity)
    )
    registries = build_registries(new_settings)
    for registry, name in (
        (registries.rerankers, new_config.reranker_spec().provider),
        (registries.query_expanders, new_config.query_expander),
    ):
        if name not in registry:
            raise UnknownComponentError(
                f"Unknown {registry.kind} '{name}'. Registered: {', '.join(registry.names()) or '(none)'}"
            )
    return new_settings, new_config


async def build_variant(
    settings: Settings, config: RagConfig, variant: Variant, ks: Sequence[int], granularity: str = "chunk"
) -> Container:
    new_settings, new_config = validate_variant(settings, config, variant, ks, granularity)
    container = await Container.build(new_settings, role="cli", config=new_config)
    try:
        await container.start()
        await container.reranker.start()  # load the model now: a bad one fails the run, not the first query
    except BaseException:
        await container.close()
        raise
    return container
