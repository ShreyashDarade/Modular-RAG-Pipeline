from __future__ import annotations

import importlib

from src.chat.stores import register_builtin_stores
from src.chunking.recursive import register_builtin_chunkers
from src.core.config import Settings
from src.core.errors import ConfigError
from src.core.registry import Registries
from src.jobs.inprocess import register_inprocess
from src.jobs.redis_streams import register_redis
from src.models.providers import register_builtin_providers
from src.parsing.builtin import register_builtin_parsers
from src.retrieval.expansion import register_builtin_expanders
from src.retrieval.rerank import register_builtin_rerankers
from src.runtime.cache import register_builtin_caches
from src.runtime.ratelimit import register_builtin_limiters


def build_registries(settings: Settings) -> Registries:
    """Built-in components first, then every plug-in module named in ``PLUGINS``.

    A plug-in is any importable module exposing ``register(registries)``; it adds parsers,
    providers, chunkers, rerankers, ... by name. Nothing is discovered implicitly.
    """
    registries = Registries()
    for register in (
        register_builtin_providers,
        register_builtin_parsers,
        register_builtin_chunkers,
        register_builtin_expanders,
        register_builtin_rerankers,
        register_builtin_caches,
        register_builtin_limiters,
        register_builtin_stores,
        register_inprocess,
        register_redis,
    ):
        register(registries)
    for name in settings.plugins:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            raise ConfigError(f"plugin '{name}' could not be imported: {exc}") from exc
        hook = getattr(module, "register", None)
        if not callable(hook):
            raise ConfigError(f"plugin '{name}' has no register(registries) function")
        hook(registries)
    return registries
