from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("turinton-rag")
except metadata.PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
