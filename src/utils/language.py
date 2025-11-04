from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from langdetect import DetectorFactory, detect, detect_langs

DetectorFactory.seed = 42

SUPPORTED_LANGS = {"en": "english", "mr": "marathi"}


@lru_cache(maxsize=1)
def get_default_langs() -> Iterable[str]:
    return SUPPORTED_LANGS.keys()


def detect_language(text: str) -> str:
    """Return ISO language code based on input text."""
    cleaned = text.strip()
    if not cleaned:
        return "unknown"
    try:
        lang = detect(cleaned)
        return lang if lang in SUPPORTED_LANGS else "unknown"
    except Exception:
        return "unknown"


def detect_language_confidence(text: str) -> tuple[str, float]:
    cleaned = text.strip()
    if not cleaned:
        return "unknown", 0.0
    try:
        result = detect_langs(cleaned)
        if not result:
            return "unknown", 0.0
        best = max(result, key=lambda r: r.prob)
        code = best.lang if best.lang in SUPPORTED_LANGS else "unknown"
        return code, best.prob
    except Exception:
        return "unknown", 0.0


__all__ = ["detect_language", "detect_language_confidence", "SUPPORTED_LANGS"]
