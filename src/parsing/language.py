from __future__ import annotations

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42  # deterministic results

SUPPORTED_LANGS = {"en": "english", "mr": "marathi", "hi": "hindi"}


def detect_language(text: str) -> str:
    """ISO code for English/Marathi/Hindi, or ``"unknown"`` (also for text with no letters)."""
    cleaned = text.strip()
    if not cleaned:
        return "unknown"
    try:
        language = detect(cleaned)
    except LangDetectException:
        return "unknown"
    return language if language in SUPPORTED_LANGS else "unknown"
