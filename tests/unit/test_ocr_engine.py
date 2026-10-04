"""EasyOcrEngine decision logic with stub readers (no model weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("easyocr")

from src.core.config import Settings  # noqa: E402
from src.parsing.ocr.easyocr_engine import EasyOcrEngine  # noqa: E402
from src.ports.parsing import OcrResult  # noqa: E402


class StubReader:
    def __init__(self, confidences: list[float], text: str = "text") -> None:
        self.confidences, self.text, self.reads = list(confidences), text, 0

    def read(self, image: np.ndarray, language: str) -> OcrResult:
        self.reads += 1
        return OcrResult(self.text, language, self.confidences.pop(0))


def engine(**settings) -> EasyOcrEngine:
    return EasyOcrEngine(Settings(_env_file=None, ocr_gpu_enabled=False, **settings))


BASE = np.full((40, 40, 3), 255, np.uint8)


def test_english_second_pass_only_when_the_first_reads_poorly(monkeypatch):
    monkeypatch.setattr("src.parsing.ocr.easyocr_engine.preprocess_image", lambda image: image)
    e = engine(ocr_early_exit_confidence=0.85)
    confident = StubReader([0.95])
    e._readers["en"] = confident  # noqa: SLF001
    assert e._best("en", "en", BASE).confidence == 0.95 and confident.reads == 1  # noqa: SLF001
    poor = StubReader([0.5, 0.7])
    e._readers["en"] = poor  # noqa: SLF001
    assert e._best("en", "en", BASE).confidence == 0.7 and poor.reads == 2, (
        "the better of the two passes wins"
    )  # noqa: SLF001
    worse = StubReader([0.6, 0.4])
    e._readers["en"] = worse  # noqa: SLF001
    assert e._best("en", "en", BASE).confidence == 0.6  # noqa: SLF001


def test_devanagari_accepts_the_first_pass_by_default(monkeypatch):
    monkeypatch.setattr("src.parsing.ocr.easyocr_engine.preprocess_image", lambda image: image)
    e = engine()
    reader = StubReader([0.3])
    e._readers["dev"] = reader  # noqa: SLF001
    assert e._best("dev", "hi", BASE).confidence == 0.3 and reader.reads == 1, "no second pass for a 0.3 read"  # noqa: SLF001


def test_devanagari_threshold_is_configurable(monkeypatch):
    monkeypatch.setattr("src.parsing.ocr.easyocr_engine.preprocess_image", lambda image: image)
    e = engine(ocr_early_exit_confidence_devanagari=0.9)
    reader = StubReader([0.3, 0.5])
    e._readers["dev"] = reader  # noqa: SLF001
    assert e._best("dev", "hi", BASE).confidence == 0.5 and reader.reads == 2  # noqa: SLF001
