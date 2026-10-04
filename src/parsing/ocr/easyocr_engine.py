"""EasyOCR-backed ``OcrEngine`` for English, Hindi and Marathi."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.errors import ConfigError
from src.core.logger import logger
from src.parsing.language import detect_language
from src.parsing.ocr.cleanup import clean_ocr_text
from src.parsing.ocr.image_ops import decode_image, deskew_image, limit_side, preprocess_image
from src.ports.parsing import OcrResult

if TYPE_CHECKING:
    from src.core.config import Settings

_DEVANAGARI = ("mr", "hi")


def _devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if 0x0900 <= ord(ch) <= 0x097F) / len(letters)


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isascii()) / len(letters)


class _Reader:
    """One EasyOCR reader plus the lock that serialises inference on it (``readtext`` is not
    documented as thread-safe)."""

    def __init__(self, reader: Any, batch_size: int) -> None:
        self.reader = reader
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def read(self, image: np.ndarray, language: str) -> OcrResult:
        with self.lock:
            # paragraph=False: with paragraph=True EasyOCR drops the per-line confidence.
            lines = self.reader.readtext(image, detail=1, paragraph=False, batch_size=self.batch_size)
        texts = [str(line[1]) for line in lines]
        weights = [max(len(t), 1) for t in texts]
        confidence = (
            float(sum(float(line[2]) * w for line, w in zip(lines, weights, strict=True)) / sum(weights))
            if lines
            else 0.0
        )
        return OcrResult(
            text=clean_ocr_text("\n".join(texts), language=language), language=language, confidence=confidence
        )


class EasyOcrEngine:
    def __init__(self, settings: Settings) -> None:
        try:
            import easyocr
            import torch
        except ImportError as exc:
            raise ConfigError(
                "OCR needs EasyOCR and torch: pip install 'turinton-rag[worker]' (or set OCR_ENABLED=false)"
            ) from exc
        self._easyocr = easyocr
        self._gpu = settings.ocr_gpu_enabled and torch.cuda.is_available()
        self._model_dir = Path(settings.ocr_model_dir)
        self._download = settings.ocr_download_models
        self._max_side = settings.ocr_max_side
        self._early_exit = settings.ocr_early_exit_confidence
        self._batch_size = settings.ocr_batch_size
        self._supported = tuple(settings.supported_ocr_languages)
        self._readers: dict[str, _Reader] = {}
        self._build_lock = threading.Lock()
        logger.info("OCR engine ready (gpu=%s, models=%s)", self._gpu, self._model_dir)

    def _reader(self, group: str) -> _Reader:
        with self._build_lock:
            reader = self._readers.get(group)
            if reader is None:
                self._model_dir.mkdir(parents=True, exist_ok=True)
                languages = ["en"] if group == "en" else list(_DEVANAGARI)
                logger.info("loading EasyOCR models for %s", languages)
                reader = _Reader(
                    self._easyocr.Reader(
                        languages,
                        gpu=self._gpu,
                        verbose=False,
                        model_storage_directory=str(self._model_dir),
                        download_enabled=self._download,
                    ),
                    self._batch_size,
                )
                self._readers[group] = reader
            return reader

    def _best(self, group: str, language: str, base: np.ndarray) -> OcrResult:
        """Raw image first; only when it reads poorly pay for the (slow) pre-processed pass."""
        reader = self._reader(group)
        raw = reader.read(base, language)
        if raw.confidence >= self._early_exit:
            return raw
        processed = reader.read(preprocess_image(base), language)
        return processed if processed.confidence > raw.confidence else raw

    def read(self, image: bytes, language_hint: str | None) -> OcrResult:
        if language_hint is not None and language_hint not in self._supported:
            raise ConfigError(
                f"unsupported OCR language '{language_hint}' (supported: {', '.join(self._supported)})"
            )
        # Straighten first: confidence measures how well words were recognised, not whether they
        # came back in reading order, so a skewed page can score high yet be scrambled.
        base = deskew_image(limit_side(decode_image(image), self._max_side))

        if language_hint in _DEVANAGARI:
            return replace(self._best("dev", language_hint, base), language=language_hint)
        if language_hint == "en":
            return self._best("en", "en", base)

        devanagari = self._best("dev", "mr", base)
        english = self._best("en", "en", base)
        dev_ratio, eng_ratio = _devanagari_ratio(devanagari.text), _latin_ratio(english.text)

        def devanagari_language() -> str:
            guess = detect_language(devanagari.text)
            return guess if guess in _DEVANAGARI else "mr"

        if dev_ratio > 0.1 and dev_ratio > eng_ratio:
            return replace(devanagari, language=devanagari_language())
        if eng_ratio > dev_ratio:
            return english
        if devanagari.confidence > english.confidence:
            return replace(devanagari, language=devanagari_language())
        return english

    def close(self) -> None:
        self._readers.clear()
