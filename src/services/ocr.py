from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import easyocr
import numpy as np
from PIL import Image
import cv2

from src.core.logger import logger
from src.utils.image_ops import preprocess_image
from src.utils.language import detect_language

_MODEL_STORAGE = Path(__file__).resolve().parents[2] / "models"
_MODEL_STORAGE.mkdir(parents=True, exist_ok=True)


@dataclass
class OcrResult:
    text: str
    language: str
    confidence: float


class BaseOCREngine:
    def read(self, image: Image.Image | np.ndarray) -> OcrResult:
        raise NotImplementedError


class EasyEnglishOCREngine(BaseOCREngine):
    def __init__(self) -> None:
        self.reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False,
            model_storage_directory=str(_MODEL_STORAGE),
        )

    def read(self, image: Image.Image | np.ndarray) -> OcrResult:
        result = self.reader.readtext(image, detail=1, paragraph=True)
        text = "\n".join(chunk[1] for chunk in result if len(chunk) >= 2)
        confidence = _compute_confidence(result)
        return OcrResult(text=text, language="en", confidence=confidence)


class MarathiOCREngine(BaseOCREngine):
    def __init__(self) -> None:
        logger.info("Initializing Marathi OCR with EasyOCR pretrained weights")
        self.reader = easyocr.Reader(
            ["mr"],
            gpu=False,
            verbose=False,
            model_storage_directory=str(_MODEL_STORAGE),
        )

    def read(self, image: Image.Image | np.ndarray) -> OcrResult:
        result = self.reader.readtext(image, detail=1, paragraph=True)
        text = "\n".join(chunk[1] for chunk in result if len(chunk) >= 2)
        confidence = _compute_confidence(result)
        return OcrResult(text=text, language="mr", confidence=confidence)


def _compute_confidence(chunks: Sequence) -> float:
    scores = [chunk[2] for chunk in chunks if isinstance(chunk, (list, tuple)) and len(chunk) >= 3]
    return float(np.mean(scores)) if scores else 0.0


def _devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    devanagari_letters = [ch for ch in letters if 0x0900 <= ord(ch) <= 0x097F]
    return len(devanagari_letters) / len(letters)


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin_letters = [ch for ch in letters if ch.isascii()]
    return len(latin_letters) / len(letters)


def _to_bgr_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


class OCRService:
    def __init__(self) -> None:
        self.eng_engine = EasyEnglishOCREngine()
        self.mr_engine = MarathiOCREngine()
        self.eng_code = "en"
        self.mr_code = "mr"

    def _best_result(self, engine: BaseOCREngine, candidates: Sequence[np.ndarray]) -> OcrResult:
        best: OcrResult | None = None
        for candidate in candidates:
            result = engine.read(candidate)
            if best is None or result.confidence > best.confidence:
                best = result
        return best or OcrResult(text="", language="unknown", confidence=0.0)

    def read(self, image: Image.Image | np.ndarray, language_hint: str | None = None) -> OcrResult:

        base = _to_bgr_array(image)
        processed = preprocess_image(base.copy())
        candidates = [base, processed]

        if language_hint == self.mr_code:
            chosen = self._best_result(self.mr_engine, candidates)
            return OcrResult(text=chosen.text, language=self.mr_code, confidence=chosen.confidence)
        if language_hint == self.eng_code:
            chosen = self._best_result(self.eng_engine, candidates)
            return OcrResult(text=chosen.text, language=self.eng_code, confidence=chosen.confidence)

        marathi = self._best_result(self.mr_engine, candidates)
        english = self._best_result(self.eng_engine, candidates)

        marathi_script_ratio = _devanagari_ratio(marathi.text)
        english_latin_ratio = _latin_ratio(english.text)
        marathi_latin_ratio = _latin_ratio(marathi.text)

        marathi_lang_guess = detect_language(marathi.text)
        english_lang_guess = detect_language(english.text)

        if marathi_lang_guess == self.mr_code and marathi.confidence >= english.confidence - 0.05:
            return marathi
        if english_lang_guess == self.eng_code and english.confidence >= marathi.confidence - 0.05:
            return english

        if marathi_script_ratio >= 0.35 and english_latin_ratio < 0.45 and marathi.confidence >= english.confidence - 0.1:
            return marathi

        if english_latin_ratio >= 0.55 and english.confidence >= marathi.confidence - 0.15:
            return english

        if marathi.confidence >= english.confidence + 0.1 and marathi_script_ratio >= 0.25:
            return marathi
        if english.confidence >= marathi.confidence + 0.1 and english_latin_ratio >= 0.35:
            return english

        if marathi_script_ratio - marathi_latin_ratio >= 0.2 and marathi.confidence >= english.confidence:
            return marathi

        return english if english.confidence >= marathi.confidence else marathi


__all__ = ["OCRService", "OcrResult"]
