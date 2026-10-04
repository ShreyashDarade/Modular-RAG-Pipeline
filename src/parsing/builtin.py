from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registries
from src.parsing.docx import DocxParser
from src.parsing.html import HtmlParser
from src.parsing.image import ImageParser
from src.parsing.pdf import PdfParser
from src.parsing.tabular import CsvParser, XlsxParser
from src.parsing.text import TextParser

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.ports.parsing import OcrEngine


def _easyocr(settings: Settings) -> OcrEngine:
    # imported on use: numpy / OpenCV / torch belong to the worker image only
    from src.parsing.ocr.easyocr_engine import EasyOcrEngine

    return EasyOcrEngine(settings)


def register_builtin_parsers(registries: Registries) -> None:
    for parser in (PdfParser, ImageParser, TextParser, HtmlParser, CsvParser, XlsxParser, DocxParser):
        registries.parsers.register(parser.name, parser)
    registries.ocr_engines.register("easyocr", _easyocr)
