from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

os.environ.setdefault("PYMUPDF_NO_FRONTEND", "1")

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

from src.utils.language import detect_language


@dataclass
class PdfTextBlock:
    page: int
    content: str
    language: str


@dataclass
class PdfTable:
    page: int
    dataframe: pd.DataFrame
    language: str


@dataclass
class PdfImage:
    page: int
    image: Image.Image
    label: str


@dataclass
class PdfExtractionResult:
    text_blocks: List[PdfTextBlock]
    tables: List[PdfTable]
    images: List[PdfImage]


def extract_from_pdf(path: Path) -> PdfExtractionResult:
    doc = fitz.open(path)
    text_blocks: List[PdfTextBlock] = []
    tables: List[PdfTable] = []
    images: List[PdfImage] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        language = detect_language(text)
        if text.strip():
            text_blocks.append(
                PdfTextBlock(page=page_index + 1, content=text, language=language)
            )

        try:
            table_finder = page.find_tables()
            for table_index, table in enumerate(table_finder.tables):
                df = table.to_pandas()
                lang = detect_language("\n".join(df.astype(str).stack().tolist()))
                tables.append(PdfTable(page=page_index + 1, dataframe=df, language=lang))
        except Exception:
            # Table detection may fail on some PDFs; ignore.
            pass

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            label = f"pdf-page-{page_index + 1}-img-{img_index + 1}"
            images.append(PdfImage(page=page_index + 1, image=pil_img, label=label))

    return PdfExtractionResult(text_blocks=text_blocks, tables=tables, images=images)


__all__ = [
    "PdfExtractionResult",
    "PdfTextBlock",
    "PdfTable",
    "PdfImage",
    "extract_from_pdf",
]
