"""Real EasyOCR on real images. Skipped unless model weights are available (RAG_TEST_OCR_MODELS)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.core.container import Container
from src.core.errors import InvalidRequestError
from src.core.types import JobSpec

from tests.helpers import make_pdf, text_png

MODELS = Path(os.environ.get("RAG_TEST_OCR_MODELS", "/opt/ocr-models"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ocr,
    pytest.mark.skipif(
        not (MODELS / "english_g2.pth").exists(), reason=f"EasyOCR weights not found in {MODELS}"
    ),
]

LINES = ["Quarterly revenue grew 12 percent", "Operating margin improved to 18 percent"]


@pytest.fixture
async def ocr_container(make_settings, rag_config, run_id):
    settings = make_settings(
        ocr_enabled=True, ocr_model_dir=MODELS, ocr_download_models=False, ocr_gpu_enabled=False
    )
    built = await Container.build(settings, role="api", with_ingestion=True, config=rag_config)
    await built.start()
    try:
        yield built
    finally:
        names = list(await built.elastic.client.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            await built.elastic.client.indices.delete(index=names, ignore_unavailable=True)
        await built.close()


async def test_standalone_skewed_image_is_read_in_order_with_real_confidence(
    ocr_container: Container, tmp_path: Path
):
    image = tmp_path / "scan.png"
    image.write_bytes(text_png(LINES, angle=6.0))
    summary = await ocr_container.ingestion.ingest(
        JobSpec(collection="alpha", path=str(image), image_language="en")
    )
    assert summary.image_chunks >= 1 and summary.parser == "image"

    result = await ocr_container.retrieval.retrieve(
        "operating margin improved", ocr_container.retrieval.scope(["alpha"], ["image"])
    )
    top = result.documents[0]
    assert top.metadata["type"] == "image" and top.kind == "image"
    assert top.content.index("Quarterly") < top.content.index("Operating"), "deskew keeps reading order"
    assert top.metadata["ocr_confidence"] > 0.5, "confidence is real (it was always 0.0 with paragraph=True)"
    assert top.metadata["language"] == "en"


async def test_image_repeated_across_pdf_pages_is_ocred_once(
    ocr_container: Container, tmp_path: Path, monkeypatch
):
    png = text_png(LINES)
    pdf = make_pdf(
        tmp_path / "logo.pdf",
        ["Page one body text here about the company logo.", "Page two body text again."],
        image_on_page=1,
        image_png=png,
    )
    # also place the same image on page 2 by building a second document that references it twice
    import pymupdf

    doc = pymupdf.open(pdf)
    doc[1].insert_image(pymupdf.Rect(72, 400, 372, 520), stream=png)
    doc.saveIncr()
    doc.close()

    reads: list[int] = []
    engine_ocr = ocr_container.ingestion._ocr  # noqa: SLF001
    original = engine_ocr.read

    async def counting(image, hint):
        reads.append(len(image))
        return await original(image, hint)

    monkeypatch.setattr(engine_ocr, "read", counting)
    summary = await ocr_container.ingestion.ingest(
        JobSpec(collection="alpha", path=str(pdf), image_language="en")
    )
    assert summary.image_chunks >= 1
    assert len(reads) == 1, f"the same embedded image must be OCR'd once, got {len(reads)} reads"


async def test_ocr_is_not_loaded_unless_images_are_selected(ocr_container: Container, tmp_path: Path):
    pdf = make_pdf(
        tmp_path / "text.pdf",
        ["Plain text page with enough words to form a chunk of text."],
        image_on_page=1,
        image_png=text_png(LINES),
    )
    summary = await ocr_container.ingestion.ingest(
        JobSpec(collection="alpha", path=str(pdf), kinds=("text",))
    )
    assert summary.image_chunks == 0 and summary.text_chunks >= 1
    assert ocr_container.ingestion._ocr._engine is None, "no OCR model was loaded"  # noqa: SLF001


async def test_unsupported_ocr_language_is_rejected_not_defaulted(ocr_container: Container, tmp_path: Path):
    image = tmp_path / "scan.png"
    image.write_bytes(text_png(LINES))
    with pytest.raises(InvalidRequestError):
        await ocr_container.ingestion.ingest(
            JobSpec(collection="alpha", path=str(image), image_language="fr")
        )
