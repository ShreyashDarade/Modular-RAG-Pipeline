from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.core.errors import ParseError
from src.ports.parsing import ImageRef, ParsedUnit


class ImageParser:
    """A standalone image (or each frame of a multi-page TIFF) is one unit needing OCR."""

    name = "image"
    extensions = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})

    def __init__(self, settings: object | None = None) -> None:
        pass

    def iter_units(self, path: Path) -> Iterator[ParsedUnit]:
        import io

        from PIL import Image, ImageSequence, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                frames = getattr(image, "n_frames", 1)
                if frames == 1:
                    yield ParsedUnit(unit=1, images=[ImageRef(label=path.name, data=path.read_bytes())])
                    return
                for number, frame in enumerate(ImageSequence.Iterator(image), start=1):
                    buffer = io.BytesIO()
                    frame.convert("RGB").save(buffer, format="PNG")
                    yield ParsedUnit(
                        unit=number, images=[ImageRef(label=f"{path.name}#{number}", data=buffer.getvalue())]
                    )
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            raise ParseError(f"{path.name}: cannot read image: {exc}") from exc
