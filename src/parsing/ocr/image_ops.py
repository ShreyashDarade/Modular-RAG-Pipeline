"""Image decoding and OCR pre-processing (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.errors import ParseError

#: Search range of the skew estimator; larger rotations are not corrected (rotated page, vertical text).
MAX_DESKEW_DEGREES = 15.0


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (PNG/JPEG/TIFF/BMP/WebP/...) to a BGR array."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ParseError("image could not be decoded")
    return image


def limit_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longer edge is at most ``max_side`` (keeps memory and OCR time bounded)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def estimate_skew(image: np.ndarray) -> float:
    """Skew of the text in degrees, within +/-``MAX_DESKEW_DEGREES``. Positive means the text is
    rotated counter-clockwise; rotating the image by the *returned* angle (cv2 convention)
    straightens it.

    Projection profile: binarise (Otsu), then find the rotation that makes the rows of ink
    sharpest - lines of text give large row-to-row jumps only when they are horizontal. It is
    accurate on short snippets (a bounding-box estimate such as ``minAreaRect`` is not, and its
    angle range has changed across OpenCV versions) and costs ~10 ms on a downscaled copy.
    Returns 0 for blank images or when no rotation improves on the original.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(ink) < 50:
        return 0.0
    height, width = ink.shape
    if (scale := 600.0 / max(height, width)) < 1.0:
        ink = cv2.resize(ink, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = ink.shape

    def sharpness(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        rows = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_LINEAR).sum(
            axis=1, dtype=np.float64
        )
        return float(np.sum(np.diff(rows) ** 2))

    coarse = max(np.arange(-MAX_DESKEW_DEGREES, MAX_DESKEW_DEGREES + 1e-9, 1.0), key=sharpness)
    best = float(max(np.arange(coarse - 1.0, coarse + 1.0 + 1e-9, 0.1), key=sharpness))
    return best if sharpness(best) > sharpness(0.0) * 1.01 else 0.0


def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre on an enlarged white canvas so no corner is cropped."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width, new_height = int(height * sin + width * cos), int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - width / 2.0
    matrix[1, 2] += new_height / 2.0 - height / 2.0
    return cv2.warpAffine(
        image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


def deskew_image(image: np.ndarray) -> np.ndarray:
    angle = estimate_skew(image)
    if abs(angle) <= 0.5:
        return image
    return rotate_expand(image, angle)


def preprocess_image(image: np.ndarray, max_length: int = 1280) -> np.ndarray:
    """Contrast + noise clean-up applied before the (second) OCR pass."""
    if image.size == 0:
        raise ParseError("empty image")
    working = limit_side(image, max_length)
    working = cv2.normalize(working, np.empty_like(working), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    working = cv2.fastNlMeansDenoisingColored(
        working, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
    return deskew_image(enhanced)
