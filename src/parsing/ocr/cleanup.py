"""OCR post-processing for English, Hindi and Marathi.

Every rule here is *conservative*: text that the OCR engine read correctly must come out
unchanged. An earlier version rewrote letter pairs inside every word (``rn`` -> ``m``,
``cl`` -> ``d``: "learn" became "leam"), replaced every ASCII ``l`` and ``0`` in Devanagari
text, dropped symbols such as ``&`` and ``%``, and inserted a space after every ``.`` (so URLs
and e-mail addresses were split). Those substitutions damaged correct text far more often than
they repaired OCR mistakes, so they are gone. What remains fixes whitespace, stray control
characters, obvious punctuation runs, Devanagari mark spacing and isolated noise glyphs.
"""

from __future__ import annotations

import re

#: Single glyphs OCR engines emit for specks, rules and borders; never meaningful on their own.
_NOISE_GLYPHS = frozenset("|¦~`_^")


def clean_ocr_text(text: str, language: str = "en") -> str:
    """Normalise OCR output without altering correctly read words.

    Args:
        text: Raw OCR text.
        language: ``en``, ``mr`` or ``hi``.
    """
    if not text or not text.strip():
        return ""
    text = remove_control_chars(normalize_whitespace(text))
    if language in ("mr", "hi"):
        text = fix_devanagari_errors(text)
    text = fix_punctuation(text)
    return remove_noise_chars(text).strip()


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/blank lines, tidy spacing around sentence punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # no space before closing punctuation: "word ." -> "word."
    text = re.sub(r" +([।,.!?:;])", r"\1", text)
    # a sentence end glued to the next sentence: "ended.Next" -> "ended. Next". Only a lower-case
    # letter followed by an upper-case one, so "U.S.A", "e.g.", URLs, versions and e-mail
    # addresses are left alone.
    text = re.sub(r"(?<=[a-z\u0900-\u097F])([.!?])(?=[A-Z])", r"\1 ", text)
    # danda glued to the next word
    text = re.sub(r"(।)(?=\w)", r"\1 ", text)
    # comma / semicolon glued to a following letter ("a,b" -> "a, b"); never before digits (1,000)
    return re.sub(r"([,;])(?=[^\W\d_])", r"\1 ", text)


def remove_control_chars(text: str) -> str:
    """Drop non-printable control characters (keeping newlines and tabs)."""
    return "".join(ch for ch in text if ch in "\n\t" or (ord(ch) >= 32 and ord(ch) != 127))


def fix_devanagari_errors(text: str) -> str:
    """Spacing fixes for Devanagari marks and a doubled danda."""
    text = text.replace("।।", "।")
    # A combining mark that drifted away from its base letter: remove the space *before* it. (A space
    # after a mark is an ordinary word boundary - "करोड़ रहा" - and must stay.)
    return re.sub("\\s+([\u093c\u0901\u0902\u0903])", r"\1", text)  # nukta, chandrabindu, anusvara, visarga


def fix_punctuation(text: str) -> str:
    """Collapse runs of dots and commas."""
    text = re.sub(r"\.{4,}", "...", text)  # keep an ellipsis, collapse "........"
    return re.sub(r",{2,}", ",", text)


def remove_noise_chars(text: str) -> str:
    """Remove tokens consisting of a single noise glyph. Letters, digits and symbols such as
    ``$ % & + - = / #`` are real content and are kept."""
    lines = []
    for line in text.split("\n"):
        lines.append(" ".join(word for word in line.split() if word not in _NOISE_GLYPHS))
    return "\n".join(lines)


__all__ = [
    "clean_ocr_text",
    "fix_devanagari_errors",
    "fix_punctuation",
    "normalize_whitespace",
    "remove_control_chars",
    "remove_noise_chars",
]
