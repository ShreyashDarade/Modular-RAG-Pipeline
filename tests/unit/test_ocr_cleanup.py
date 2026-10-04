"""OCR cleanup must never damage text the engine read correctly."""

from __future__ import annotations

import pytest
from src.parsing.ocr.cleanup import (
    clean_ocr_text,
    fix_devanagari_errors,
    fix_punctuation,
    normalize_whitespace,
    remove_control_chars,
    remove_noise_chars,
)

CORRECT_ENGLISH = [
    "turn learn class clear concern vivid environment",  # contain rn / cl / vv inside real words
    "Price $5 & 20% - done + more = result / total # 7 @ home",
    "Visit www.example.com or mail john.doe@example.org for v1.2.3 (see e.g. section 3.5)",
    "The U.S.A. economy grew 3.5% in 2020, and 1,000 firms were surveyed; see Table 2.",
    "Appendix B lists plan A and plan C, plus x-axis values.",
    "http://localhost:8000/docs?x=1&y=2",
    "Line one.\n\nLine two after a blank line.",
]
CORRECT_DEVANAGARI = [
    "वर्ष 2020 में कंपनी का राजस्व 100 करोड़ रहा।",  # ASCII digits and a danda
    "महाराष्ट्र सरकारने 15% वाढ जाहीर केली. l ही रेषा नाही",  # a Latin letter is not a danda
    "यह पहला वाक्य है। यह दूसरा वाक्य है।",
    "प्रश्न: उत्तर क्या है? श्री म. च. व. जोशी",  # single consonant abbreviations are real tokens
]


@pytest.mark.parametrize("text", CORRECT_ENGLISH)
def test_correct_english_is_unchanged(text):
    assert clean_ocr_text(text, "en") == text


@pytest.mark.parametrize("language", ["hi", "mr"])
@pytest.mark.parametrize("text", CORRECT_DEVANAGARI)
def test_correct_devanagari_is_unchanged(text, language):
    assert clean_ocr_text(text, language) == text


def test_the_old_destructive_rewrites_are_gone():
    assert clean_ocr_text("turn learn class", "en") == "turn learn class"
    assert clean_ocr_text("2020 l 100", "mr") == "2020 l 100"
    assert clean_ocr_text("Table 0x1F l", "en") == "Table 0x1F l"


def test_real_ocr_defects_are_still_repaired():
    assert normalize_whitespace("too    many   spaces") == "too many spaces"
    assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize_whitespace("word .") == "word." and normalize_whitespace("end,next") == "end, next"
    assert normalize_whitespace("ended.Next sentence") == "ended. Next sentence"
    assert normalize_whitespace("यह है।अगला") == "यह है। अगला"
    assert fix_devanagari_errors("हैं।।") == "हैं।"
    assert fix_devanagari_errors("ड \u093c") == "ड\u093c", "a detached nukta is re-attached to its letter"
    assert fix_devanagari_errors("100 करोड़ रहा") == "100 करोड़ रहा", (
        "a word-final nukta is not glued to the next word"
    )
    assert fix_devanagari_errors("हिन् दी ं") == "हिन् दीं"
    assert fix_punctuation("wait........ what,,, ok") == "wait... what, ok"
    assert remove_control_chars("a\x00b\x07c\nd\te\x7f") == "abc\nd\te"


def test_noise_glyphs_are_dropped_but_content_symbols_are_kept():
    assert remove_noise_chars("text | more ~ words ` end _ ^") == "text more words end"
    assert remove_noise_chars("5 % & $ + - = # B x") == "5 % & $ + - = # B x"
    assert remove_noise_chars("line1 |\n~ line2") == "line1\nline2"


def test_blank_input():
    assert clean_ocr_text("", "en") == "" and clean_ocr_text("  \n ", "hi") == ""
