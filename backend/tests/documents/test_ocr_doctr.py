"""
tests/documents/test_ocr_doctr.py
===================================
Vision + OCR tests for the docTR adapter and grid OCR helpers.

Creates synthetic images in-memory — no PDF fixtures or external files needed.
Tests exercise the actual docTR model end-to-end for recognize() and page_text().

Run:
    pytest tests/documents/test_ocr_doctr.py -v
    pytest tests/documents/test_ocr_doctr.py -k "TestRecognize" -v     # just cell OCR
    pytest tests/documents/test_ocr_doctr.py -k "TestPageText" -v      # full page OCR
    pytest tests/documents/test_ocr_doctr.py -k "TestGridHelpers" -v   # number/header repair
"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Helpers — generate test images with known text
# ---------------------------------------------------------------------------

def _font(size=28):
    """Try to load a monospace font; fall back to PIL default."""
    for name in ("cour.ttf", "Courier New.ttf", "DejaVuSansMono.ttf",
                 "consola.ttf", "LiberationMono-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_crop(text: str, width=200, height=48, bg=255, fg=0) -> np.ndarray:
    """Render one cell of text as a grayscale numpy array (H, W)."""
    img = Image.new("L", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), text, fill=fg, font=_font(max(14, height - 12)))
    return np.asarray(img)


def make_rgb_crop(text: str, width=200, height=48) -> np.ndarray:
    """Render one cell as an RGB numpy array (H, W, 3)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), text, fill=(0, 0, 0), font=_font(max(14, height - 12)))
    return np.asarray(img)


def make_page(lines: list[str], width=700, line_height=40) -> Image.Image:
    """Render a multi-line page as a PIL Image."""
    height = max(120, line_height * (len(lines) + 2))
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _font(max(14, line_height - 10))
    for i, line in enumerate(lines):
        draw.text((12, 10 + i * line_height), line, fill=(0, 0, 0), font=font)
    return img


def save_temp_image(img: Image.Image, suffix=".png") -> str:
    """Save to a temp file; caller must delete after use."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(f.name)
    f.close()
    return f.name


# ===================================================================
# 1. _rgb preprocessing
# ===================================================================

class TestRgbPreprocessing:
    """ocr_doctr._rgb converts any image to uint8 RGB for docTR."""

    def test_rgb_passthrough(self):
        from documents.ocr_doctr import _rgb
        arr = np.zeros((20, 30, 3), dtype=np.uint8)
        out = _rgb(arr)
        assert out.shape == (20, 30, 3) and out.dtype == np.uint8

    def test_grayscale_to_rgb(self):
        from documents.ocr_doctr import _rgb
        arr = np.zeros((20, 30), dtype=np.uint8)
        out = _rgb(arr)
        assert out.shape == (20, 30, 3)

    def test_rgba_drops_alpha(self):
        from documents.ocr_doctr import _rgb
        arr = np.zeros((20, 30, 4), dtype=np.uint8)
        out = _rgb(arr)
        assert out.shape == (20, 30, 3)

    def test_float_cast(self):
        from documents.ocr_doctr import _rgb
        arr = np.full((20, 30, 3), 128.5, dtype=np.float32)
        out = _rgb(arr)
        assert out.dtype == np.uint8


# ===================================================================
# 2. recognize() — cell-level OCR
# ===================================================================

class TestRecognize:
    """Test docTR recognition on synthetic cell crops."""

    def test_empty_list_returns_empty(self):
        from documents.ocr_doctr import recognize
        assert recognize([]) == []

    def test_single_digit(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("8", width=60, height=48)
        results = recognize([crop])
        assert len(results) == 1
        text, score = results[0]
        assert isinstance(text, str) and len(text) >= 1
        assert 0.0 <= score <= 1.0

    def test_multi_digit_number(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("12345", width=120, height=48)
        text, score = recognize([crop])[0]
        # Should contain digits
        assert any(c.isdigit() for c in text), f"Expected digits in: {text!r}"

    def test_number_with_comma(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("1,234", width=120, height=48)
        text, _ = recognize([crop])[0]
        assert any(c.isdigit() for c in text)

    def test_percentage(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("45.6%", width=120, height=48)
        text, _ = recognize([crop])[0]
        # Should contain digits; % may or may not survive OCR
        assert any(c.isdigit() for c in text)

    def test_negative_number(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("-892", width=100, height=48)
        text, _ = recognize([crop])[0]
        assert any(c.isdigit() for c in text)

    def test_text_label(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("Revenue", width=160, height=48)
        text, score = recognize([crop])[0]
        assert isinstance(text, str) and len(text) > 0
        assert score > 0.0

    def test_batch_of_three(self):
        from documents.ocr_doctr import recognize
        crops = [
            make_crop("ABC", width=100, height=48),
            make_crop("123", width=100, height=48),
            make_crop("XYZ", width=100, height=48),
        ]
        results = recognize(crops)
        assert len(results) == 3
        assert all(isinstance(t, str) and isinstance(s, float) for t, s in results)

    def test_rgb_crop_accepted(self):
        from documents.ocr_doctr import recognize
        crop = make_rgb_crop("5678", width=120, height=48)
        results = recognize([crop])
        assert len(results) == 1

    def test_inverted_crop(self):
        """White text on dark background (dark band headers)."""
        from documents.ocr_doctr import recognize
        crop = make_crop("Total", width=120, height=48, bg=40, fg=240)
        text, _ = recognize([crop])[0]
        assert isinstance(text, str) and len(text) > 0

    def test_indian_lakh_number(self):
        from documents.ocr_doctr import recognize
        crop = make_crop("1,06,532", width=160, height=48)
        text, _ = recognize([crop])[0]
        assert any(c.isdigit() for c in text)

    def test_large_batch(self):
        """Batch of 20 crops shouldn't crash."""
        from documents.ocr_doctr import recognize
        crops = [make_crop(str(i * 111), width=100, height=48) for i in range(20)]
        results = recognize(crops)
        assert len(results) == 20

    def test_tiny_crop(self):
        """Very small crop (edge case)."""
        from documents.ocr_doctr import recognize
        crop = np.full((8, 20), 200, dtype=np.uint8)
        results = recognize([crop])
        assert len(results) == 1  # Should not crash


# ===================================================================
# 3. page_text() — full page OCR
# ===================================================================

class TestPageText:
    """Test docTR full-page OCR on synthetic page images."""

    def test_single_line(self):
        from documents.ocr_doctr import page_text
        path = save_temp_image(make_page(["Hello World"]))
        try:
            result = page_text(path)
        finally:
            os.unlink(path)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_multiple_lines(self):
        from documents.ocr_doctr import page_text
        path = save_temp_image(make_page([
            "Revenue: 1,500 Cr",
            "Expenses: 800 Cr",
            "Net Profit: 700 Cr",
        ]))
        try:
            result = page_text(path)
        finally:
            os.unlink(path)
        # Should return some text with multiple tokens
        assert len(result.split()) >= 3

    def test_blank_page(self):
        from documents.ocr_doctr import page_text
        blank = Image.new("RGB", (500, 300), color=(255, 255, 255))
        path = save_temp_image(blank)
        try:
            result = page_text(path)
        finally:
            os.unlink(path)
        assert len(result.strip()) < 20  # Blank or near-blank

    def test_numbers_page(self):
        from documents.ocr_doctr import page_text
        path = save_temp_image(make_page([
            "Jan-25    1,234    5.6%",
            "Feb-25    2,345    7.8%",
            "Mar-25    3,456    9.1%",
        ]))
        try:
            result = page_text(path)
        finally:
            os.unlink(path)
        # Should find some digits
        assert any(c.isdigit() for c in result)

    def test_table_like_page(self):
        from documents.ocr_doctr import page_text
        path = save_temp_image(make_page([
            "Metric        Q1     Q2     Q3     Q4",
            "Advances    1,200  1,350  1,500  1,620",
            "Deposits    2,400  2,500  2,600  2,750",
            "NIM          3.2%   3.1%   3.3%   3.4%",
        ]))
        try:
            result = page_text(path)
        finally:
            os.unlink(path)
        assert len(result) > 20


# ===================================================================
# 4. Grid OCR helpers — repair_number
# ===================================================================

class TestRepairNumber:
    """gridocr.repair_number fixes OCR misreads in numbers."""

    def test_valid_number_unchanged(self):
        from documents.gridocr import repair_number
        assert repair_number("1,234.56", commas=True) == "1,234.56"

    def test_comma_point_swap(self):
        from documents.gridocr import repair_number
        # '3.723.6' → multiple decimal points; the last one is the decimal
        result = repair_number("3.723.6", commas=True)
        assert result == "3,723.6"

    def test_lakh_format(self):
        from documents.gridocr import repair_number
        result = repair_number("8.83,623", commas=True)
        assert result == "8,83,623"

    def test_trailing_pipe(self):
        from documents.gridocr import repair_number
        result = repair_number("4,035|", commas=True)
        assert result == "4,035"

    def test_letter_B_as_digit_8(self):
        from documents.gridocr import repair_number
        result = repair_number("1,0B6", commas=True)
        assert result == "1,086"

    def test_already_valid_percentage(self):
        from documents.gridocr import repair_number
        assert repair_number("45.6%", commas=False) == "45.6%"

    def test_negative_preserved(self):
        from documents.gridocr import repair_number
        assert repair_number("-1,234", commas=True) == "-1,234"

    def test_parenthesized_negative(self):
        from documents.gridocr import repair_number
        result = repair_number("(1,234)", commas=True)
        # Should stay parseable
        assert "1" in result and "234" in result

    def test_pure_text_unchanged(self):
        from documents.gridocr import repair_number
        assert repair_number("Revenue", commas=True) == "Revenue"

    def test_empty_string(self):
        from documents.gridocr import repair_number
        assert repair_number("", commas=True) == ""


# ===================================================================
# 5. Grid OCR helpers — repair_months
# ===================================================================

class TestRepairMonths:
    """gridocr.repair_months fixes misread month headers in sequence."""

    def test_fixes_damaged_month(self):
        from documents.gridocr import repair_months
        header = ["Label", "Jan-25", "eb-25", "Mar-25", "Apr-25", "May-25"]
        result = repair_months(header)
        assert result[2] == "Feb-25"

    def test_fixes_clipped_month(self):
        from documents.gridocr import repair_months
        header = ["Label", "Jan-25", "Feb-25", "Mar-25", "pr-25", "May-25"]
        result = repair_months(header)
        assert result[4] == "Apr-25"

    def test_too_few_months_unchanged(self):
        from documents.gridocr import repair_months
        header = ["Col A", "Jan-25"]
        assert repair_months(header) == header

    def test_correct_sequence_unchanged(self):
        from documents.gridocr import repair_months
        header = ["Label", "Jan-25", "Feb-25", "Mar-25"]
        result = repair_months(header)
        assert result == header

    def test_uppercase_preserved(self):
        from documents.gridocr import repair_months
        header = ["Label", "JAN-25", "EB-25", "MAR-25", "APR-25"]
        result = repair_months(header)
        assert result[2] == "FEB-25"


# ===================================================================
# 6. Grid OCR helpers — clean
# ===================================================================

class TestClean:
    """gridocr.clean handles OCR artefacts."""

    def test_chinese_dash(self):
        from documents.gridocr import clean
        assert clean("\u4e00") == "-"

    def test_em_dash(self):
        from documents.gridocr import clean
        assert clean("\u2014") == "-"

    def test_underscore_blank(self):
        from documents.gridocr import clean
        assert clean("_") == ""

    def test_pipe_blank(self):
        from documents.gridocr import clean
        assert clean("|") == ""

    def test_normal_text_passes(self):
        from documents.gridocr import clean
        assert clean("Revenue") == "Revenue"

    def test_number_passes(self):
        from documents.gridocr import clean
        assert clean("1,234") == "1,234"


# ===================================================================
# 7. Grid OCR helpers — merge_sparse_columns
# ===================================================================

class TestMergeSparseColumns:
    """gridocr.merge_sparse_columns folds artefact columns."""

    def test_sparse_column_merged(self):
        from documents.gridocr import merge_sparse_columns
        grid = [
            ["Label",  "",     "100"],
            ["Row A",  "n.m.", "200"],
            ["Row B",  "",     "300"],
            ["Row C",  "",     "400"],
            ["Row D",  "",     "500"],
        ]
        result = merge_sparse_columns(grid)
        # The sparse middle column should be absorbed
        assert len(result[0]) <= len(grid[0])

    def test_all_full_columns_unchanged(self):
        from documents.gridocr import merge_sparse_columns
        grid = [
            ["A", "B", "C"],
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
        ]
        result = merge_sparse_columns(grid)
        assert len(result[0]) == 3

    def test_single_column_unchanged(self):
        from documents.gridocr import merge_sparse_columns
        grid = [["A"], ["B"]]
        assert merge_sparse_columns(grid) == grid


# ===================================================================
# 8. Grid OCR helpers — Cell class
# ===================================================================

class TestCellClass:
    """gridocr.Cell preserves OCR metadata."""

    def test_cell_creation(self):
        from documents.gridocr import Cell
        c = Cell.of("1,234", score=0.95)
        assert str(c) == "1,234"
        assert c.score == 0.95

    def test_cell_keep_carries_metadata(self):
        from documents.gridocr import Cell
        c = Cell.of("1.234", score=0.88, crop=np.zeros((10, 20)))
        fixed = c.keep("1,234")
        assert isinstance(fixed, Cell)
        assert fixed.score == 0.88

    def test_cell_is_string(self):
        from documents.gridocr import Cell
        c = Cell.of("hello", score=1.0)
        assert c.upper() == "HELLO"
        assert "ell" in c

    def test_cell_alts(self):
        from documents.gridocr import Cell
        c = Cell.of("1O6", score=0.7, alts=("106", "1O6"))
        assert c.alts == ("106", "1O6")


# ===================================================================
# 9. Grid OCR helpers — is_total / fix_label_misreads
# ===================================================================

class TestLabelHelpers:

    def test_is_total_variations(self):
        from documents.gridocr import is_total
        assert is_total("Total") is True
        assert is_total("Grand Total") is True
        assert is_total("Subtotal") is True
        assert is_total("  Total ") is True
        assert is_total("Revenue") is False
        assert is_total("") is False

    def test_fix_B_code_misread(self):
        from documents.gridocr import fix_label_misreads
        rows = [["B2", "100"], ["81", "200"], ["B3", "300"]]
        fix_label_misreads(rows)
        assert rows[1][0] == "B1"

    def test_fix_underscore_band(self):
        from documents.gridocr import fix_label_misreads
        rows = [["0_12", "100"], ["13_24", "200"], ["25_36", "300"], ["3748", "400"]]
        fix_label_misreads(rows)
        assert rows[3][0] == "37_48"


# ===================================================================
# 10. Grid OCR helpers — normalize_numbers
# ===================================================================

class TestNormalizeNumbers:

    def test_point_groups_to_commas(self):
        from documents.gridocr import normalize_numbers
        grid = [
            ["Label", "1,463"],
            ["Row A", "7.982"],
            ["Row B", "2,150"],
        ]
        normalize_numbers(grid)
        # 7.982 in a comma-grouped column should become 7,982
        assert grid[1][1] == "7,982"

    def test_decimal_restored(self):
        from documents.gridocr import normalize_numbers
        grid = [
            ["Label", "60.4"],
            ["Row A", "581"],
            ["Row B", "58.1"],
            ["Row C", "55.2"],
        ]
        normalize_numbers(grid)
        # 581 in a column of single-decimal numbers should become 58.1
        assert "." in grid[1][1]


# ===================================================================
# 11. docTR model loading
# ===================================================================

class TestModelLoading:

    def test_recognizer_loads(self):
        from documents.ocr_doctr import recognizer
        rec = recognizer()
        assert rec is not None

    def test_recognizer_singleton(self):
        from documents.ocr_doctr import recognizer
        r1 = recognizer()
        r2 = recognizer()
        assert r1 is r2

    def test_page_engine_loads(self):
        from documents.ocr_doctr import page_engine
        eng = page_engine()
        assert eng is not None

    def test_page_engine_singleton(self):
        from documents.ocr_doctr import page_engine
        e1 = page_engine()
        e2 = page_engine()
        assert e1 is e2
