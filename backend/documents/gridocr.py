"""Read picture tables without the vision model: find the cell grid from pixels (picture.py),
then OCR every cell on its own with the selected recognizer (docTR by default; RapidOCR retained
for rollback). With no text detection at this stage, dense rows cannot merge.
tables.extract_page_tables calls gridocr_tables for every picture table.
Run on its own it scores grid OCR alone (no vision fallback) with the checks of eval_tables.py:
    python gridocr.py --pdf tests/fixtures/tables_image_150.pdf --pages 12
    python gridocr.py both image fixture PDFs
    python gridocr.py --real input/CC12.pdf --pages 9 10 11 12 13
options: --rec small|medium --text-px 32 (cell text height given to the recognizer) --dump (save cell grids)
Writes data/table_eval_gridocr.md; does not touch data/table_eval.md.
"""
import argparse
import difflib
import json
import os
import time
from pathlib import Path

import numpy as np
import pymupdf
import re

from . import config
from . import picture
from . import tables
from .tables import LEFT, make_table, parse_number, pop_title_row, split_stacked, structure

# OCR backend: docTR by default; RapidOCR retained for rollback.
# Paddle/RapidOCR is intentionally inactive. To revert, comment the line above and uncomment this:
#   from . import ocr_rapidocr as ocr_backend
try:
    from . import ocr_doctr as ocr_backend
except ImportError:
    ocr_backend = None  # OCR backend not available; gridocr_tables will fail at runtime

# eval_tables is only needed for the standalone CLI (main()); not for app integration.
try:
    from . import eval_tables
except ImportError:
    eval_tables = None

OCR_BACKEND = ocr_backend.NAME if ocr_backend else "none"

# Settings when imported by the app; the command line replaces them.
ARGS = argparse.Namespace(
    rec=os.getenv("GRIDOCR_REC", "small"),
    text_px=float(os.getenv("GRIDOCR_TEXT_PX", "32")),
    dump=False,
)


class Cell(str):
    """Cell text that remembers how it was read: the recognizer's score and the crop, so a doubtful
    number can be read again later. String operations return plain str; use ``keep`` to carry it."""
    score: float = 1.0
    crop: np.ndarray | None = None
    alts: tuple[str, ...] = ()  # other readings of the same crop, best first

    @classmethod
    def of(cls, text: str, score: float = 1.0, crop=None, alts=()) -> "Cell":
        c = cls(text)
        c.score, c.crop, c.alts = score, crop, tuple(alts)
        return c

    def keep(self, text: str) -> str:
        return text if text == self else Cell.of(text, self.score, self.crop, self.alts)


def recognize(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    if ocr_backend is None:
        raise RuntimeError("No OCR backend available (ocr_doctr not installed)")
    return ocr_backend.recognize(crops)


def page_text(image_path: str) -> str:
    """Whole-page OCR for scans, using the selected backend."""
    if ocr_backend is None:
        raise RuntimeError("No OCR backend available (ocr_doctr not installed)")
    return ocr_backend.page_text(image_path)


def _runs(xs: list[int]) -> list[tuple[int, int]]:
    runs = []
    for x in xs:
        if runs and x == runs[-1][1]:
            runs[-1] = (runs[-1][0], x + 1)
        else:
            runs.append((x, x + 1))
    return runs


def thin_rules(g: picture.Gray, box, lay: picture.Layout) -> set[int]:
    """Vertical rules: tall marks a few px wide. A wider run is a filled area (a coloured label
    column), which holds text and must not be cut."""
    x0, y0, x1, y1 = box
    width = max(3, 0.3 * lay.text_px)

    def light_side(x):  # mostly light: outside any fill
        if not 0 <= x < x1 - x0:
            return True
        col = g.cols[x0 + x][y0:y1]
        return sum(1 for v in col if v < 200) < 0.5 * len(col)

    return {x for a, b in _runs(picture.vertical_rules(g, box))
            if b - a <= width and light_side(a - 2) and light_side(b + 1)
            for x in range(a, b)}


def column_splits(g: picture.Gray, box, lay: picture.Layout, rules: set[int]) -> list[int]:
    """x (block coordinates) blank in (almost) every body line, or on a thin rule."""
    x0, y0, x1, _ = box
    lines = [(a, b) for a, b in lay.texts if a >= lay.header_end] or lay.texts
    votes = [0 if x in rules
             else sum(1 for a, b in lines if min(g.cols[x0 + x][y0 + a:y0 + b]) < 200)
             for x in range(x1 - x0)]
    n = max(1, max(votes, default=1))
    return [x for x, v in enumerate(votes) if x in rules or v <= 0.02 * n]


def columns(g: picture.Gray, box, lay: picture.Layout) -> list[tuple[int, int]]:
    """Column x-ranges (block coordinates): cut at vertical rules and at blank runs wider than a
    word space, so 'Branch 01' stays one cell."""
    w = box[2] - box[0]
    rules = thin_rules(g, box, lay)
    min_gap = max(3, 0.8 * lay.text_px)
    cuts = []
    for a, b in _runs(column_splits(g, box, lay, rules)):
        if a == 0:
            continue  # margin
        if b == w:
            # A margin, unless a rule inside it closes off a column left blank in the body (a
            # month with no figures yet): cut at the last such rule, the header still has text.
            inner = [x for x in rules if a < x < w - 2 * lay.text_px]
            if inner:
                cuts.append((a, max(inner) + 1))
            continue
        if any(a <= x < b for x in rules) or b - a >= min_gap:
            cuts.append((a, b))
    edges = [0] + [x for a, b in cuts for x in (a, b)] + [w]
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] > 1]


def rows(g: picture.Gray, box, lay: picture.Layout) -> list[tuple[int, int, bool]]:
    """Row y-ranges (block coordinates) and whether the row is light text on a dark band.
    A band holding two lines of header text gives two rows."""
    # A sliver 1-2 px from a line of text (an accent, a broken underline) belongs to that line.
    thin = 0.5 * lay.text_px
    merged = []
    for a, b in lay.texts:
        if merged and a - merged[-1][1] <= 2 and (b - a < thin or merged[-1][1] - merged[-1][0] < thin):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = [(a, b, False) for a, b in merged if b - a >= thin]
    x0, y0, x1, _ = box
    for a, b in lay.bands:
        # Look for light text only inside the dark fill (the white margin beside it is not text).
        mid = g.rows[y0 + (a + b) // 2][x0:x1]
        dk = [x for x, v in enumerate(mid) if v < 200]
        xa, xb = (x0 + dk[0] + 2, x0 + dk[-1] - 2) if dk else (x0, x1)
        bright = [sum(1 for v in g.rows[y0 + y][xa:xb] if v > 200) >= 2 for y in range(a, b)]
        lines = [(a + s, a + e) for s, e in _runs([i for i, v in enumerate(bright) if v]) if e - s >= thin]
        out += [(s, e, True) for s, e in lines] or [(a, b, True)]
    return sorted(out)


def _trim_rules(c: np.ndarray) -> np.ndarray:
    """Drop edge columns/rows that are one solid line (a grid rule caught in the crop)."""
    dark = c < 160
    while c.shape[1] > 2 and dark[:, 0].mean() > 0.7:
        c, dark = c[:, 1:], dark[:, 1:]
    while c.shape[1] > 2 and dark[:, -1].mean() > 0.7:
        c, dark = c[:, :-1], dark[:, :-1]
    while c.shape[0] > 2 and dark[0].mean() > 0.7:
        c, dark = c[1:], dark[1:]
    while c.shape[0] > 2 and dark[-1].mean() > 0.7:
        c, dark = c[:-1], dark[:-1]
    return c


def read_grid(fpage: pymupdf.Page, brect: pymupdf.Rect, dpi: float) -> tuple[list[list[str]], dict]:
    g = picture.Gray(fpage.get_pixmap(clip=brect, dpi=round(dpi), colorspace=pymupdf.csGRAY))
    box = (0, 0, g.w, g.h)
    lay = picture.layout(g, box)
    cols, rws = columns(g, box, lay), rows(g, box, lay)
    # Render finer for the recognizer: its input is 48 px tall, text ~ARGS.text_px reads best.
    k = max(1.0, min(4.0, ARGS.text_px / max(4.0, lay.text_px)))
    pix = fpage.get_pixmap(clip=brect, dpi=round(dpi * k), colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    # The finer render is not exactly k times larger (dpi is rounded): map with the real ratios,
    # or cells drift off their columns towards the right edge.
    kx, ky = pix.width / g.w, pix.height / g.h
    pad = max(1, round(0.25 * lay.text_px * ky))
    crops, where = [], []
    grid = [[""] * len(cols) for _ in rws]
    rules = thin_rules(g, box, lay)
    gaps = [(cols[j][1], cols[j + 1][0]) for j in range(len(cols) - 1)]
    spanning, span_marks = 0, []
    long_mark = max(40, 4 * lay.text_px)  # text never makes an unbroken mark this long
    for i, (a, b, inverted) in enumerate(rws):
        text_ys = ([y - a for y in range(a, b)
                    if inverted or picture.longest_mark(g.rows[y]) < long_mark]
                   or list(range(b - a)))
        mid_row = g.rows[(a + b) // 2]
        fill = sorted(mid_row)[len(mid_row) // 2]  # typical pixel of this row (band fill or paper)

        def inked(x):
            if x in rules:
                return False
            col = g.cols[x][a:b]
            v = [col[y] for y in text_ys]
            if inverted:
                # Light text on a dark band; a light line through the whole row is a cell border.
                n = sum(1 for p in v if p > 200)
                return 0 < n < 0.85 * len(v)
            return min(v) < 160

        def border(x):
            """A cell border drawn through this row (absent in a merged cell such as a title bar)."""
            if x not in rules:
                return False
            col = g.cols[x][a:b]
            v = [col[y] for y in text_ys]
            if inverted:  # a border on a dark band is clearly darker than the band's fill
                return max(v) <= 200 and sorted(v)[len(v) // 2] < fill - 30
            return sum(1 for p in v if p < 160) >= 0.8 * len(v)

        # Text running through a column gap (a title, a header spanning columns): read the
        # whole line and place each piece in the columns it covers. Text never crosses a border.
        crossed = [any(inked(x) for x in range(g0 + 1, g1 - 1))
                   and not any(border(x) for x in range(g0, g1))
                   for g0, g1 in gaps]
        if any(crossed):
            # Join only the columns whose gap this row's text runs through; inside each joined
            # run, cut at the row's own blank runs (wider than a word space) and give each piece
            # to the column it overlaps most. A piece covering most of several columns spans them.
            spanning += 1
            min_gap = max(3, 0.8 * len(text_ys))  # a word space scales with this row's own text
            pieces, j = [], 0
            while j < len(cols):
                j1 = j
                while j1 < len(gaps) and crossed[j1]:
                    j1 += 1
                x_from, x_to = cols[j][0], cols[j1][1]
                s0, blank = None, 0
                for x in range(x_from, x_to + int(min_gap) + 1):
                    if x < x_to and border(x) and s0 is not None:
                        pieces.append((s0, x - blank))
                        s0, blank = None, 0
                    elif x < x_to and inked(x):
                        if s0 is None:
                            s0 = x
                        blank = 0
                    elif s0 is not None:
                        blank += 1
                        if blank >= min_gap or x >= x_to:
                            pieces.append((s0, x - blank + 1))
                            s0, blank = None, 0
                j = j1 + 1
            for p0, p1 in pieces:
                ov = [max(0, min(p1, c1) - max(p0, c0)) for c0, c1 in cols]
                j0 = max(range(len(cols)), key=lambda j: ov[j])
                if ov[j0] == 0:
                    continue
                c = img[max(0, round(a * ky) - pad):round(b * ky) + pad,
                        max(0, round(p0 * kx) - pad):round(p1 * kx) + pad]
                dark_fill = inverted or (c.size and np.median(c) < 128)
                crops.append(np.ascontiguousarray(255 - c if dark_fill else c))
                where.append((i, j0))
                for j, (c0, c1) in enumerate(cols):
                    if j != j0 and ov[j] > 0.6 * (c1 - c0):
                        span_marks.append((i, j))
            continue
        for j, (c0, c1) in enumerate(cols):
            # Out to the middle of the gaps on both sides: a header is often wider than its numbers.
            c0 = (cols[j - 1][1] + c0) // 2 if j else c0
            c1 = (c1 + cols[j + 1][0]) // 2 if j + 1 < len(cols) else c1
            c = img[max(0, round(a * ky) - pad):round(b * ky) + pad,
                    round(c0 * kx):round(c1 * kx)]
            # Light text on a dark fill, as a whole band or just one cell (a coloured label column).
            if inverted or (c.size and np.median(c) < 128):
                c = 255 - c
            c = _trim_rules(c)
            if c.size == 0 or (c < 160).sum() < 3:
                continue  # blank cell
            crops.append(np.ascontiguousarray(c))
            where.append((i, j))
    for (i, j), (t, s), c in zip(where, recognize(crops), crops):
        grid[i][j] = (grid[i][j] + " " + t).strip() if grid[i][j] else Cell.of(t, s, c)
    for i, j in span_marks:
        if not grid[i][j]:
            grid[i][j] = LEFT
    grid = [[clean(c) for c in r] for r in grid]
    grid = [r for r in grid if any(c and c != LEFT for c in r)]
    grid = merge_sparse_columns(grid)
    normalize_numbers(grid)
    return grid, {"rows": len(rws), "cols": len(cols), "cells": len(crops),
                  "spanning_rows": spanning, "text_px": lay.text_px, "k": k}


def clean(t: str) -> str:
    if t == LEFT:
        return t
    # The recognizer is trained on Chinese too: a lone dash comes back as \u4e00 or \u4e8c.
    if re.fullmatch(r"[\u4e00\u4e8c\u2014\u2013\-\u00b7\u2022]+", t):  # a small dash can come back as a dot
        return t.keep("-") if isinstance(t, Cell) else "-"
    return "" if re.fullmatch(r"[_\s|]", t) else t


def merge_sparse_columns(grid: list[list[str]]) -> list[list[str]]:
    """A column that is empty in most rows is a split artefact (a centred dash or 'n.m.' next to
    right-aligned numbers): fold it into the neighbour it never clashes with."""

    def filled(r, j):
        return bool(r[j]) and r[j] != LEFT

    changed = True
    while changed and grid and len(grid[0]) > 2:
        changed = False
        n = len(grid[0])
        for j in range(1, n):
            share = sum(filled(r, j) for r in grid) / len(grid)
            if share >= 0.25:
                continue
            clash = {d: sum(filled(r, j) and filled(r, j + d) for r in grid)
                     for d in (-1, 1) if 0 < j + d < n}
            ok = [d for d, c in clash.items() if c == 0]
            if not ok:
                continue
            # Prefer the right neighbour: numbers are right-aligned, so stray pieces sit left of them.
            d = 1 if 1 in ok else ok[0]
            for r in grid:
                if filled(r, j):
                    r[j + d] = r[j] if not filled(r, j + d) else r[j + d]
                elif r[j + d] == LEFT and r[j] and r[j] != LEFT:
                    r[j + d] = r[j]
            for r in grid:
                del r[j]
            changed = True
            break
    return grid


def find_header(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Split off heading and paragraph lines above the table: the header is the first row filling
    half the columns with numbers below it; a row just above it with 2+ texts (FY spans) stays."""

    def texts(r):
        return [c for c in r if c and c != LEFT]

    def numeric(r):
        return sum(1 for c in r if parse_number(c) is not None) >= 0.3 * len(r)

    h = next((i for i, r in enumerate(grid)
              if len(texts(r)) >= 0.5 * len(r) and any(numeric(x) for x in grid[i + 1:i + 3])), 0)
    start, skip = h, None
    # A label-only row right above the header is the header's first cell spanning two header
    # rows ("Segment" beside FY24/Q1): read again in the header row, so drop it.
    if (start > 1 and len(texts(grid[start - 1])) == 1 and grid[start - 1][0]
            and len(texts(grid[start - 2])) >= 2):
        skip, start = start - 1, start - 1
    if start > 0 and len(texts(grid[start - 1])) >= 2:
        start -= 1
        spread_groups(grid[start], grid[start + 1])
    above = [" ".join(texts(r)) for r in grid[:start] if texts(r)]
    return above, [r for i, r in enumerate(grid) if i >= start and i != skip]


def split_side_by_side(grid: list[list[str]]) -> list[list[list[str]]]:
    """Tables placed side by side with a narrow gutter come out as one grid: a column after
    number columns whose cells are all labels (text, not numbers) starts another table."""
    if len(grid) < 3:
        return [grid]
    body = grid[1:]

    def labels(j):
        vals = [r[j] for r in body if r[j] and r[j] != LEFT]
        return (len(vals) >= 0.6 * len(body)
                and all(parse_number(v) is None and not tables.is_blank(v) for v in vals))

    def numeric(j):
        vals = [r[j] for r in body if r[j] and r[j] != LEFT and not tables.is_blank(r[j])]
        return vals and sum(parse_number(v) is not None for v in vals) >= 0.6 * len(vals)

    for j in range(2, len(grid[0]) - 1):
        if labels(j) and numeric(j - 1) and numeric(j + 1):
            return [[r[:j] for r in grid]] + split_side_by_side([r[j:] for r in grid])
    return [grid]


def spread_groups(group: list[str], header: list[str]) -> None:
    """A group row above the header ('FY24' over Q1..Total) names the columns nearest to it:
    each value column takes the nearest group label, as a merged cell (LEFT marker)."""
    at = [j for j, c in enumerate(group) if c and c != LEFT and j > 0]
    if len(at) < 2 or len(at) > len(group) // 2:
        return
    owner = {j: min(at, key=lambda a: (abs(a - j), a)) for j in range(1, len(group)) if header[j]}
    first = {}
    for j in sorted(owner):
        first.setdefault(owner[j], j)
    labels = {a: group[a] for a in at}
    for j in range(1, len(group)):
        group[j] = ""
    for j, a in owner.items():
        group[j] = labels[a] if first[a] == j else LEFT


POINT_GROUPS = re.compile(r"^-?\d{1,3}(\.\d{2,3})*\.\d{3}$")

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_RE = re.compile(r"([A-Za-z]{3})([-\u2019' ]?)(\d{2})")


def repair_months(header: list[str]) -> list[str]:
    """Month headers run in sequence ('Jan-25', 'Feb-25', ...): a cell the recognizer clipped or
    misread ('eb-25', 'lun-25', 'May-2') takes the month its position in the run calls for."""
    months = {}
    for j, h in enumerate(header):
        m = MONTH_RE.fullmatch(h.strip())
        if m and m.group(1).lower() in MONTHS:
            months[j] = (int(m.group(3)) * 12 + MONTHS.index(m.group(1).lower()), m)
    if len(months) < 3:
        return header
    # Month number at column 0 if the run steps one month per column: the most common guess wins.
    starts = [n - j for j, (n, _) in months.items()]
    start = max(set(starts), key=starts.count)
    if starts.count(start) < max(3, 0.6 * len(months)):
        return header
    style = next(m for n, m in months.values() if n - start in range(len(header)))
    sep = style.group(2)
    upper = style.group(1).isupper()
    lo, hi = min(months), max(months)
    out = list(header)
    for j in range(len(header)):
        h = header[j].strip()
        if j in months and months[j][0] - start == j:
            continue
        n = start + j
        name = MONTHS[n % 12].upper() if upper else MONTHS[n % 12].capitalize()
        want = f"{name}{sep}{n // 12:02d}"
        # Only a cell that looks like a damaged month, inside the run or right next to it.
        inside = lo <= j <= hi or (j in (lo - 1, hi + 1) and h)
        if (inside and h and len(h) <= len(want) + 2
                and difflib.SequenceMatcher(None, h.lower(), want.lower()).ratio() >= 0.6):
            out[j] = want
    return out


def _set(r: list[str], j: int, text: str) -> None:
    r[j] = r[j].keep(text) if isinstance(r[j], Cell) else text


# Letters the recognizer gives for digits in a small font, only trusted inside a number.
DIGIT_LOOKALIKES = str.maketrans({
    "O": "0", "o": "0", "D": "0", "B": "8", "S": "5", "l": "1", "I": "1",
})


def repair_number(s: str, commas: bool) -> str:
    """Undo the recognizer's separator slips in a number, if that makes it parse:
    '3.723.6' -> '3,723.6', '8.83,623' -> '8,83,623', '1,03.242' -> '1,03,242', '34,.192' -> '34,192',
    '1.0B6' -> '1,086', '4,035|' -> '4,035'."""
    if parse_number(s) is not None and not tables.is_malformed_number(s):
        return s
    t = s.strip().strip("|").strip()
    m = re.fullmatch(r"([(-]?)([\dOoDBSlI.,]*\d[\dOoDBSlI.,]*)(%?\)?)", t)
    if not m:
        return s
    sign, core, tail = m.groups()
    if sum(ch.isalpha() for ch in core) > 1:
        return s
    core = core.translate(DIGIT_LOOKALIKES)
    core = re.sub(r"[.,]{2,}", ",", core)
    if commas:
        core = re.sub(r"(?<=\d).(?=\d{2},)", ",", core)  # lakh group: '1.14,541'
    if core.count(".") > 1:  # only the last point can be a decimal point
        head, _, frac = core.rpartition(".")
        core = head.replace(".", ",") + "." + frac
    core = re.sub(r"(?<=,\d\d).(?=\d{3}$)", ",", core)    # '1,03.242'
    core = re.sub(r"(?<=,\d{3}).(?=\d{3}$)", ",", core)    # '1,234.567' is not an MIS number
    out = sign + core + tail
    return out if parse_number(out) is not None and not tables.is_malformed_number(out) else s


def normalize_numbers(grid: list[list[str]]) -> None:
    """Repair misread separators column by column, using how the rest of the table is written."""
    table_commas = any(re.search(r"\d,\d{2,3}", c) for r in grid for c in r)
    width = max((len(r) for r in grid), default=0)
    for r in grid:
        for j, c in enumerate(r):
            if c and c != LEFT:
                _set(r, j, repair_number(c, table_commas))
    for j in range(width):
        cells = [r for r in grid if j < len(r) and parse_number(r[j]) is not None]
        col = [r[j] for r in cells]
        # '7.982' in a column written '1,463', '2,15,408': a comma read as a point. A column with
        # no 1-2 digit decimals, in a table that groups thousands with commas, has no 3-decimal
        # numbers either (MIS tables never do).
        if table_commas and not any(re.search(r"\.\d{1,2}%?$", c) for c in col):
            for r in cells:
                if POINT_GROUPS.match(r[j]):
                    _set(r, j, r[j].replace(".", ","))
        # '606' in a column of '60.4', '58.1': the decimal point was lost.
        decs = [tables.decimals(c) for c in col]
        d = max(set(decs), key=decs.count, default=0)
        if d == 0 or decs.count(d) < max(3, 0.7 * len(col)):
            continue
        typical = sorted(abs(parse_number(c)) for c in col
                         if tables.decimals(c) == d)[decs.count(d) // 2]
        for r in cells:
            m = re.fullmatch(r"([(-]?)(\d+)(%?\)?)", r[j])
            if not m or len(m.group(2)) <= d or typical <= 0:
                continue
            sign, digits, tail = m.groups()
            fixed = f"{sign}{int(digits[:-d])}.{digits[-d:]}{tail}"
            old, new = abs(parse_number(r[j])), abs(parse_number(fixed))
            if (digits.startswith("0") or
                    new and (old == 0
                             or abs(np.log10(max(new, 1e-9) / typical)) < abs(np.log10(old / typical)))):
                _set(r, j, fixed)


def gridocr_tables(fpage: pymupdf.Page, rect: pymupdf.Rect, img_stem: str, model: str = "") -> list[dict]:
    """Drop-in for tables.vision_tables: same block split, cells read by OCR."""
    rect = picture.ink_rect(fpage.get_pixmap(clip=rect, dpi=72), rect)
    dpi = picture.source_dpi(fpage, rect, tables.TABLE_DPI)
    g = picture.Gray(fpage.get_pixmap(clip=rect, dpi=round(dpi), colorspace=pymupdf.csGRAY))
    sx, sy = rect.width / g.w, rect.height / g.h
    out = []
    for n, (x0, y0, x1, y1) in enumerate(picture.table_blocks(g), start=1):
        box = (max(0, x0 - 3), max(0, y0 - 3), min(g.w, x1 + 3), min(g.h, y1 + 3))
        brect = pymupdf.Rect(rect.x0 + box[0] * sx, rect.y0 + box[1] * sy,
                             rect.x0 + box[2] * sx, rect.y0 + box[3] * sy)
        grid, info = read_grid(fpage, brect, dpi)
        above, grid = find_header(grid)
        title_lines = above[-1:]  # the line right above the table is its title
        if ARGS.dump:
            dump = config.DATA_DIR / "pages" / f"gridocr_{img_stem}_b{n}.json"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text(json.dumps({"info": info, "grid": grid}, ensure_ascii=False, indent=0),
                            encoding="utf-8")
        for side in (b for s in split_side_by_side(grid) for b in tables.split_side_block(s)):
            for part in split_stacked(side):
                title, part = pop_title_row(part)
                header, body = structure(part)
                header = repair_months(header)
                if header:
                    t = make_table(header, body, title=title or " ".join(title_lines),
                                   bbox=[round(v, 1) for v in brect], source="gridocr")
                    out.append(add_values(t))
    # Tables in one picture usually share their row labels, so they are checked together.
    fix_label_misreads([r for t in out for r in t["rows"]])
    return out


def fix_label_misreads(rows: list[list[str]]) -> None:
    """Row labels OCR misread in a way the same column proves: 'B1' read as '81' when other labels
    there are 'B3-Agency', 'B2'...; '2536' / '25 36' when another row reads '25_36' (the
    underscore sits on the baseline and gets lost)."""
    labels = [r[0] for r in rows if r and r[0]]
    b_codes = any(re.match(r"^B\d", lab) for lab in labels)
    squashed = {re.sub(r"[_\s]", "", lab): lab for lab in labels if "_" in lab}
    # Range bands ('13_24', '25_36'): a band that lost its underscore everywhere ('012') is split
    # where it meets its neighbours — bands are contiguous, so '012' before '13_24' is '0_12'.
    bands = {tuple(map(int, lab.split("_"))) for lab in labels if re.fullmatch(r"\d+_\d+", lab)}
    starts, ends = {a for a, _ in bands}, {b for _, b in bands}
    for r in rows:
        if not r or not r[0]:
            continue
        lab = r[0]
        digits = re.sub(r"\s", "", lab)
        if b_codes and re.fullmatch(r"8\d(?:[-\s][A-Za-z].*)?", lab):
            r[0] = "B" + lab[1:]
        elif re.fullmatch(r"[\d\s]+", lab) and digits in squashed:
            r[0] = squashed[digits]
        elif len(bands) >= 2 and re.fullmatch(r"\d{2,5}", digits):
            fits = [k for k in range(1, len(digits))
                    if int(digits[:k]) < int(digits[k:])
                    and (int(digits[k:]) + 1 in starts or int(digits[:k]) - 1 in ends)]
            if len(fits) == 1:
                r[0] = f"{digits[:fits[0]]}_{digits[fits[0]:]}"


UNIT_RE = re.compile(
    r"(?<![a-z])(cr|crore|crores|lakh|lakhs|lac|mn|bn|million|billion)(?![a-z])|\u20b9|\$", re.I)
LOW_SCORE = 0.85


def add_values(t: dict) -> dict:
    """Typed numbers beside the text, for JSON: value (None if not a number), the unit the title or
    row label names, and the recognizer's score for each number it read with little confidence."""
    unit = UNIT_RE.search(t["title"] or "") or UNIT_RE.search(t["header"][0] if t["header"] else "")
    t["unit"] = unit.group(0) if unit else None
    t["values"] = [[parse_number(c) if c and not tables.is_blank(c) else None for c in r]
                   for r in t["rows"]]
    t["percent"] = [[bool(c) and c.rstrip(")").endswith("%") for c in r] for r in t["rows"]]
    doubtful = [{"row": i, "col": j, "text": str(c), "score": round(c.score, 3)}
                for i, r in enumerate(t["rows"]) for j, c in enumerate(r)
                if isinstance(c, Cell) and c.score < LOW_SCORE and parse_number(c) is not None]
    t["doubtful"] = doubtful
    if doubtful:
        t["issues"].append(f"{len(doubtful)} numbers read with low confidence")
    return t


def main() -> None:
    global ARGS
    if eval_tables is None:
        print("ERROR: eval_tables module not found. The standalone CLI requires eval_tables.py.")
        return
    if ocr_backend is None:
        print("ERROR: OCR backend not found. Install ocr_doctr (or ocr_rapidocr).")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="*")
    ap.add_argument("--real")
    ap.add_argument("--pages", type=int, nargs="*")
    ap.add_argument("--text-px", type=float, default=32)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--rec", choices=["small", "medium", "tiny"], default="small",
                    help="PP-OCRv6 recognizer size")
    ARGS = ap.parse_args()
    t0 = time.time()
    ocr_backend.recognizer() if hasattr(ocr_backend, "recognizer") else ocr_backend.engine()
    print(f"OCR loaded in {time.time() - t0:.1f}s", flush=True)
    tables.vision_tables = lambda a, k: []  # grid OCR alone: no vision fallback
    out = [f"# Table extraction eval \u2014 grid + cell OCR (rec {ARGS.rec}, text {ARGS.text_px:g}px)"]
    if ARGS.real:
        out += eval_tables.run_real(Path(ARGS.real), True, ARGS.pages)
    else:
        fx = eval_tables.FIXTURES
        pdfs = ([Path(p) for p in ARGS.pdf] if ARGS.pdf
                else [fx / "tables_image_150.pdf", fx / "tables_image_100.pdf"])
        for pdf in pdfs:
            print(pdf.name, flush=True)
            lines, _ = eval_tables.run_pdf(pdf, True, ARGS.pages)
            out += lines
    report = config.DATA_DIR / "table_eval_gridocr.md"
    report.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
