"""Find, clean and validate the tables on a PDF page.

Native text tables are read by several engines (pdfplumber and PyMuPDF, ruled and whitespace
strategies). Each result is scored by validate(); the best one per region is kept. Tables that
score low, and images that may hold a pasted table, are read with the vision model instead.

Each table is a dict:
  {title, bbox, source, header: [str], rows: [[str]], score, issues: [str], low_confidence}
"""
import math
import re
from dataclasses import dataclass

import pdfplumber
import pymupdf

from . import config
from . import llm

try:
    from . import picture
except ImportError:
    picture = None  # picture module not available; vision-table fallback disabled

# When picture is not installed, use Any for the type hints so the Block
# dataclass can still be defined (its fields are only used at runtime when
# picture IS available — the vision-table path is skipped otherwise).
import typing as _typing
_Gray = picture.Gray if picture else _typing.Any
_Layout = picture.Layout if picture else _typing.Any

# Many slides paste tables as a picture, which pdfplumber cannot read.
TABLE_PROMPT = """Extract every table in this image as a Markdown table.
Precede each table with a line
### Table: <table title as printed>
Keep row labels, column headers and numbers exactly as printed (including commas, signs,
brackets and % signs); leave empty cells empty.
The first row of a table is its column header row (often white text on a coloured band);
always include it.
Do not transcribe charts or graphs, only tables.
If there is no table, reply exactly: NO_TABLE

Output only the table titles and Markdown tables, nothing else."""

MIN_IMAGE_AREA = 0.15  # fraction of the page; smaller images are logos/icons
# Retry when the model loses the header row: ask for it first.
HEADER_PROMPT = """Transcribe the table in this image as a Markdown table.
Start with the column header row exactly as printed (often white text on a coloured band),
then every row, with row labels and numbers exactly as printed. Leave empty cells empty.
Output only the Markdown table."""

# Later pieces of a table cut into pieces: the header is given as text, read from the first piece.
CONT_PROMPT = """This image shows more rows of a table whose columns are, in order:
{columns}
Transcribe every row in the image as a Markdown table with exactly these {n} columns.
Start with this header line: {header_line}
Keep row labels and numbers exactly as printed; leave empty cells empty.
Output only the Markdown table."""

# A reply cut off at the token limit: ask for the rest of the same image.
AFTER_PROMPT = """This image shows a table whose columns are, in order:
{columns}
The rows up to and including the row "{last}" are already transcribed.
Transcribe only the rows that come after it, as a Markdown table with exactly these {n} columns.
Start with this header line: {header_line}
Keep row labels and numbers exactly as printed; leave empty cells empty.
Output only the Markdown table."""

TABLE_DPI = 300  # for vector regions; a pasted picture is rendered at its own resolution
MAX_IMAGE_PX = 1600  # longest side sent to the model

# Markers for merged cells while a table is still a grid.
LEFT, UP = "\x00<", "\x00^"

# ---------- numbers ----------

BLANKS = {"", "-", "–", "—", "na", "n/a", "n.a.", "nm", "n.m.", "nil", "*", "[obscured]"}
_UNITS = re.compile(r"(₹|rs\.?|inr|\$|€|£|%|bps|bp|pp|x|cr\.?|crore|lakh|lac|mn|bn|k|l)$|^(₹|rs\.?|inr|\$|€|£)",
                    re.I)
_CORE = re.compile(r"^(\d{1,3}(,\d{3})+|\d{1,2}(,\d{2})*,\d{3}|\d+)?(\.\d+)?$")
_NUMLIKE = re.compile(r"^[\d.,]*\d[\d.,]*$")
_PRIVATE_PARENS = re.compile(
    r"\ue081\s*([+−-]?(?:₹|rs\.?|inr|\$|€|£)?\s*\d[\d,.]*"
    r"(?:\s*(?:%|bps?|pp|x|cr|crore|lakh|lac|mn|bn|k|l))?)\s*\ue082",
    re.I,
)


def _normalize_pdf_glyphs(s: str) -> str:
    """Repair known private-use glyphs emitted by embedded Inter fonts in dashboard PDFs.

    E088 is the font's hyphen/minus glyph.  E081/E082 are changed to parentheses only when
    they enclose a number, because the same private-use range also contains decorative icons.
    """
    s = (s or "").replace("\ue088", "-")
    return _PRIVATE_PARENS.sub(r"(\1)", s)


def _strip_number(s: str) -> tuple[str, bool]:
    """Remove sign, brackets, currency and unit; returns (core, negative)."""
    s = _normalize_pdf_glyphs(s).strip().replace("−", "-").replace("–", "-")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s and s[0] in "+-":
        neg, s = s[0] == "-", s[1:].strip()
    for _ in range(3):  # "₹ 1,23,456 Cr", "45 bps", "12.5x"
        s2 = _UNITS.sub("", s).strip()
        if s2 == s:
            break
        s = s2
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s and s[0] in "+-":
        neg, s = s[0] == "-", s[1:].strip()
    return s, neg


def parse_number(s: str) -> float | None:
    """'1,06,786' -> 106786, '(1,959)' -> -1959, '4.25%' -> 4.25, '45 bps' -> 45; else None."""
    core, neg = _strip_number(s or "")
    if not core or not _CORE.match(core) or core == ".":
        return None
    v = float(core.replace(",", ""))
    return -v if neg else v


def is_blank(s: str) -> bool:
    return (s or "").strip().lower() in BLANKS


def is_malformed_number(s: str) -> bool:
    """Looks like a number but the digit grouping is wrong, e.g. '11,25' or '1,2,345'."""
    core, _ = _strip_number(s or "")
    return bool(_NUMLIKE.match(core)) and not _CORE.match(core)


def is_year(s: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d\d", s.strip()))


def decimals(s: str) -> int:
    core, _ = _strip_number(s)
    return len(core.split(".")[1]) if "." in core else 0


# ---------- grids from engines ----------

def _cluster(values: list[float], tol: float = 2.0) -> list[float]:
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _index(edges: list[float], v: float) -> int:
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


def grid_from_cells(cells: list, text_of) -> list[list[str]]:
    """Cell rectangles -> text grid with LEFT/UP markers where a merged cell covers a slot."""
    cells = [c for c in cells if c]
    xs = _cluster([c[0] for c in cells] + [c[2] for c in cells])
    ys = _cluster([c[1] for c in cells] + [c[3] for c in cells])
    if len(xs) < 2 or len(ys) < 2:
        return []
    grid = [[None] * (len(xs) - 1) for _ in range(len(ys) - 1)]
    for c in cells:
        j0, j1 = _index(xs, c[0]), _index(xs, c[2])
        i0, i1 = _index(ys, c[1]), _index(ys, c[3])
        if j1 <= j0 or i1 <= i0:
            continue
        grid[i0][j0] = text_of(c)
        for i in range(i0, i1):
            for j in range(j0, j1):
                if (i, j) != (i0, j0):
                    grid[i][j] = UP if j == j0 else LEFT
    return [[c if c is not None else "" for c in row] for row in grid]


def _clean_text(s: str | None) -> str:
    # Some PDFs encode the minus sign as a soft hyphen.
    s = _normalize_pdf_glyphs(s or "").replace("­", "-").replace("|", "/")
    return re.sub(r"\s+", " ", s).strip()


# A word is (x0, top, x1, bottom, text).
def page_words(page: pdfplumber.page.Page) -> list[tuple]:
    return [(w["x0"], w["top"], w["x1"], w["bottom"], w["text"])
            for w in page.extract_words(x_tolerance=1.5, keep_blank_chars=False)]


def _text_in(words: list[tuple], bbox) -> str:
    x0, y0, x1, y1 = bbox
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    return _clean_text(" ".join(w[4] for line in _lines(inside) for w in line))


def _lines(words: list[tuple], tol: float = 2.5) -> list[list[tuple]]:
    """Words grouped into text lines, top to bottom, each left to right."""
    lines = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if lines and abs(w[1] - lines[-1][0][1]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _phrases(line: list[tuple]) -> list[tuple]:
    """Join words closer than about a space into phrases: 'Interest income', '45 bps'."""
    def value(t):
        return parse_number(t) is not None or is_blank(t)

    out, last = [], ""
    for w in line:
        gap = max(3.5, 0.45 * (w[3] - w[1]))
        # Two values side by side are two cells, however close ("571 n.m.").
        if out and w[0] - out[-1][2] <= gap and not (value(last) and value(w[4])):
            p = out[-1]
            out[-1] = (p[0], min(p[1], w[1]), w[2], max(p[3], w[3]), f"{p[4]} {w[4]}")
        else:
            out.append(w)
        last = w[4]
    return out


def _is_tabular(phrases: list[tuple]) -> bool:
    nums = sum(1 for p in phrases if parse_number(p[4]) is not None)
    blanks = sum(1 for p in phrases if is_blank(p[4]))
    return nums >= 1 and nums + blanks >= 2


def _is_columnar(phrases: list[tuple]) -> bool:
    return len(phrases) >= 2 and max(b[0] - a[2] for a, b in zip(phrases, phrases[1:])) >= 10


def _plumber(page: pdfplumber.page.Page, words: list[tuple]) -> list[tuple[tuple, list]]:
    return [(tuple(t.bbox), grid_from_cells(t.cells, lambda c: _text_in(words, c))) for t in page.find_tables()]


def _fitz(fpage: pymupdf.Page, words: list[tuple]) -> list[tuple[tuple, list]]:
    return [(tuple(t.bbox), grid_from_cells(t.cells, lambda c: _text_in(words, c)))
            for t in fpage.find_tables(strategy="lines").tables]


def _plumber_strict(page: pdfplumber.page.Page, words: list[tuple]) -> list[tuple[tuple, list]]:
    """Tables whose cell borders are explicit lines, without inferred line extensions."""
    settings = {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"}
    return [(tuple(t.bbox), grid_from_cells(t.cells, lambda c: _text_in(words, c)))
            for t in page.find_tables(settings)]


def _fitz_strict(fpage: pymupdf.Page, words: list[tuple]) -> list[tuple[tuple, list]]:
    return [(tuple(t.bbox), grid_from_cells(t.cells, lambda c: _text_in(words, c)))
            for t in fpage.find_tables(strategy="lines_strict").tables]


def _join_wrapped(parts: list[str]) -> str:
    """The lines of one wrapped header cell as one text: 'Share(Sep-' + '25)' is 'Share(Sep-25)'."""
    s = ""
    for t in parts:
        if s and (s.endswith("-") or (s.count("(") > s.count(")") and re.fullmatch(r"[\w’'.\-]{0,3}\)", t))):
            s += t
        else:
            s = f"{s} {t}".strip()
    return s


def _header_band_rows(head: list[list[tuple]], cols: list[list[float]], edges: list[float],
                      n_above: int = 0) -> list[list[str]]:
    """Header lines (phrases each) as grid rows. Lines less than a line height apart are one band:
    the wrapped parts of vertically centred header cells, which sit on staggered lines. Each band
    is one row, its cells joined per column top to bottom. A band holding one phrase across
    several value columns is a group label over all of them (a unit band, 'Portfolio (Cr)').
    The first n_above lines sit above the first column-like line: there the label column's
    wrapped name ('Active / Lender / Associations') is joined into one cell."""
    if not head:
        return []

    def top(ph):
        return min(p[1] for p in ph)

    def height(ph):
        return max(p[3] for p in ph) - top(ph)

    bands, anchor_band = [[head[0]]], 0
    for n, ph in enumerate(head[1:], start=1):
        prev = bands[-1][-1]
        if top(ph) - top(prev) < 0.8 * min(height(ph), height(prev)):
            bands[-1].append(ph)
        else:
            bands.append([ph])
        if n == n_above:
            anchor_band = len(bands) - 1

    label_parts, rows = [], []
    for b, band in enumerate(bands):
        phrases = sorted((p for ph in band for p in ph), key=lambda p: (p[1], p[0]))
        per_col: list[list[str]] = [[] for _ in cols]
        for p in phrases:
            per_col[sum(1 for e in edges if (p[0] + p[2]) / 2 > e)].append(p[4])
        values = [p for p in phrases if (p[0] + p[2]) / 2 > edges[0]]
        row = [""] * len(cols)
        if b <= anchor_band:
            label_parts += per_col[0]
        else:
            row[0] = _join_wrapped(per_col[0])
        if len(values) == 1 and len(cols) > 2 and sum(1 for a, c in cols[1:] if values[0][0] < c and a < values[0][2]) >= 2:
            row[1:] = [values[0][4]] + [LEFT] * (len(cols) - 2)
        else:
            row[1:] = [_join_wrapped(parts) for parts in per_col[1:]]
        rows.append(row)
    rows[anchor_band][0] = _join_wrapped(label_parts)
    return [[_clean_text(c) if c != LEFT else c for c in r] for r in rows if any(c and c != LEFT for c in r)]


def whitespace_tables(words: list[tuple], page_width: float) -> list[tuple[tuple, list]]:
    """Tables without vertical rules: runs of lines holding several numbers, split into columns
    at the vertical gaps shared by those lines, plus the header lines just above them."""
    lines = [(_phrases(line), line) for line in _lines(words)]
    if not lines:
        return []
    height = sorted(l[0][3] - l[0][1] for l, _ in lines)[len(lines) // 2]
    max_gap = 2.6 * height + 6

    def top(k):
        return min(p[1] for p in lines[k][0])

    def bottom(k):
        return max(p[3] for p in lines[k][0])

    def is_paragraph(k):
        ph = lines[k][0]
        return len(ph) == 1 and ph[0][2] - ph[0][0] > 0.45 * page_width

    out, k = [], 0
    while k < len(lines):
        if not _is_tabular(lines[k][0]):
            k += 1
            continue
        body, last_tab, m = [k], k, k + 1
        while m < len(lines) and top(m) - bottom(body[-1]) <= max_gap and not is_paragraph(m):
            body.append(m)
            if _is_tabular(lines[m][0]):
                last_tab = m
            m += 1
        body = [i for i in body if i <= last_tab]
        k = last_tab + 1
        tab_rows = [i for i in body if _is_tabular(lines[i][0])]
        if len(tab_rows) < 2:
            continue
        # Header lines above, up to a gap, a paragraph or another table. The topmost column-like
        # line anchors the header; a few closely spaced lines above it belong to it too (a unit
        # band over the columns, the wrapped start of the label column's name).
        bx0 = min(p[0] for i in body for p in lines[i][0]) - 12
        bx1 = max(p[2] for i in body for p in lines[i][0]) + 12
        slack = 0.03 * page_width

        def inside(i):
            # By centre: a header may stick out past the body ("(Sep'24 to Sep'25)" over a value).
            return all(bx0 - slack <= (p[0] + p[2]) / 2 <= bx1 + slack for p in lines[i][0])

        cand, h = [], body[0] - 1
        while h >= 0 and len(cand) < 12 and top(cand[0] if cand else body[0]) - bottom(h) <= max_gap:
            if not inside(h) or is_paragraph(h) or _is_tabular(lines[h][0]):
                break
            cand.insert(0, h)
            h -= 1
        anchor = next((n for n, i in enumerate(cand) if _is_columnar(lines[i][0])), None)
        head = [] if anchor is None else cand[anchor:]
        if anchor is not None:
            # Same type size as the header: a title above it is set larger.
            size = bottom(head[0]) - top(head[0])
            for i in reversed(cand[:anchor]):
                h_i = bottom(i) - top(i)
                if len(head) - len(cand[anchor:]) >= 4 or h_i > 1.1 * size or top(head[0]) - top(i) > 1.9 * h_i:
                    break
                head.insert(0, i)
        n_above = len(head) - (len(cand) - anchor) if anchor is not None else 0

        # Columns: merge the x-extents of phrases on number rows; the gaps between are borders.
        spans = sorted((p[0], p[2]) for i in tab_rows for p in lines[i][0] if not is_blank(p[4]))
        cols = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= cols[-1][1] + 1:
                cols[-1][1] = max(cols[-1][1], b)
            else:
                cols.append([a, b])
        if len(cols) < 2:
            continue
        edges = [(a[1] + b[0]) / 2 for a, b in zip(cols, cols[1:])]

        def dist(p, j):
            return max(0.0, cols[j][0] - p[2], p[0] - cols[j][1])

        grid = _header_band_rows([lines[i][0] for i in head], cols, edges, n_above)
        for i in body:
            cells = [""] * len(cols)
            # Numbers first: they define the columns. A value landing on a filled cell (a
            # left-aligned "n.m." next to a right-aligned number) goes to the nearest free one.
            for p in sorted(lines[i][0], key=lambda p: parse_number(p[4]) is None):
                c = (p[0] + p[2]) / 2
                j = sum(1 for e in edges if c > e)
                if cells[j] and (parse_number(p[4]) is not None or is_blank(p[4])):
                    free_cols = [k for k in range(len(cols)) if not cells[k]]
                    if free_cols:
                        j = min(free_cols, key=lambda k: dist(p, k))
                cells[j] = f"{cells[j]} {p[4]}".strip()
            grid.append([_clean_text(c) for c in cells])
        all_ph = [p for i in head + body for p in lines[i][0]]
        bbox = (min(p[0] for p in all_ph), min(p[1] for p in all_ph),
                max(p[2] for p in all_ph), max(p[3] for p in all_ph))
        out.append((bbox, grid))
    return out


# ---------- grid -> header + rows ----------

def _numeric_share(cells: list[str]) -> float:
    vals = [c for c in cells if c and c not in (LEFT, UP) and not is_blank(c)]
    if not vals:
        return 0.0
    return sum(1 for c in vals if parse_number(c) is not None and not is_year(c)) / len(vals)


def _is_header_row(row: list[str]) -> bool:
    rest = [c for c in row[1:] if c and c not in (LEFT, UP)]
    return bool(rest) and _numeric_share(rest) < 0.5


def pop_title_row(grid: list[list[str]]) -> tuple[str, list[list[str]]]:
    """A first row holding one text across the value columns (a title bar) is the table's title,
    not part of every column name."""
    if len(grid) < 3:
        return "", grid
    row = grid[0]
    texts = {c for c in row if c and c not in (LEFT, UP)}
    spanned = sum(1 for c in row if c == LEFT or c in texts)
    if len(texts) == 1 and len(row) >= 3 and spanned >= 0.6 * len(row) and _is_header_row(grid[1]):
        return texts.pop(), grid[1:]
    return "", grid


def _header_parts(parts: list[str], label_col: bool = False) -> list[str]:
    """The readings of one column's header rows, minus OCR noise. White text on a coloured band
    often comes back as the header plus fragments of the band's edges read as extra header lines
    ('o / FY_25 / 1「-20', 'D-+i-l- / Particulars / 1 aticutars')."""
    if len(parts) < 2:
        return parts
    import difflib

    def noise(p: str) -> bool:
        s = p.replace(" ", "")
        alnum = sum(ch.isalnum() and ch.isascii() for ch in s)
        return len(s) <= 1 or bool(re.search(r"[^\w%&()/.,'’:+\-#$₹*]", s)) or alnum < 0.6 * len(s)

    kept = [p for p in parts if not noise(p)] or parts
    if any(MONTH_RE.match(p.strip()) for p in kept):
        # Beside a period name, only real words (or acronyms) are kept ('FY24 / Total', 'ABC /
        # Sep-24'); letter-digit jumbles are the band misread ('01Fv00 / Q1_FY26', 'FY_28 / 1-20').
        word = r"(?:[A-Za-z][a-z]+|[A-Z]{2,6})"
        kept = [p for p in kept if MONTH_RE.match(p.strip())
                or re.fullmatch(rf"{word}(?:[ &/+-](?:[A-Za-z][a-z]*|[A-Z]{{2,6}}))*\.?%?", p.strip())]

    if not label_col:
        return kept

    def letters(p: str) -> int:
        return sum(ch.isalpha() for ch in p)

    # A misreading of a fuller reading of the label column's name ('1 aticutars' beside
    # 'Particulars'). Not applied to value columns, where 'Cost / Count' are two real levels.
    return [p for p in kept if not any(q != p and (letters(q), len(q)) > (letters(p), len(p))
                                       and difflib.SequenceMatcher(None, p.lower(), q.lower()).ratio() >= 0.5
                                       for q in kept)]


def split_side_block(grid: list[list[str]]) -> list[list[list[str]]]:
    """A second table beside the first that shares its row labels but starts lower down, under
    its own header row of period names over the right-hand columns only (e.g. a ratios block
    next to a P&L, titled in the cell above that header). Returned as its own table, with the
    shared row labels, so its columns are not read as more periods of the first table."""
    if len(grid) < 4:
        return [grid]
    width = max(len(r) for r in grid)
    g = [list(r) + [""] * (width - len(r)) for r in grid]

    def filled(c: str) -> bool:
        return bool(c) and c not in (LEFT, UP)

    for i in range(2, len(g) - 2):
        row = g[i]
        cols = [j for j, c in enumerate(row) if filled(c)]
        if not cols or filled(row[0]) or cols[0] < 2:
            continue
        j0 = cols[0]
        heads = [row[j] for j in cols]
        if len(heads) < 2 or sum(1 for h in heads if MONTH_RE.match(h.strip())) < 0.8 * len(heads):
            continue
        data_above = [r for r in g[1:i - 1] if _numeric_share(r[1:j0]) >= 0.5]
        # Words of a comment box beside the table can land in those columns (even a stray "6%"),
        # but not a run of numbers, which would mean the block really does extend up there.
        stray = sum(1 for r in data_above for c in r[j0:] if filled(c) and parse_number(c) is not None)
        if len(data_above) < 2 or stray > max(1, len(data_above) // 4):
            continue
        below = g[i + 1:]
        if sum(1 for r in below if _numeric_share(r[j0:]) >= 0.5) < max(2, 0.5 * len(below)):
            continue
        title = " ".join(c for c in g[i - 1][j0:] if filled(c))
        left = [r[:j0] for k, r in enumerate(g) if k != i]
        first_data = next(k for k, r in enumerate(left) if _numeric_share(r[1:]) >= 0.5)
        head_rows = left[:first_data]
        named = sum(1 for j in range(1, j0)
                    if any(MONTH_RE.match(part.strip()) for r in head_rows for part in r[j].split(" / ") if filled(r[j])))
        if len(heads) == j0 - 1 and named < (j0 - 1) / 2:
            # The main block's header band was misread ('01Fv00' for Q1_FY26), and the side block
            # sits over the same periods, column for column: its header names them.
            label = next((r[0] for r in head_rows if filled(r[0])), "")
            left = [[label, *heads]] + left[first_data:]
        right = ([[title] + [LEFT] * (width - j0)] if title else []) + [[""] + row[j0:]] + \
            [[r[0]] + r[j0:] for r in below if any(filled(c) for c in r[j0:])]
        return [left] + split_side_block(right)
    return [grid]


def structure(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Split a grid into one flat header and body rows; fill merged cells."""
    grid = [r for r in grid if any(c and c not in (LEFT, UP) for c in r)]
    if not grid:
        return [], []
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [j for j in range(width) if any(r[j] and r[j] not in (LEFT, UP) for r in grid)]
    grid = [[r[j] for j in keep] for r in grid]

    nh = 0
    while nh < min(3, len(grid) - 1) and _is_header_row(grid[nh]):
        nh += 1
    head = [row[:] for row in grid[:nh]]
    for i, row in enumerate(head):
        for j, c in enumerate(row):
            if c == LEFT:
                row[j] = row[j - 1] if j else ""
            elif c == UP:
                row[j] = head[i - 1][j] if i else ""
    # A spanning header in a borderless table is text over the first of its columns with blanks
    # to its right; fill it right while the row below has names there.
    for i in range(nh - 1):
        for j in range(1, len(head[i])):
            if not head[i][j] and head[i + 1][j] and head[i][j - 1] and head[i + 1][j - 1]:
                head[i][j] = head[i][j - 1]
    header = []
    for j in range(len(grid[0])):
        parts = []
        for row in head:
            if row[j] and (not parts or parts[-1] != row[j]):
                parts.append(row[j])
        header.append(" / ".join(_header_parts(parts, label_col=j == 0)))

    rows = []
    for row in grid[nh:]:
        r = []
        for j, c in enumerate(row):
            if c == UP:
                r.append(rows[-1][j] if rows else "")
            elif c == LEFT:
                r.append("")
            else:
                r.append(c)
        rows.append(r)

    # A "0, 1, 2, ..." column with no header is a row index the vision model added.
    if rows and not header[0] and [r[0] for r in rows] == [str(i) for i in range(len(rows))]:
        header, rows = header[1:], [r[1:] for r in rows]
    # Two label columns (Region, Branch): a blank group label continues the one above.
    if len(header) > 2 and rows and _numeric_share([r[1] for r in rows]) < 0.5             and _numeric_share([r[0] for r in rows]) < 0.5:
        for i in range(1, len(rows)):
            if not rows[i][0] and rows[i][1] and rows[i - 1][0]:
                rows[i][0] = rows[i - 1][0]
    return header, rows


MONTH_RE = re.compile(  # period names used as column headers: Jan-25, March 2024, Q1 FY25, H1, FY26
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-'_ ]?\d{2,4}$"
    r"|^(q[1-4]|h[12])(?:\b|_)|^fy\s?[_'’]?\d{2,4}\b", re.I)


def split_stacked(grid: list[list[str]]) -> list[list[list[str]]]:
    """Tables stacked in one picture and returned as one: split before each repeated header row
    (a label followed by month or period names) that comes after data rows."""
    parts, cur = [], []
    for row in grid:
        rest = [c for c in row[1:] if c]
        is_head = len(rest) >= 2 and sum(1 for c in rest if MONTH_RE.match(c)) >= 0.6 * len(rest)
        has_data = any(_numeric_share(r[1:]) >= 0.5 for r in cur)
        # Power BI exports can draw two vertically stacked tables with touching borders. The lower
        # header can be contaminated by the clipped edge of the upper table, while its paired
        # label heading remains a reliable boundary.
        first = " ".join(c for c in row[:3] if c and c not in (LEFT, UP))
        paired_labels = bool(re.search(r"\b(?:product|category)\b.*\bsub\s*(?:product|category)\b",
                                       first, re.I))
        if paired_labels and has_data:
            title = []
            if cur:
                label = next((c for c in cur[-1] if c and c not in (LEFT, UP)
                              and parse_number(c) is None), "")
                if label and len(label) < 80:
                    cur.pop()
                    title = [[label] + [LEFT] * (len(row) - 1)]
            if cur:
                parts.append(cur)
            cur = title
            has_data = False
            is_head = False
        if is_head and has_data:
            parts.append(cur)
            cur = []
        cur.append(row)
    parts.append(cur)
    return [p for p in parts if p]


def split_structured_side(header: list[str], rows: list[list[str]]) -> list[tuple[list[str], list[list[str]], int, int]]:
    """Split independent tables placed side by side in one ruled region."""
    width = len(header)
    if width < 6 or not rows:
        return [(header, rows, 0, width)]

    def compact(part_rows):
        out = []
        for row in part_rows:
            if not any(row):
                continue
            if out and row == out[-1]:
                continue
            out.append(row)
        return out

    for j in range(2, width - 2):
        labels = [r[j] for r in rows]
        # A period column is a measure axis even if a damaged value makes the column look textual.
        if any(MONTH_RE.match(part.strip()) for part in header[j].split("/")):
            continue
        left_numeric = sum(_numeric_share([r[k] for r in rows]) >= 0.5 for k in range(1, j))
        right_numeric = sum(_numeric_share([r[k] for r in rows]) >= 0.5 for k in range(j + 1, width))
        if (_numeric_share(labels) < 0.2 and sum(bool(c) for c in labels) >= 2
                and left_numeric >= 2 and right_numeric >= 2):
            left = (header[:j], compact([r[:j] for r in rows]), 0, j)
            right_parts = split_structured_side(header[j:], compact([r[j:] for r in rows]))
            return [left] + [(h, rs, a + j, b + j) for h, rs, a, b in right_parts]
    return [(header, compact(rows), 0, width)]


def merge_split_header_columns(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Rejoin a two-word measure header accidentally emitted as adjacent columns."""
    j = 0
    while j + 1 < len(header):
        left = [p.strip() for p in header[j].rsplit("/", 1)]
        right = [p.strip() for p in header[j + 1].rsplit("/", 1)]
        if (len(left) == len(right) == 2 and left[0] == right[0]
                and left[1].lower() == "current" and right[1].lower() == "balance"):
            pairs = [(r[j], r[j + 1]) for r in rows]
            compatible = all(not a or not b or _clean_text(a) == _clean_text(b) for a, b in pairs)
            if compatible:
                header[j] = f"{left[0]} / Current Balance"
                header.pop(j + 1)
                for r in rows:
                    r[j] = r[j] or r[j + 1]
                    r.pop(j + 1)
                continue
        j += 1
    return header, rows


# ---------- validation ----------

TOTAL_RE = re.compile(r"\b(sub[\s-]?total|total|grand total)\b", re.I)
SKIP_SUM_RE = re.compile(r"%|bps|ratio|margin|yield|rate|change|growth|avg|average|efficiency|x\b", re.I)


def _label_cols(rows: list[list[str]]) -> int:
    n = 0
    for j in range(len(rows[0])):
        if _numeric_share([r[j] for r in rows]) >= 0.5:
            break
        n += 1
    return n


def _row_label(row: list[str], nl: int) -> str:
    return " ".join(c for c in row[:max(nl, 1)] if c)


def _close(a: float, b: float, terms: int, dec: int) -> bool:
    tol = max(abs(b) * 0.001, terms * 0.5 * 10 ** -dec, 10 ** -dec)
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol + 1e-12)


def check_totals(header: list[str], rows: list[list[str]], nl: int, num_cols: list[int]) -> tuple[int, list[str]]:
    """Checks total rows (and total columns) against the items they sum. Returns (checked, issues)."""
    checked, issues = 0, []
    cols = [j for j in num_cols if not SKIP_SUM_RE.search(header[j] if j < len(header) else "")
            and not any("%" in r[j] or "x" in r[j].lower() for r in rows)]

    def val(r, j):
        return parse_number(r[j]) if r[j] and not is_blank(r[j]) else None

    def measure_label(value):
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def sums_to(t, terms, j):
        parts = [p for p in (val(x, j) for x in terms) if p is not None]
        dec = max([decimals(t[j])] + [decimals(x[j]) for x in terms])
        return _close(sum(parts), val(t, j), len(parts), dec)

    def rolled_up(terms):
        """Rows summed by a later row among the items ('PreX-UNA', 'PreX-Allo' under 'PreX'):
        a total counts the roll-up row, not its parts again."""
        children = set()
        for i, r in enumerate(terms):
            for k in range(2, min(8, i) + 1):
                kids = [x for x in terms[i - k:i] if id(x) not in children]
                if len(kids) < 2:
                    continue
                js = [j for j in cols if val(r, j) is not None and any(val(x, j) is not None for x in kids)]
                if len(js) >= 2 and sum(sums_to(r, kids, j) for j in js) >= 0.6 * len(js):
                    children |= {id(x) for x in kids}
                    break
        return [x for x in terms if id(x) not in children]

    # Rows: a subtotal sums the items since the previous total; a (grand) total sums all items.
    has_sub = any(re.search(r"sub[\s-]?total", _row_label(x, nl), re.I) for x in rows)
    items, segment = [], []
    for r in rows:
        label = _row_label(r, nl)
        if TOTAL_RE.search(label):
            is_sub = re.search(r"sub[\s-]?total", label, re.I)
            terms = items if (has_sub and not is_sub) else segment
            # A subtotal printed before the rows of a new top-level category describes that new
            # category; it must not be checked against the preceding category's rows.
            if is_sub and nl >= 2 and segment and r[0] and segment[-1][0] and r[0] != segment[-1][0]:
                segment = []
                terms = []
            # Some grids repeat each category for different measures, for example Amount and
            # No of Accounts, followed by one grand-total row for each measure.  Compare a total
            # only with rows carrying the same secondary label instead of adding unlike units.
            selectors = [(j, measure_label(r[j])) for j in range(1, nl)
                         if r[j] and not TOTAL_RE.search(r[j])]
            if selectors:
                matching = [x for x in terms if all(j < len(x) and measure_label(x[j]) == value
                                                    for j, value in selectors)]
                if len(matching) >= 2:
                    terms = matching
            tops = None
            for j in cols:
                t = val(r, j)
                parts = [val(x, j) for x in terms]
                parts = [p for p in parts if p is not None]
                if t is None or len(parts) < 2:
                    continue
                checked += 1
                if sums_to(r, terms, j):
                    continue
                if tops is None:
                    tops = rolled_up(terms)
                if len(tops) < len(terms) and sums_to(r, tops, j):
                    continue
                dec = max([decimals(r[j])] + [decimals(x[j]) for x in terms])
                issues.append(f"total mismatch: '{label}' / '{header[j]}' is {r[j]}, items sum to "
                              f"{sum(parts):,.{dec}f}")
            segment = []
        elif any(val(r, j) is not None for j in cols):
            items.append(r)
            segment.append(r)

    # Columns: a "Total" column sums the columns since the previous total column.
    def terminal_measure(value: str) -> str:
        """'Grand Total / POS' and 'NPA / POS' both aggregate the POS measure."""
        return measure_label(value.rsplit("/", 1)[-1])

    seg = []
    for j in cols:
        if TOTAL_RE.search(header[j]) or header[j].strip().lower() in ("fytd", "ytd"):
            for r in rows:
                same_measure = [k for k in seg if terminal_measure(header[k]) == terminal_measure(header[j])]
                source_cols = same_measure if len(same_measure) >= 2 else seg
                t, parts = val(r, j), [val(r, k) for k in source_cols]
                if t is None or len(source_cols) < 2 or any(p is None for p in parts):
                    continue
                checked += 1
                dec = max([decimals(r[j])] + [decimals(r[k]) for k in source_cols])
                if not _close(sum(parts), t, len(parts), dec):
                    issues.append(f"total mismatch: '{_row_label(r, nl)}' / '{header[j]}' is {r[j]}, "
                                  f"columns sum to {sum(parts):,.{dec}f}")
            # Hierarchical headers can have adjacent totals for POS, Recovery, and Rate.  Keep
            # their source columns available until all terminal measures have been checked.
            if len(same_measure) < 2:
                seg = []
        else:
            seg.append(j)
    return checked, issues


def validate(header: list[str], rows: list[list[str]], allow_sparse: bool = False) -> tuple[float, list[str]]:
    """Score 0..1 of how trustworthy an extracted table looks, plus the problems found."""
    issues = []
    if not rows or len(rows) < 2 or len(header) < 2:
        return 0.0, ["too small to be a table"]
    width = len(header)
    long_cells = sum(1 for r in rows for c in r if len(c) > 60)
    if long_cells:
        issues.append(f"{long_cells} cells hold long text (paragraph captured?)")

    nl = _label_cols(rows)
    num_cols = [j for j in range(nl, width) if _numeric_share([r[j] for r in rows]) >= 0.6]
    data_cols = max(1, width - nl)
    if nl == 0:
        issues.append("no label column")
    if nl > 2:
        issues.append(f"{nl} text columns before the numbers")

    header_ok = sum(1 for h in header[nl:] if h and not _numeric_share([h])) / data_cols
    if header_ok < 1:
        issues.append("header missing or numeric")
    if nl == 0 and header_ok == 0:
        return 0.0, issues + ["numbers without header or labels (chart labels?)"]
    no_header = header_ok == 0

    cells = [r[j] for r in rows for j in num_cols]
    filled = [c for c in cells if c and not is_blank(c)]
    parsed = [c for c in filled if parse_number(c) is not None]
    coverage = len(parsed) / len(filled) if filled else 0.0
    malformed = [c for c in filled if is_malformed_number(c)]
    if malformed:
        issues.append(f"malformed numbers: {', '.join(malformed[:5])}")
    unparsed = [c for c in filled if parse_number(c) is None and c not in malformed]
    if unparsed:
        issues.append(f"text inside number columns: {', '.join(unparsed[:5])}")

    # Data rows (not section headings) should fill most of the number columns.
    data_rows = [r for r in rows if any(r[j] for j in num_cols)]
    ragged = sum(1 for r in data_rows if sum(1 for j in num_cols if r[j]) < 0.5 * len(num_cols))
    consistency = 1 - ragged / len(data_rows) if data_rows else 0.0
    if ragged:
        issues.append(f"{ragged} rows fill under half of the number columns")

    if num_cols and consistency < 0.5 and not allow_sparse:
        return 0.0, issues + ["most rows are sparse (chart labels?)"]
    if not num_cols and long_cells:
        return 0.0, issues + ["long text and no numbers (text block, not a table)"]

    empty = sum(1 for r in rows for c in r[nl:] if not c) / (len(rows) * data_cols)
    checked, total_issues = check_totals(header, rows, nl, num_cols)
    issues += total_issues

    if not num_cols:
        # A text table: nothing to check the numbers against, trust it moderately.
        return round(0.4 * header_ok + 0.3 * (1 - empty) + 0.3 * (nl > 0) - 0.2 * bool(long_cells), 3), \
            issues + ["no number columns"]
    score = (0.2 * header_ok + 0.25 * coverage + 0.2 * len(num_cols) / data_cols + 0.15 * consistency
             + 0.1 * (1 - empty) + 0.1 * (0 < nl <= 2))
    score -= (0.1 * min(len(malformed), 3) + 0.1 * min(len(unparsed), 3)
              + 0.2 * bool(long_cells) + (0.05 if total_issues else 0))
    score -= 0.25 * no_header  # values without column names cannot be used
    return round(max(0.0, min(1.0, score)), 3), issues


# ---------- markdown ----------

def table_to_markdown(table: list[list[str | None]]) -> str:
    rows = [[(c or "").replace("\n", " ").replace("|", "/").strip() for c in r] for r in table if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # Drop empty columns; chart legends/gridlines often parse as sparse "tables".
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    width = len(keep)
    if len(rows) < 2 or width < 2:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def to_markdown(t: dict) -> str:
    md = table_to_markdown([t["header"]] + t["rows"])
    if md and t.get("title"):
        md = f"**{t['title']}**\n\n{md}"
    return md


def markdown_tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Vision output -> [(title, grid)]."""
    out, title, grid = [], "", []

    def flush():
        nonlocal grid
        if grid:
            out.append((title, grid))
        grid = []

    for line in text.splitlines():
        # granite-vision sometimes wraps a table in <md>...</md>, header on the <md> line.
        s = re.sub(r"</?md>", "", line).strip()
        if s.startswith("|"):
            cells = [_clean_text(c).strip("*").strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            grid.append(cells)
        else:
            flush()
            m = re.match(r"#+\s*(?:table:\s*)?(.*)", s, re.I)
            if m and m.group(1) and "<" not in m.group(1):  # "<table title as printed>" echoed
                title = m.group(1).strip()
    flush()
    return out


# ---------- finding tables on a page ----------

def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap(a, b) -> float:
    """Intersection as a share of the smaller box."""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    small = min(_area(a), _area(b))
    return _area(inter) / small if small else 0.0


def _title_above(page: pdfplumber.page.Page, bbox) -> str:
    top = max(0, bbox[1] - 20)
    if top >= bbox[1] - 2:
        return ""
    try:
        text = page.crop((max(0, bbox[0] - 2), top, min(page.width, bbox[2]), bbox[1] - 1), strict=False).extract_text()
    except ValueError:
        return ""
    lines = [s.strip() for s in (text or "").splitlines() if s.strip()]
    if not lines or len(lines[-1]) >= 120:
        return ""
    title = lines[-1]
    # A closely preceding total row is evidence, not the title of the next stacked table.
    return "" if sum(parse_number(tok) is not None for tok in title.split()) >= 2 else title


def _major_heading_above(page: pdfplumber.page.Page, bbox) -> str:
    """Nearest larger section heading above a table, such as a business unit name."""
    try:
        words = page.extract_words(extra_attrs=["size"], keep_blank_chars=False)
    except Exception:
        return ""
    groups = {}
    for word in words:
        if word["bottom"] >= bbox[1] or float(word.get("size", 0)) < 9.5:
            continue
        groups.setdefault(round(word["top"]), []).append(word)
    for _, line in sorted(groups.items(), reverse=True):
        line.sort(key=lambda word: word["x0"])
        text = " ".join(word["text"] for word in line).strip()
        if text and len(text) < 80 and not any(parse_number(tok) is not None for tok in text.split()):
            return text
    return ""


def make_table(header, rows, **meta) -> dict:
    score, issues = validate(header, rows, allow_sparse=meta.get("allow_sparse", False))
    return {"title": meta.get("title", ""), "bbox": meta.get("bbox"), "source": meta["source"],
            "header": header, "rows": rows, "score": score, "issues": issues,
            "low_confidence": score < config.TABLE_MIN_SCORE}


def native_candidates(page: pdfplumber.page.Page, fpage: pymupdf.Page) -> list[dict]:
    """Ruled tables from both engines, then whitespace tables from the text outside them."""
    words = page_words(page)

    def build(found, name, ruled):
        out = []
        for bbox, grid in found:
            # Tables drawn touching come out as one grid: split at each repeated header row,
            # giving each part its share of the box so both survive the overlap check.
            start = 0
            for part in split_stacked(grid):
                y0 = bbox[1] + (bbox[3] - bbox[1]) * start / len(grid)
                start += len(part)
                y1 = bbox[1] + (bbox[3] - bbox[1]) * start / len(grid)
                title, part = pop_title_row(part)
                header, rows = structure(part)
                header, rows = merge_split_header_columns(header, rows)
                for sub_header, sub_rows, col0, col1 in split_structured_side(header, rows):
                    if not sub_header:
                        continue
                    nl = _label_cols(sub_rows) if sub_rows else 0
                    if nl:
                        sub_rows = [r for r in sub_rows if any(r[:nl])]
                    width = max(1, len(header))
                    x0 = bbox[0] + (bbox[2] - bbox[0]) * col0 / width
                    x1 = bbox[0] + (bbox[2] - bbox[0]) * col1 / width
                    t = make_table(sub_header, sub_rows, title=title, allow_sparse=ruled,
                                   bbox=[round(v, 1) for v in (x0, y0, x1, y1)], source=f"native:{name}")
                    t["_ruled"] = ruled
                    out.append(t)
        return out

    cands = []
    for name, run in (("plumber-lines", lambda: _plumber(page, words)), ("fitz-lines", lambda: _fitz(fpage, words))):
        try:
            cands += build(run(), name, True)
        except Exception as e:  # one engine failing must not lose the page
            print(f"  {name} failed: {e}")
    trusted = [c["bbox"] for c in cands if not c["low_confidence"]]
    free = [w for w in words if not any(b[0] - 1 <= w[0] and w[2] <= b[2] + 1 and b[1] - 20 <= w[1] and w[3] <= b[3] + 1
                                        for b in trusted)]
    cands += build(whitespace_tables(free, float(page.width)), "whitespace", False)

    # Some dashboard exports use dense decorative rules.  The normal line strategies then merge
    # several independent tables into one unusable grid.  Strict line detection is deliberately a
    # fallback: pages that already have a trustworthy candidate keep their established behaviour.
    if not cands or any(c["low_confidence"] for c in cands):
        for name, run in (("plumber-lines-strict", lambda: _plumber_strict(page, words)),
                          ("fitz-lines-strict", lambda: _fitz_strict(fpage, words))):
            try:
                cands += build(run(), name, True)
            except Exception as e:
                print(f"  {name} failed: {e}")
    return cands


def _period_header_count(candidate: dict) -> int:
    """Number of distinct reporting periods represented by a candidate's headers."""
    from . import periods  # local import keeps the low-level table parser independent at import time

    found = periods.find_all(" | ".join(candidate.get("header", [])))
    return len({period.key for period, _ in found})


def pick(cands: list[dict]) -> list[dict]:
    """One table per region: reliable ruled tables, period-aware ruled tables, then the rest."""
    good = [c for c in cands if c["score"] > 0]
    def priority(c):
        if c["_ruled"] and not c["low_confidence"]:
            return 0
        # A wide, ruled table with several explicit period headers is more useful than a slightly
        # higher-scoring whitespace grid whose navigation/title text became its header.
        if c["_ruled"] and _period_header_count(c) >= 2:
            return 1
        return 2

    good.sort(key=lambda c: (priority(c), -c["score"]))
    chosen = []
    for c in good:
        if all(_overlap(c["bbox"], k["bbox"]) < 0.3 for k in chosen):
            chosen.append(c)
    for c in chosen:
        c.pop("_ruled", None)
    return sorted(chosen, key=lambda c: (round(c["bbox"][1] / 20), c["bbox"][0]))


def image_regions(fpage: pymupdf.Page, skip_backgrounds: bool = True) -> list[pymupdf.Rect]:
    """Bounding boxes of images large enough to hold a table. A page-sized picture with PDF text
    laid over it is a background (a cover or section slide), not a pasted table: its words are
    in the text layer already, and a scan's OCR text layer is read as native text."""
    page_area = fpage.rect.width * fpage.rect.height
    rects = [pymupdf.Rect(info["bbox"]) & fpage.rect for info in fpage.get_image_info()]
    rects = [r for r in rects if r.width * r.height >= MIN_IMAGE_AREA * page_area]
    if skip_backgrounds:
        words = fpage.get_text("words")
        rects = [r for r in rects if not is_background(fpage, r, words)]
    return rects


def is_background(fpage: pymupdf.Page, r: pymupdf.Rect, words: list) -> bool:
    """A picture covering most of the page with PDF text laid over it."""
    if r.width * r.height < 0.6 * fpage.rect.width * fpage.rect.height:
        return False
    return sum(1 for w in words if r.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))) >= 2


# The model reads text down to ~8px (measured on the fixtures); smaller text is scaled up to
# ~10px. Scaling more only makes images bigger and forces tiling.
MIN_TEXT_PX, TARGET_TEXT_PX = 8, 10


def text_scale(text_px: float, size_px: float = 0) -> float:
    """Render scale for a block: small text up to TARGET_TEXT_PX; a block bigger than
    MAX_IMAGE_PX shrunk to fit while its text stays readable (else it gets tiled)."""
    s = min(2.0, TARGET_TEXT_PX / text_px) if text_px < MIN_TEXT_PX else 1.0
    if size_px * s > MAX_IMAGE_PX and text_px * MAX_IMAGE_PX / size_px >= MIN_TEXT_PX:
        s = MAX_IMAGE_PX / size_px
    return s


@dataclass
class Block:
    """One table in a picture: its rendering at source resolution and its layout."""
    fpage: pymupdf.Page
    rect: pymupdf.Rect  # page coordinates
    dpi: float  # source resolution
    g: _Gray
    lay: _Layout
    stem: str
    model: str

    def page_rect(self, x0, y0, x1, y1) -> pymupdf.Rect:
        sx, sy = self.rect.width / self.g.w, self.rect.height / self.g.h
        return pymupdf.Rect(self.rect.x0 + x0 * sx, self.rect.y0 + y0 * sy,
                            self.rect.x0 + x1 * sx, self.rect.y0 + y1 * sy)


def _compose(fpage: pymupdf.Page, pieces: list[list[pymupdf.Rect]], dpi: float, path) -> None:
    """Render page rects at dpi and paste them as a grid (rows of pieces) into one image."""
    rows = [[fpage.get_pixmap(clip=r, dpi=max(72, round(dpi))) for r in row] for row in pieces]
    width = max(sum(p.width for p in row) for row in rows)
    height = sum(max(p.height for p in row) for row in rows)
    out = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    out.clear_with(255)
    y = 0
    for row in rows:
        x = 0
        for p in row:
            if p.n != 3 or p.alpha:
                p = pymupdf.Pixmap(pymupdf.csRGB, p)
            p.set_origin(x, y)
            out.copy(p, p.irect)
            x += p.width
        y += max(p.height for p in row)
    out.save(str(path))


def _nearest(cuts: list[int], target: float, lo: int, hi: int) -> int | None:
    inside = [c for c in cuts if lo < c < hi]
    return min(inside, key=lambda c: abs(c - target)) if inside else None


def _ask_tile(b: Block, ys: tuple[int, int], xs: tuple[int, int], dpi: float, tag: str,
              prompt: str = "") -> tuple[str, bool]:
    """Show the model rows ys x columns xs of a block, with the row-label column pasted on the
    left when the tile does not include it."""
    lab = picture.label_end(b.g, (0, 0, b.g.w, b.g.h), b.lay) if xs[0] > 0 else 0
    col_parts = ([(0, lab)] if xs[0] > 0 else []) + [xs]
    pieces = [[b.page_rect(x0, ys[0], x1, ys[1]) for x0, x1 in col_parts]]
    path = config.PAGE_IMG_DIR / f"{b.stem}{tag}.png"
    _compose(b.fpage, pieces, dpi, path)
    content = [llm.image_part(str(path)), {"type": "text", "text": prompt or TABLE_PROMPT}]
    text, cut = llm.chat_full(b.model, [{"role": "user", "content": content}], temperature=0,
                              max_tokens=config.VISION_MAX_TOKENS)
    # Keep the reply next to its image, to audit any extracted table.
    path.with_suffix(".txt").write_text(("[cut off]\n" if cut else "") + text, encoding="utf-8")
    return text.strip(), cut


def _best(text: str) -> tuple[str, list[str], list[list[str]]] | None:
    """The largest table in a reply, as (title, header, rows)."""
    parts = []
    for title, grid in markdown_tables(text):
        for part in (p for block in split_side_block(grid) for p in split_stacked(block)):
            bar, part = pop_title_row(part)
            header, rows = structure(part)
            if header:
                parts.append((title or bar, header, rows))
    return max(parts, key=lambda p: len(p[2]) * len(p[1]), default=None)


def _continuation(text: str, columns: list[str]) -> tuple[str, list[str], list[list[str]]] | None:
    """Rows of a later piece: any header line the model repeats is dropped; rows are fitted to
    the known columns."""
    grids = markdown_tables(text)
    if not grids:
        return None
    grid = max((g for _, g in grids), key=len)
    want = [c.strip().lower() for c in columns]
    rows = []
    for r in grid:
        if [c.strip().lower() for c in r] == want or (_is_header_row(r) and not rows and
                                                         not any(parse_number(c) is not None for c in r)):
            continue
        rows.append((r + [""] * len(columns))[:len(columns)])
    return "", list(columns), rows


def _columns_prompt(template: str, columns: list[str], **extra) -> str:
    return template.format(columns="\n".join(f"- {c or '(label)'}" for c in columns), n=len(columns),
                           header_line="| " + " | ".join(columns) + " |", **extra)


def _row_key(row: list[str]) -> str:
    """Row identity for stitching: the label, plus a second label column when there is one."""
    parts = [row[0]] + ([row[1]] if len(row) > 1 and row[1] and parse_number(row[1]) is None else [])
    return " / ".join(p.strip() for p in parts if p.strip())


def _read_tile(b: Block, ys, xs, dpi, tag, issues: list[str], depth: int = 0, columns=None):
    """(title, header, rows) for one tile. A tile that is not the first piece of its table gets
    the column names as text (columns). When the reply is cut off at the token limit, the rows
    received are kept and the model is asked, on the same image, for the rows after the last
    complete one (the image is then cached by the server)."""
    prompt = _columns_prompt(CONT_PROMPT, columns) if columns else ""
    text, cut = _ask_tile(b, ys, xs, dpi, tag, prompt)
    lines = sum(1 for line in text.splitlines() if line.strip().startswith("|"))
    if not cut and lines < 3 and "NO_TABLE" not in text:
        text, cut = _ask_tile(b, ys, xs, dpi * 0.67, tag + "_small", prompt)
    found = _continuation(text, columns) if columns else _best(text)
    rounds = 0
    while cut and found and found[2] and rounds < 6:
        rounds += 1
        title, header, rows = found
        rows = rows[:-1]  # the last row may be incomplete
        if not rows or not _row_key(rows[-1]):
            break
        last = _row_key(rows[-1])
        text, cut = _ask_tile(b, ys, xs, dpi, f"{tag}_more{rounds}", _columns_prompt(AFTER_PROMPT, header, last=last))
        more = _continuation(text, header)
        seen = {_row_key(r) for r in rows}
        new = [r for r in (more[2] if more else []) if _row_key(r) not in seen]
        if not new:
            break
        found = (title, header, rows + new)
    if cut:
        issues.append("reply cut off at the token limit; last rows may be missing")
    return found


def _stack_rows(parts: list, issues: list[str]):
    """Tiles of consecutive rows (later ones read against the first one's header) -> one table."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    title, header, rows = parts[0][0], parts[0][1], list(parts[0][2])
    for _, h, r in parts[1:]:
        if len(h) != len(header):
            issues.append(f"row tiles disagree on column count ({len(header)} vs {len(h)})")
        rows += [(row + [""] * len(header))[:len(header)] for row in r]
    return title, header, rows


def _join_cols(parts: list, issues: list[str]):
    """Tiles of columns (each with the label column) -> one table, joined row by row."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    title, header, rows = parts[0][0], list(parts[0][1]), [list(r) for r in parts[0][2]]
    for _, h, r in parts[1:]:
        if len(r) != len(rows):
            issues.append(f"column tiles disagree on row count ({len(rows)} vs {len(r)})")
        labels_differ = sum(1 for a, c in zip(rows, r) if a and c and a[0].strip().lower() != c[0].strip().lower())
        if labels_differ:
            issues.append(f"{labels_differ} row labels differ between column tiles")
        header += h[1:]
        for i in range(len(rows)):
            rows[i] += (r[i][1:] if i < len(r) else [""] * (len(h) - 1))
    return title, header, rows


def _read_all(b: Block, dpi: float, tag: str) -> tuple:
    """Read a block in tiles; returns (found, issues, n_tiles)."""
    w_px, h_px = b.rect.width * dpi / 72, b.rect.height * dpi / 72
    lab = picture.label_end(b.g, (0, 0, b.g.w, b.g.h), b.lay)
    n_cols = max(1, math.ceil(w_px / MAX_IMAGE_PX))
    n_rows = max(1, math.ceil(h_px / MAX_IMAGE_PX))
    col_cuts = picture.column_splits(b.g, (0, 0, b.g.w, b.g.h), b.lay)
    xs = [0]
    for k in range(1, n_cols):
        c = _nearest(col_cuts, lab + (b.g.w - lab) * k / n_cols, xs[-1] + 1, b.g.w - 1)
        if c is not None:
            xs.append(c)
    xs.append(b.g.w)
    ys = [0]
    for k in range(1, n_rows):
        c = _nearest(b.lay.splits, b.g.h * k / n_rows, max(ys[-1], b.lay.header_end) + 1, b.g.h - 1)
        if c is not None:
            ys.append(c)
    ys.append(b.g.h)

    issues, bands = [], []
    first = []  # header of each column tile, from the first row band
    for i, (y0, y1) in enumerate(zip(ys, ys[1:])):
        tiles = [_read_tile(b, (y0, y1), (x0, x1), dpi, f"{tag}_r{i}c{j}", issues,
                            columns=first[j][1] if i and j < len(first) and first[j] else None)
                 for j, (x0, x1) in enumerate(zip(xs, xs[1:]))]
        if i == 0:
            first = tiles
        bands.append(_join_cols(tiles, issues) if len(tiles) > 1 else tiles[0])
    found = _stack_rows(bands, issues) if len(bands) > 1 else bands[0]
    if found and not any(found[1][1:]):
        # Header lost: ask again for the first tile, header first.
        text, _ = _ask_tile(b, (ys[0], ys[1]), (xs[0], xs[1]), dpi, f"{tag}_hdr", HEADER_PROMPT)
        again = _best(text)
        if again and any(again[1][1:]):
            if len(xs) > 2:
                issues.append("header lost in a tiled table; not recovered")
            elif len(again[1]) == len(found[1]):
                found = (found[0] or again[0], again[1], found[2])
            elif len(again[2]) >= len(found[2]):  # a different reading: take it whole
                found = (found[0] or again[0], again[1], again[2])
    n_tiles = (len(xs) - 1) * (len(ys) - 1)
    if n_tiles > 1:
        issues.append(f"read in {len(xs) - 1}x{len(ys) - 1} tiles")
    return found, issues, n_tiles


def _mismatches(t: dict) -> int:
    return sum(1 for i in t["issues"] if i.startswith("total mismatch"))


def read_block(b: Block) -> list[dict]:
    """Read one table from a picture. Small text is scaled up; a block too big for one image is
    cut into column tiles (label column repeated) and row tiles (header given as text). When the
    table's totals do not add up, it is read once more at another scale and the reading whose
    totals reconcile best is kept: the table checks its own transcription."""
    dpi = b.dpi * text_scale(b.lay.text_px, max(b.g.w, b.g.h))

    def build(found, issues):
        if not found:
            return None
        title, header, rows = found
        t = make_table(header, rows, title=title, bbox=[round(v, 1) for v in b.rect], source="vision")
        t["issues"] += issues
        return t

    t = build(*_read_all(b, dpi, "")[:2])
    if t and _mismatches(t):
        # Another scale gives the model a different image; keep the reading that adds up.
        alt_scale = 0.8 if b.lay.text_px * dpi / b.dpi * 0.8 >= MIN_TEXT_PX else 1.25
        t2 = build(*_read_all(b, dpi * alt_scale, "_v2")[:2])
        if t2 and (_mismatches(t2), -t2["score"]) < (_mismatches(t), -t["score"]):
            t2["issues"].append(f"re-read after {_mismatches(t)} total mismatches; this reading has {_mismatches(t2)}")
            t = t2
        else:
            t["issues"].append("re-read after total mismatches did not reconcile better")
    return [t] if t and t["score"] > 0 else []


def vision_tables(fpage: pymupdf.Page, rect: pymupdf.Rect, img_stem: str, model: str) -> list[dict]:
    """Tables in a picture region: split it into one block per table and read each block on its
    own. Several tables in one call get merged, lose their headers, or all but the first are
    skipped."""
    rect = picture.ink_rect(fpage.get_pixmap(clip=rect, dpi=72), rect)
    dpi = picture.source_dpi(fpage, rect, TABLE_DPI)
    g = picture.Gray(fpage.get_pixmap(clip=rect, dpi=round(dpi), colorspace=pymupdf.csGRAY))
    sx, sy = rect.width / g.w, rect.height / g.h
    out = []
    for k, (x0, y0, x1, y1) in enumerate(picture.table_blocks(g), start=1):
        # A little white margin around the block, inside the region.
        box = (max(0, x0 - 3), max(0, y0 - 3), min(g.w, x1 + 3), min(g.h, y1 + 3))
        brect = pymupdf.Rect(rect.x0 + box[0] * sx, rect.y0 + box[1] * sy,
                             rect.x0 + box[2] * sx, rect.y0 + box[3] * sy)
        bg = picture.Gray(fpage.get_pixmap(clip=brect, dpi=round(dpi), colorspace=pymupdf.csGRAY))
        block = Block(fpage, brect, dpi, bg, picture.layout(bg, (0, 0, bg.w, bg.h)), f"{img_stem}_b{k}", model)
        out += read_block(block)
    return out


def extract_page_tables(
    page: pdfplumber.page.Page,
    fpage: pymupdf.Page,
    img_stem: str,
    use_vision: bool = True,
    model: str = config.VISION_MODEL,
) -> list[dict]:
    candidates = native_candidates(page, fpage)
    # Once the strict fallback finds trustworthy tables, do not mix in low-confidence fragments.
    # They are commonly chart grids or two dashboards merged together and can create plausible
    # but wrong facts.
    strict_trusted = any("-strict" in t.get("source", "") and not t["low_confidence"]
                         for t in candidates)
    if strict_trusted:
        strict_regions = [t["bbox"] for t in candidates if "-strict" in t.get("source", "")]
        candidates = [t for t in candidates if not t["low_confidence"] and (
            "-strict" in t.get("source", "")
            or not any(_overlap(t["bbox"], region) >= 0.3 for region in strict_regions)
        )]
    # A numeric block with no row labels cannot produce addressable facts.  Dashboard exports
    # sometimes repeat a selected period in such a detached block beside the complete table.
    # Filter before overlap selection so such a fragment cannot suppress a usable table beneath it.
    candidates = [t for t in candidates if t.get("rows") and _label_cols(t["rows"]) > 0]
    tables = pick(candidates)
    # Reconstructed dashboards often draw charts with ruled rectangles.  Their text layer is
    # complete, but table detectors see the bars/grid as a table with no usable headers or numeric
    # columns.  Sending those regions to the vision table reader is both inaccurate and expensive.
    # Keep genuine weak tables eligible for fallback; reject only the strong chart signatures.
    vector_text_page = len(fpage.get_text("words")) >= 50 and not fpage.get_image_info()
    if vector_text_page:
        def chart_like(t):
            issues = " ".join(t.get("issues", [])).lower()
            return t.get("low_confidence") and (
                "no number columns" in issues
                or "numbers without header or labels" in issues
                or (len(t.get("rows", [])) <= 3 and "header missing or numeric" in issues)
            )
        tables = [t for t in tables if not chart_like(t)]
    previous = None
    for t in tables:
        # Stacked dashboard subtables often print the period header once, then reuse the same
        # aligned columns.  Inherit it only across tables with the same geometry and width.
        if (previous and not any(t["header"]) and len(t["header"]) == len(previous["header"])
                and abs(t["bbox"][0] - previous["bbox"][0]) <= 3
                and abs(t["bbox"][2] - previous["bbox"][2]) <= 3):
            t["header"] = list(previous["header"])
            t["score"], t["issues"] = validate(t["header"], t["rows"], allow_sparse=True)
            t["low_confidence"] = t["score"] < config.TABLE_MIN_SCORE
        if t["header"] and t["header"][0].strip().lower() == "oneview":
            t["header"][0] = "Overview"
        if not t["title"] and t["bbox"]:
            t["title"] = _title_above(page, t["bbox"])
        if not t["title"] and t["header"] and t["header"][0].strip().lower() == "overview":
            t["title"] = "Overview"
        major = _major_heading_above(page, t["bbox"]) if t.get("bbox") else ""
        if major and major.lower() not in t["title"].lower():
            t["title"] = " / ".join(x for x in (major, t["title"]) if x)
        if any(t["header"]):
            previous = t

    vision_regions = 0

    def picture_tables(rect, stem):
        """Grid OCR first (seconds on CPU); the vision model only when that reads the region
        badly and vision is on, keeping whichever scores better."""
        nonlocal vision_regions
        from . import gridocr
        found = gridocr.gridocr_tables(fpage, rect, stem)
        weak = not found or any(t["low_confidence"] for t in found)
        if weak and use_vision and vision_regions < config.VISION_TABLE_MAX_REGIONS:
            vision_regions += 1
            try:
                seen = vision_tables(fpage, rect, stem, model)
                if seen and _mean_score(seen) > _mean_score(found):
                    return seen
            except Exception as exc:
                # A model timeout/unload must not discard native or OCR extraction for the page.
                print(f"  vision table fallback failed for {stem}: {exc}")
        return found

    # Weak native tables: read the same region from its picture and keep the better result.
    for i, t in enumerate(tables):
        if not t["low_confidence"]:
            continue
        rect = pymupdf.Rect(t["bbox"]) + (-6, -24, 6, 6)
        seen = picture_tables(rect & fpage.rect, f"{img_stem}_t{i + 1}")
        best = max(seen, key=lambda v: v["score"], default=None)
        comparable = (best and len(best.get("rows", [])) >= 0.6 * len(t.get("rows", []))
                      and len(best.get("header", [])) >= 0.6 * len(t.get("header", [])))
        if best and comparable and best["score"] > t["score"]:
            best["title"] = best["title"] or t["title"]
            best["bbox"] = t["bbox"]
            best["issues"].append(f"replaced native table scoring {t['score']}")
            tables[i] = best

    # Images that may be pasted tables, unless a native table already sits on them.
    for j, rect in enumerate(image_regions(fpage), start=1):
        if any(_overlap(t["bbox"], tuple(rect)) > 0.5 for t in tables if t["source"] not in ("vision", "gridocr")):
            continue
        tables += picture_tables(rect, f"{img_stem}_img{j}")
    return tables


def _mean_score(found: list[dict]) -> float:
    return sum(t["score"] for t in found) / len(found) if found else 0.0
