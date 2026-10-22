"""Layout of tables in a picture, from pixels only (no model).
Finds the separate tables in a picture, measures each one (rows, columns, text height, header
band) and plans how to show it to the vision model: at what scale, and in how many tiles so each
call holds a readable, bounded number of cells.
"""
from dataclasses import dataclass, field

import pymupdf

DARK = bytes(1 if v < 200 else 0 for v in range(256))  # translate table: 1 = dark pixel
RULE_RUN = 0.5   # a ruling line is one unbroken mark over half the block or more; text never is
BAND_SHARE = 0.6  # a filled band (header bar) is dark over most of the width


class Gray:
    """A grayscale picture as byte rows and byte columns."""

    def __init__(self, pix: pymupdf.Pixmap):
        if pix.n != 1 or pix.alpha:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        self.w, self.h = pix.width, pix.height
        s = pix.samples
        self.rows = [s[y * pix.stride:y * pix.stride + self.w] for y in range(self.h)]
        self.cols = [bytes(c) for c in zip(*self.rows)]


def dark(line: bytes) -> int:
    return len(line) - line.translate(DARK).count(0)


def _inked(line: bytes) -> bool:
    # A few dark pixels (noise, a frame line) do not make a line part of a table.
    return dark(line) >= max(2, 0.004 * len(line))


def _gaps(ink: list[bool]) -> list[tuple[int, int]]:
    """Blank runs strictly between the first and last ink."""
    runs, start = [], None
    for i, v in enumerate(ink):
        if not v and start is None:
            start = i
        elif v and start is not None:
            runs.append((start, i))
            start = None
    first = next((i for i, v in enumerate(ink) if v), None)
    return [r for r in runs if first is not None and r[0] > first]


def _segments(ink: list[bool]) -> list[tuple[int, int]]:
    """Split a profile at separator gaps: much wider than the usual gap and than ~1.2% of it."""
    idx = [i for i, v in enumerate(ink) if v]
    if not idx:
        return []
    lo, hi = idx[0], idx[-1] + 1
    gaps = _gaps(ink)
    if not gaps:
        return [(lo, hi)]
    lengths = sorted(b - a for a, b in gaps)
    floor = max(0.012 * len(ink), 8)
    # Many gaps = the line spacing of a table without rules; only much wider ones separate
    # tables. A few gaps between solid (ruled or banded) tables are the gutters themselves.
    cut = floor if len(gaps) <= 3 else max(3 * lengths[len(lengths) // 2], floor)
    # The one widest gap also separates when it clearly stands out from all the others
    # (a gutter between a table and a chart whose bars have regular gaps).
    widest = lengths[-1]
    standout = widest >= floor and (len(lengths) == 1 or widest >= 1.8 * lengths[-2])
    segs, a = [], lo
    for g0, g1 in gaps:
        if g1 - g0 >= cut or (standout and g1 - g0 == widest):
            segs.append((a, g0))
            a = g1
    segs.append((a, hi))
    return segs


# ---------- one table: rows, columns, header ----------

@dataclass
class Layout:
    text_px: float                                          # typical text height in pixels
    header_end: int                                         # y (block coordinates) above which a table is never cut (header region)
    splits: list[int] = field(default_factory=list)         # y between lines of text: safe to cut at
    bands: list[tuple[int, int]] = field(default_factory=list)   # dark filled bands (y0, y1)
    texts: list[tuple[int, int]] = field(default_factory=list)   # lines of text (y0, y1)


LIGHT = bytes(1 if v < 230 else 0 for v in range(256))  # 1 = any visible mark, incl. light rules


def light(line: bytes) -> int:
    return len(line) - line.translate(LIGHT).count(0)


def longest_mark(line: bytes) -> int:
    """Longest unbroken run of marked pixels."""
    return max(map(len, line.translate(LIGHT).split(b"\x00")), default=0)


def _runs(kinds: list) -> list[tuple]:
    out = []
    for i, k in enumerate(kinds):
        if out and out[-1][0] == k:
            out[-1] = (k, out[-1][1], i + 1)
        else:
            out.append((k, i, i + 1))
    return out


def vertical_rules(g: Gray, box: tuple[int, int, int, int]) -> list[int]:
    """x (block coordinates) of lines marked over most of the block height."""
    x0, y0, x1, y1 = box
    need = max(40, 0.3 * (y1 - y0))  # table rules can be shorter than a block with a heading
    return [x for x in range(x1 - x0) if longest_mark(g.cols[x0 + x][y0:y1]) >= need]


def layout(g: Gray, box: tuple[int, int, int, int]) -> Layout:
    """Lines of text, dark bands and header end of one table, ignoring ruling lines."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    vx = vertical_rules(g, box)
    kinds = []
    for y in range(h):
        row = g.rows[y0 + y][x0:x1]
        if dark(row) >= BAND_SHARE * w:
            kinds.append("band")  # dark fill (header band), or just a dense line of text
        elif longest_mark(row) >= RULE_RUN * w:
            kinds.append("rule")  # an unbroken line or light shading: part of the table, not text
        else:
            d = dark(row) - sum(1 for x in vx if row[x] < 200)
            kinds.append("text" if d >= max(2, 0.004 * w) else "blank")
    # A filled band is several pixels tall and unbroken; a thinner "band" is a dark rule if it is
    # unbroken, else dense text.
    runs = _runs(kinds)
    for k, a, b in runs:
        if k == "band" and b - a < 5:
            for y in range(a, b):
                kinds[y] = "rule" if longest_mark(g.rows[y0 + y][x0:x1]) >= RULE_RUN * w else "text"
    runs = _runs(kinds)
    bands = [(a, b) for k, a, b in runs if k == "band" and b - a >= 5]
    texts = [(a, b) for k, a, b in runs if k == "text" and b - a >= 2]
    heights = sorted(b - a for a, b in texts)
    text_px = float(heights[len(heights) // 2]) if heights else 10.0
    # Safe cuts: the middle of blank or rule runs lying between two lines of text.
    splits = [(a + b) // 2 for k, a, b in runs if k in ("blank", "rule") and 0 < a and b < h]
    # The table starts at its first ruling line or band; text above that (a page heading, a
    # table title) is not the header. Without rules, skip heading-sized lines at the top.
    marks = [(a, b) for k, a, b in runs if k in ("rule", "band")]
    lines_after = texts
    if marks:
        top = marks[0][0]
        lines_after = [t for t in texts if t[0] >= top]
    else:
        big = 1.3 * text_px
        k = 0
        while k < len(texts) - 1 and texts[k][1] - texts[k][0] > big:
            k += 1
        lines_after = texts[k:]
    top = lines_after[0][0] if lines_after else 0
    header_end = lines_after[0][1] if lines_after else 0
    first_band = next((bd for bd in bands if bd[0] >= top - 1), None)
    if first_band and (not lines_after or first_band[0] <= lines_after[0][0]):
        header_end = first_band[1]
        nxt = next((bd for bd in bands if bd[0] > first_band[0]), None)
        if nxt and nxt[0] - header_end <= 3:  # title bar + header band
            header_end = nxt[1]
    header_end = next((c for c in splits if c >= header_end), header_end)
    # Cuts only below the header.
    splits = [c for c in splits if c > header_end]
    return Layout(text_px=text_px, header_end=header_end, splits=splits,
                  bands=bands, texts=texts)


def column_ink(g: Gray, box: tuple[int, int, int, int], lay: Layout) -> tuple[list[int], set[int]]:
    """Per x (block coordinates): how many body lines have ink there; and the vertical rules."""
    x0, y0, x1, y1 = box
    rules = set(vertical_rules(g, box))
    lines = [(a, b) for a, b in lay.texts if a >= lay.header_end] or lay.texts
    votes = [0 if x in rules else sum(1 for a, b in lines if min(g.cols[x0 + x][y0 + a:y0 + b]) < 200)
             for x in range(x1 - x0)]
    return votes, rules


def column_splits(g: Gray, box: tuple[int, int, int, int], lay: Layout) -> list[int]:
    """x positions safe to cut at: vertical rules, or blank in (almost) every body line."""
    votes, rules = column_ink(g, box, lay)
    n = max(1, max(votes, default=1))
    return sorted(x for x in range(len(votes)) if x in rules or votes[x] <= 0.02 * n)


def label_end(g: Gray, box: tuple[int, int, int, int], lay: Layout) -> int:
    """x (block coordinates) where the row-label column ends: the first vertical rule or wide
    blank gap after the labels start."""
    votes, rules = column_ink(g, box, lay)
    n = max(1, max(votes, default=1))
    started, gap = False, 0
    for x, v in enumerate(votes):
        if x in rules:
            if started:
                return x
            continue
        if v > 0.1 * n:
            started, gap = True, 0
        elif started:
            gap += 1
            if gap >= max(4, 1.5 * lay.text_px):
                return x - gap // 2
    return len(votes) // 4


# ---------- the tables in a picture ----------

def band_splits(g: Gray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Tables stacked without a gutter: a dark header band with lines of text both above and
    below it (not a filled totals row at the bottom) starts a new table."""
    x0, y0, x1, y1 = box
    lay = layout(g, box)
    starts = []
    for a, b in lay.bands[1:]:
        prev = starts[-1] if starts else 0
        above = [t for t in lay.texts if prev < t[0] and t[1] <= a]
        below = [t for t in lay.texts if t[0] >= b]
        if len(above) >= 2 and len(below) >= 2:
            starts.append(a)
    if not starts:
        return [box]
    edges = [0] + starts + [y1 - y0]
    return [(x0, y0 + a, x1, y0 + b) for a, b in zip(edges, edges[1:])]


def table_blocks(g: Gray, max_depth: int = 4) -> list[tuple[int, int, int, int]]:
    """Pixel boxes of the separate tables in a picture: recursive XY-cut on white gutters, then
    a cut before each header band that follows rows of text. One box for the whole ink area when
    nothing separates."""
    blocks = []

    def cut(x0, y0, x1, y1, depth):
        segs = _segments([_inked(r[x0:x1]) for r in g.rows[y0:y1]])
        if not segs:
            return
        if len(segs) > 1 and depth < max_depth:
            for a, b in segs:
                cut(x0, y0 + a, x1, y0 + b, depth + 1)
            return
        y0, y1 = y0 + segs[0][0], y0 + segs[-1][1]
        csegs = _segments([_inked(c[y0:y1]) for c in g.cols[x0:x1]])
        if len(csegs) > 1 and depth < max_depth:
            for a, b in csegs:
                cut(x0 + a, y0, x0 + b, y1, depth + 1)
            return
        if csegs:
            blocks.append((x0 + csegs[0][0], y0, x0 + csegs[-1][1], y1))

    cut(0, 0, g.w, g.h, 0)
    # Headings, footnotes and logos are one or two lines; a table, however small, has more.

    def lines(b):  # lines of text, not counting ruling lines
        lay = layout(g, b)
        return len(lay.texts) + len(lay.bands)

    big = [b for b in blocks if b[2] - b[0] >= 60 and b[3] - b[1] >= 30
           and ((b[2] - b[0]) * (b[3] - b[1]) >= 0.04 * g.w * g.h or lines(b) >= 3)]
    if len(big) < 2:
        allb = blocks or [(0, 0, g.w, g.h)]
        big = [(min(b[0] for b in allb), min(b[1] for b in allb),
                max(b[2] for b in allb), max(b[3] for b in allb))]
    return [part for b in big for part in band_splits(g, b)]


def ink_rect(pix: pymupdf.Pixmap, rect: pymupdf.Rect, pad: float = 4) -> pymupdf.Rect:
    """rect shrunk to the non-white content of its rendering."""
    g = Gray(pix)
    ys = [y for y, r in enumerate(g.rows) if min(r) < 200]
    if not ys:
        return rect
    xs = [x for x, c in enumerate(g.cols) if min(c[ys[0]:ys[-1] + 1]) < 200]
    sx, sy = rect.width / g.w, rect.height / g.h
    ink = pymupdf.Rect(rect.x0 + xs[0] * sx, rect.y0 + ys[0] * sy,
                       rect.x0 + (xs[-1] + 1) * sx, rect.y0 + (ys[-1] + 1) * sy)
    return (ink + (-pad, -pad, pad, pad)) & rect


def source_dpi(fpage: pymupdf.Page, rect: pymupdf.Rect, default: float) -> float:
    """Resolution of the pictures under rect; rendering finer only adds blur and tokens."""
    dpis = [info["width"] / max(1.0, pymupdf.Rect(info["bbox"]).width) * 72
            for info in fpage.get_image_info() if pymupdf.Rect(info["bbox"]).intersects(rect)]
    return min(default, max(dpis)) if dpis else default
