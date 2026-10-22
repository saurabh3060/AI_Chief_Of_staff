"""Chart values read from the PDF itself: the printed number, its month, its series by color.
granite-vision reads chart numbers right but pairs them with the legend in reading order, so a
chart with one number inside each bar and one on the line comes out with the two series swapped
(CC12 p.7), whatever the prompt says. Slides exported from PowerPoint keep every data label, axis
label and legend entry as text, so the only thing left to work out is which series a label
belongs to. The legend gives each series a color, and a label sits on its own point: next to a
line marker of that color, inside a bar or stacked segment of that color, or just above a bar.
read_charts() finds each chart by its row of period labels (Jun-25 ... Feb-26). A chart it is
sure of comes back with granite-style text ("### Chart: title" + CSV, what facts.chart_csvs
reads); a chart it is not sure of comes back with only its region, for the vision model.
"""
import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pymupdf

ZOOM = 3  # pixels per point for color sampling
PERIOD = re.compile(
    r"(?i)^(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-\u2019' ]?\d{2,4}"
    r"|(?:q[1-4]|h[12])[-\u2019' ]?(?:fy)?[-\u2019' ]?\d{2,4}|fy[-\u2019' ]?\d{2,4})$")
NUMBER = re.compile(r"^[-+\u2212(]?[\u20b9$]?\d[\d,]*(?:\.\d+)?%?\)?$")
WHITE = 236          # every channel above this counts as background
LEGEND_BAND = 50     # points below the axis where a legend may sit
LEGEND_MAX_WORDS = 6


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    words: list

    @property
    def numeric(self) -> bool:
        return all(NUMBER.match(w[4]) for w in self.words)


@dataclass
class Series:
    name: str
    color: tuple
    kind: str   # "bar" (solid swatch) or "line" (thin swatch, marker or outline box)
    line: Line


@dataclass
class Chart:
    title: str
    bbox: tuple          # points
    periods: list[str]
    series: list[Series] = field(default_factory=list)
    text: str = ""       # granite-style output; empty when the chart was not read with confidence
    reason: str = ""


# ---------- page geometry ----------

def _lines(fpage: pymupdf.Page) -> list[Line]:
    groups = {}
    for w in fpage.get_text("words"):
        groups.setdefault((w[5], w[6]), []).append(w)
    out = []
    for ws in groups.values():
        ws.sort(key=lambda w: w[0])
        out.append(Line(min(w[0] for w in ws), min(w[1] for w in ws),
                        max(w[2] for w in ws), max(w[3] for w in ws),
                        " ".join(w[4] for w in ws), ws))
    return out


def _axes(words: list) -> list[list]:
    """Rows of 3+ period labels on one baseline, split where two charts sit side by side."""
    rows = []
    for w in sorted((w for w in words if PERIOD.match(w[4])),
                    key=lambda w: (w[1] + w[3]) / 2):
        cy = (w[1] + w[3]) / 2
        if rows and abs(cy - rows[-1][0]) <= 3:
            rows[-1][1].append(w)
        else:
            rows.append([cy, [w]])
    axes = []
    for _, ws in rows:
        ws.sort(key=lambda w: w[0])
        centers = [(w[0] + w[2]) / 2 for w in ws]
        gaps = [b - a for a, b in zip(centers, centers[1:])]
        if not gaps:
            continue
        typical = float(np.median(gaps))
        run = [ws[0]]
        for w, g in zip(ws[1:], gaps):
            if g > 1.6 * typical:
                axes.append(run)
                run = []
            run.append(w)
        axes.append(run)
    return [a for a in axes if len(a) >= 3]


class Pixels:
    def __init__(self, fpage: pymupdf.Page, words: list):
        pix = fpage.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), alpha=False)
        self.img = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        self.text = np.zeros(self.img.shape[:2], bool)  # pixels covered by any word: never a mark
        for w in words:
            x0, y0, x1, y1 = self._px(w[:4], pad=1)
            self.text[y0:y1, x0:x1] = True

    def _px(self, box, pad=0):
        h, w = self.img.shape[:2]
        x0, y0, x1, y1 = (int(round(v * ZOOM)) for v in box)
        return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)

    def colors(self, box, skip_text=True) -> np.ndarray:
        """Non-background colors inside a box (points), as an (n, 3) int array."""
        x0, y0, x1, y1 = self._px(box)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 3), int)
        px = self.img[y0:y1, x0:x1].astype(int)
        keep = ~(px > WHITE).all(axis=2)
        if skip_text:
            keep &= ~self.text[y0:y1, x0:x1]
        return px[keep]

    def ring(self, box, gap=1.5, width=1.5) -> tuple | None:
        """Most common color just outside a box, or None when it is background."""
        x0, y0, x1, y1 = box
        o = (x0 - gap - width, y0 - gap - width, x1 + gap + width, y1 + gap + width)
        i = (x0 - gap, y0 - gap, x1 + gap, y1 + gap)
        ox0, oy0, ox1, oy1 = self._px(o)
        ix0, iy0, ix1, iy1 = self._px(i)
        px = self.img[oy0:oy1, ox0:ox1].astype(int)
        mask = np.ones(px.shape[:2], bool)
        mask[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] = False
        mask &= ~self.text[oy0:oy1, ox0:ox1]
        px = px[mask]
        if not len(px):
            return None
        white = (px > WHITE).all(axis=1)
        if white.mean() >= 0.5:
            return None
        return _dominant(px[~white])

    def beside(self, box, gap=0.8, width=1.5) -> tuple | None:
        """Fill color left and right of a label, level with its middle; None on background.
        The sides stay inside a bar or a thin stacked segment where the ring above and below
        would already reach the neighbouring segments."""
        x0, y0, x1, y1 = box
        mid = (y1 - y0) * 0.2
        px = np.concatenate([
            self.colors((x0 - gap - width, y0 + mid, x0 - gap, y1 - mid), skip_text=False),
            self.colors((x1 + gap, y0 + mid, x1 + gap + width, y1 - mid), skip_text=False)])
        xa, ya, xb, yb = self._px((x0 - gap - width, y0 + mid, x0 - gap, y1 - mid))
        total = 2 * max(1, (xb - xa) * (yb - ya))
        if len(px) < 0.5 * total:
            return None
        return _dominant(px)


def _dominant(px: np.ndarray) -> tuple | None:
    if not len(px):
        return None
    q = (px // 8) * 8
    vals, counts = np.unique(q, axis=0, return_counts=True)
    top = vals[counts.argmax()]
    near = px[(np.abs(q - top) <= 8).all(axis=1)]
    return tuple(int(v) for v in near.mean(axis=0))


def _dist(a, b) -> float:
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def _near(px: np.ndarray, color, tol: float = 24) -> np.ndarray:
    """Which pixels are this color. Tight: light series colors sit close to gridline grey."""
    return np.sqrt(((px - np.array(color)) ** 2).sum(axis=-1)) <= tol


# ---------- legend ----------

def _legend_entry(line: Line, pix: Pixels) -> Series | None:
    """A short text line on background with a small colored swatch just left of it."""
    if line.numeric or len(line.words) > LEGEND_MAX_WORDS or any(PERIOD.match(w[4]) for w in line.words):
        return None
    if pix.ring((line.x0, line.y0, line.x1, line.y1)) is not None:
        return None  # text on a banner or a colored box: a title or a comment, not a legend
    h = line.y1 - line.y0
    region = (line.x0 - 26, line.y0 - 2, line.x0 - 0.5, line.y1 + 2)  # swatches run up to ~20pt
    px = pix.colors(region)
    if len(px) < 4:
        return None
    color = _dominant(px)
    x0, y0, x1, y1 = pix._px(region)
    block = pix.img[y0:y1, x0:x1].astype(int)
    match = _near(block, color) & ~pix.text[y0:y1, x0:x1]
    ys, xs = np.nonzero(match)
    if not len(xs) or (xs.max() - xs.min() + 1) > 0.9 * block.shape[1]:
        return None  # color runs across the whole region: a panel, not a swatch
    sw, sh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    solid_rows = sum(match[r].sum() >= 0.6 * sw for r in range(ys.min(), ys.max() + 1))
    kind = "bar" if sh >= 0.25 * h * ZOOM and solid_rows >= 0.5 * sh else "line"
    return Series(line.text, color, kind, line)


# ---------- one chart ----------

def _match(color, series: list[Series], prefer: str = "bar") -> Series | None:
    """The series drawn in this color; on a tie (an outline box colored like a bar) the kind asked for."""
    close = [s for s in series if _dist(s.color, color) <= 45]
    close.sort(key=lambda s: (s.kind != prefer, _dist(s.color, color)))
    return close[0] if close else None


def _evidence(label, series: list[Series], pix: Pixels) -> tuple[Series | None, Series | None]:
    """(sure, guess) series for one label. Sure: a line marker right around it, or the fill it
    sits in. Guess: the first fill under it (a label above a bar, or the outline of its box)."""
    x0, y0, x1, y1 = label[:4]
    fill = pix.beside((x0, y0, x1, y1))
    # A line colored like the fill behind the label, or like any bar, would only be counting fill.
    bar_colors = [s.color for s in series if s.kind == "bar"] + ([fill] if fill else [])
    around = pix.colors((x0 - 3, y0 - 8, x1 + 3, y1 + 8))
    best, best_n = None, 0
    for s in series:
        if s.kind != "line" or any(_dist(s.color, c) <= 45 for c in bar_colors) or not len(around):
            continue
        n = int(_near(around, s.color).sum())
        if n > best_n:
            best, best_n = s, n
    if best_n >= 8:
        return best, None
    if fill is not None:
        return _match(fill, series), None
    cx = (x0 + x1) / 2
    below = pix.colors((cx - 1, y1 + 0.5, cx + 1, y1 + 14))
    return None, (_match(tuple(int(v) for v in below[0]), series, prefer="line") if len(below) else None)


def _read(chart: Chart, labels: list, pix: Pixels, centers: list[float]) -> None:
    names = [s.name for s in chart.series]
    if len(set(names)) < len(names):
        chart.reason = "two legend entries with the same name"
        return
    series = chart.series or [Series(chart.title or "Value", (0, 0, 0), "bar", None)]
    columns: dict[int, list] = {}
    for w in labels:
        col = min(range(len(centers)), key=lambda i: abs(centers[i] - (w[0] + w[2]) / 2))
        columns.setdefault(col, []).append(w)
    grid: dict[tuple[int, str], str] = {}
    for col, ws in columns.items():
        if len(ws) > len(series):
            chart.reason = f"{len(ws)} numbers in {chart.periods[col]} for {len(series)} series"
            return
        sure, open_ = {}, []
        for w in ws:
            s, guess = (series[0], None) if len(series) == 1 else _evidence(w, series, pix)
            if s is None:
                open_.append((w, guess))
            elif s.name in sure:
                chart.reason = f"two values for {s.name} in {chart.periods[col]}"
                return
            else:
                sure[s.name] = w
        # Numbers printed for one period belong to different series: one number left over and
        # one series left over go together; otherwise fall back on the fill under the number.
        left = [s for s in series if s.name not in sure]
        for w, guess in open_:
            s = left[0] if len(open_) == 1 and len(left) == 1 else guess
            if s is None or s.name in sure:
                chart.reason = f"label {w[4]} in {chart.periods[col]} matches no legend color"
                return
            sure[s.name] = w
    for name, w in sure.items():
        grid[(col, name)] = w[4]
    for s in series:
        n = sum(1 for (c, name) in grid if name == s.name)
        if n < len(centers) / 2:
            chart.reason = f"{s.name} has values for only {n} of {len(centers)} periods"
            return
    buf = io.StringIO()
    out = csv.writer(buf, lineterminator="\n")
    out.writerow(["Month" if PERIOD.match(chart.periods[0]) else "Period"] + [s.name for s in series])
    for i, p in enumerate(chart.periods):
        out.writerow([p] + [grid.get((i, s.name), "") for s in series])
    chart.text = f"### Chart: {chart.title}\n{buf.getvalue().strip()}"


def read_charts(fpage: pymupdf.Page) -> list[Chart]:
    words = fpage.get_text("words")
    axes = _axes(words)
    if not axes:
        return []
    lines = _lines(fpage)
    pix = Pixels(fpage, words)
    legend = {id(ln): e for ln in lines if (e := _legend_entry(ln, pix))}
    # Pass 1: each axis, its data labels and its title.
    found = []
    for axis in axes:
        centers = [(w[0] + w[2]) / 2 for w in axis]
        gap = (centers[-1] - centers[0]) / (len(centers) - 1)
        x_lo, x_hi = centers[0] - gap / 2, centers[-1] + gap / 2
        a_top, a_bot = min(w[1] for w in axis), max(w[3] for w in axis)

        def in_column(w):
            cx = (w[0] + w[2]) / 2
            return abs(min(centers, key=lambda c: abs(c - cx)) - cx) <= 0.45 * gap

        # Walk up from the axis: data labels, then the first other text line is the title.
        # Before any label only a comment in a colored box may sit between the axis and the
        # labels; plain text there means this row is a table header, not a chart axis.
        above = sorted(
            (ln for ln in lines if ln.y1 <= a_top + 1 and ln.x1 > x_lo and ln.x0 < x_hi),
            key=lambda ln: -ln.y1)
        labels, title = [], None
        for ln in above:
            if id(ln) in legend:
                continue
            if ln.numeric:
                labels += [w for w in ln.words if in_column(w)]
            elif labels:
                title = ln
                break
            elif any(PERIOD.match(w[4]) for w in ln.words) or pix.ring((ln.x0, ln.y0, ln.x1, ln.y1)) is None:
                break
        if labels:
            top = title.y0 if title else min(w[1] for w in labels)
            found.append((axis, centers, gap, x_lo, x_hi, top, a_top, a_bot, labels, title))
    # Pass 2: each legend entry goes to one chart: the one it sits inside (between title and
    # axis), else the nearest axis above it within LEGEND_BAND.
    owner = {}
    for ln in lines:
        if id(ln) not in legend:
            continue
        inside, below = [], []
        for i, (_, _, gap, x_lo, x_hi, top, a_top, a_bot, _, _) in enumerate(found):
            if not (ln.x1 > x_lo - gap and ln.x0 < x_hi + gap):
                continue
            if top - (ln.y1 - ln.y0) <= ln.y0 <= a_top:
                inside.append(i)
            elif a_bot <= ln.y0 <= a_bot + LEGEND_BAND:
                below.append((ln.y0 - a_bot, i))
        if inside:
            owner[id(ln)] = inside[0]
        elif below:
            owner[id(ln)] = min(below)[1]
    charts = []
    for i, (axis, centers, gap, x_lo, x_hi, top, a_top, a_bot, labels, title) in enumerate(found):
        series = [legend[id(ln)] for ln in lines if owner.get(id(ln)) == i]
        series.sort(key=lambda s: (round(s.line.y0 / 4), s.line.x0))
        bottom = max([a_bot] + [s.line.y1 for s in series])
        chart = Chart(title.text if title else "", (x_lo, top, x_hi, bottom),
                      [w[4] for w in axis], series)
        _read(chart, labels, pix, centers)
        charts.append(chart)
    return charts


def retitle(text: str, title: str) -> str:
    """Vision output for one cropped chart, under the title found on the page."""
    body = "\n".join(ln for ln in text.splitlines() if not re.match(r"\s*#+\s*chart:", ln, re.I))
    return f"### Chart: {title}\n{body.strip()}" if title and body.strip() else text
