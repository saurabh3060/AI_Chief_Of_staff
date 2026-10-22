"""Turn one extracted page (ingest.py's page record) into blocks, metric series and facts.

Reads the table dicts from tables.py and the chart CSVs from ingest.describe_charts as they are;
nothing here changes how a table is read. What is added per value:
  - its period (periods.py), metric (metrics.yaml), unit and row hierarchy
  - a check against the PDF's own text layer: a number printed on the page is `validated`;
    a chart value that is not printed anywhere is an estimate
  - review issues: weak tables, totals that do not add up, unknown metric names, chart/table
    disagreements
"""
import csv
import difflib
import io
import re
from dataclasses import dataclass, field

import yaml

from . import config
from . import periods
from .periods import Period
from .tables import is_blank, is_malformed_number, parse_number

# ---------- metrics ----------


def norm_label(s: str) -> str:
    """'Spends (Cr.)' -> 'spends'; 'GNPA %' -> 'gnpa %'; 'Spend/Card/Day' -> 'spend/card/day'."""
    s = (s or "").lower().replace("₹", " ")
    s = re.sub(r"\((?:[^()]*\b(?:cr|crores?|lakhs?|lacs?|mn|bn|rs|inr|in|bps|no|nos|count|units|000|thousands?)\b[^()]*|%|₹)\)",
               " ", s)
    s = re.sub(r"\b(in\s+)?(₹|rs\.?|inr)\s*(cr|crore|lakh|mn|bn)\b\.?", " ", s)
    s = re.sub(r"[^\w%/+&\- ]", " ", s)
    s = re.sub(r"(\w)%", r"\1 %", s)  # 'GNPA%' is 'GNPA %'
    return re.sub(r"\s+", " ", s).strip(" -")


@dataclass
class Metric:
    id: str
    name: str
    kind: str
    range: tuple[float, float] | None = None
    synonyms: list[str] = field(default_factory=list)


def load_metrics() -> tuple[dict[str, Metric], dict[str, str]]:
    """(id -> Metric, normalized synonym -> id)."""
    items = yaml.safe_load(config.METRICS_YAML.read_text(encoding="utf-8")) or []
    by_id, lookup = {}, {}
    for it in items:
        m = Metric(it["id"], it["name"], it.get("kind", ""), tuple(it["range"]) if it.get("range") else None,
                   it.get("synonyms", []))
        by_id[m.id] = m
        for s in [m.name, *m.synonyms]:
            lookup.setdefault(norm_label(s), m.id)
    return by_id, lookup


METRICS, SYNONYMS = load_metrics()


def match_metric(name: str, *context: str) -> str | None:
    """Canonical id for a label, trying the label alone, then with its table title/section. A
    multi-word synonym ("sourcing total") also matches when its words appear in that order
    somewhere across the row's hierarchy and its name ("Sourcing Trends" > ... > "Total"), since a
    metric's name is often split across section, table title and row rather than said in one cell."""
    for cand in (name, *(f"{c} {name}" for c in context if c)):
        mid = SYNONYMS.get(norm_label(cand))
        if mid:
            return mid
    words = norm_label(" ".join([*context, name])).split()
    for syn, mid in SYNONYMS.items():
        syn_words = syn.split()
        if len(syn_words) < 2:
            continue
        it = iter(words)
        if all(w in it for w in syn_words):
            return mid
    return None


# ---------- units ----------

_SCALE_RE = re.compile(r"(₹|rs\.?|inr)?\s*(?:in\s+)?(?:\b|')(crores?|cr|lakhs?|lacs?|mn|million|bn|billion|000|thousands?)\b\.?",
                       re.I)
_SCALE_NAMES = {"crore": "Cr", "crores": "Cr", "cr": "Cr", "lakh": "lakh", "lakhs": "lakh", "lac": "lakh",
                "lacs": "lakh", "mn": "mn", "million": "mn", "bn": "bn", "billion": "bn", "000": "'000",
                "thousand": "'000", "thousands": "'000"}


def unit_of(*texts: str) -> str:
    """Unit named in labels/titles, most specific first: '%', 'bps', 'Cr', 'lakh', ..."""
    for t in texts:
        t = t or ""
        if re.search(r"%|\bpercent\b|\bratio\b", t, re.I):
            return "%"
        if re.search(r"\bbps\b", t, re.I):
            return "bps"
        m = _SCALE_RE.search(t)
        if m:
            return _SCALE_NAMES[m.group(2).lower()]
    return ""


# Multiplier to base units, for units the page names. Values are stored as printed (7,458 Cr is
# 7458); the scale is only there to compare across units, never guessed for an unnamed unit.
SCALES = {"Cr": 1e7, "lakh": 1e5, "mn": 1e6, "bn": 1e9, "'000": 1e3}


def kind_of(metric_id: str | None, unit: str, printed: list[str], name: str = "") -> str:
    if metric_id and METRICS[metric_id].kind:
        return METRICS[metric_id].kind
    if unit == "%" or (printed and all("%" in p for p in printed if p)):
        return "percent"
    if unit == "bps":
        return "bps"
    if re.search(r"\b(cases|accounts|cards|count|number of|customers)\b|\bno\.", name, re.I):
        return "count"
    if unit in ("Cr", "mn", "bn") or re.search(r"₹|\brs\b|\binr\b|amount|disbursement|spends?\b|balance|cost", name, re.I):
        return "amount"
    return ""


def entity_of(*texts: str) -> str:
    for t in texts:
        if re.search(r"\bconsolidated\b", t or "", re.I):
            return "consolidated"
        if re.search(r"\bstandalone\b", t or "", re.I):
            return "standalone"
    return ""


# ---------- the page's own numbers ----------


def printed_numbers(text: str) -> set[float]:
    """Every number in the page's text layer, for checking values read from pictures."""
    out = set()
    for tok in re.findall(r"\(?[-−+]?[₹$]?\d[\d,]*(?:\.\d+)?%?\)?", text or ""):
        v = parse_number(tok)
        if v is not None:
            out.add(round(v, 4))
            out.add(round(abs(v), 4))
    return out


def on_page(value: float, numbers: set[float]) -> bool:
    return round(value, 4) in numbers or round(abs(value), 4) in numbers


# ---------- tables ----------


@dataclass
class Series:
    name: str  # as printed: "Spends (Cr.)", "Proprietary"
    parent: str  # group row above, if any
    points: list[tuple[Period, str]]  # (period, printed value)


def header_period(h: str) -> Period | None:
    """A column header that names a period: 'Jun-25', 'FY24 / Q1', 'Q4FY26 (till Feb)'."""
    p = periods.parse(h.replace(" / ", " "))
    if p:
        return p
    parts = [periods.parse(x) for x in h.split(" / ")]
    parts = [x for x in parts if x]
    if parts:
        return parts[-1] if len(parts) == 1 else None
    # A measure named with its period ('POS % Share (Sep-24)'), unless the words make it a
    # comparison against that period ('Growth vs Sep-24'), which is not a value for it.
    found = periods.find_all(h)
    if len(found) == 1 and not _COMPARISON.search(column_measure(h)):
        return found[0][0]
    return None


_COMPARISON = re.compile(r"\b(?:vs\.?|versus|over|since|from|against|growth|change|chg|var(?:iance)?|"
                         r"[yqm]-?o-?[yqm]|diff(?:erence)?|delta)\b|%\s*(?:inc|dec)", re.I)


def header_from_text(page_text: str, n_cols: int) -> list[Period] | None:
    """A table printed without a header row often has its periods on a text line just above
    ('Q1FY26 Q2FY26 Q3FY26 Q4FY26 (till Feb)'). Returns the periods if one line has exactly
    one per data column."""
    for line in (page_text or "").splitlines():
        hits = periods.find_all(line)
        if len(hits) != n_cols or n_cols < 2:
            continue
        # Each period plus the words after it up to the next one, so "(till Feb)" is kept.
        segs = [line[s:(hits[i + 1][1][0] if i + 1 < len(hits) else len(line))] for i, (_, (s, _e)) in enumerate(hits)]
        parsed = [periods.parse(seg) or hits[i][0] for i, seg in enumerate(segs)]
        if len({p.key for p in parsed}) == n_cols:
            return parsed
    return None


def _label_cols(rows: list[list[str]]) -> int:
    n = 0
    for j in range(len(rows[0]) if rows else 0):
        vals = [r[j] for r in rows if j < len(r) and r[j] and not is_blank(r[j])]
        if vals and sum(parse_number(v) is not None for v in vals) / len(vals) >= 0.5:
            break
        n += 1
    return n


def table_series(t: dict, page_text: str) -> tuple[list[Series], str]:
    """Series in a table and the orientation found: 'columns' (periods across), 'rows'
    (periods down) or '' (no periods: nothing to store as facts)."""
    header, rows = list(t["header"]), [list(r) for r in t["rows"]]
    if not rows:
        return [], ""
    width = max(len(header), max(len(r) for r in rows))
    header += [""] * (width - len(header))
    rows = [r + [""] * (width - len(r)) for r in rows]
    nl = max(1, _label_cols(rows))

    row_periods = [periods.parse(r[0]) for r in rows]
    if sum(1 for p in row_periods if p) >= max(2, len(rows) // 2):
        out = []
        for j in range(nl, width):
            pts = [(p, r[j]) for p, r in zip(row_periods, rows) if p and r[j] and not is_blank(r[j])]
            if pts and header[j]:
                out.append(Series(header[j], "", pts))
        return out, "rows"

    col_periods = {j: header_period(header[j]) for j in range(nl, width)}
    if not any(col_periods.values()):
        guessed = header_from_text(page_text, width - nl)
        if guessed:
            col_periods = dict(zip(range(nl, width), guessed))
    if sum(1 for p in col_periods.values() if p) < 2:
        return [], ""
    measures = column_measures(header, col_periods)
    out, parent = [], ""
    for r in rows:
        label = " › ".join(c for c in r[:nl] if c).strip()
        cells = {j: r[j] for j, p in col_periods.items() if p}
        if label and all(not v or is_blank(v) for v in cells.values()):
            parent = label  # group heading row, e.g. "Retail" above its products
            continue
        if not label:
            continue
        for measure, cols in measures:
            pts = [(col_periods[j], cells[j]) for j in cols if cells[j] and not is_blank(cells[j])]
            if pts:
                out.append(Series(label, " › ".join(x for x in (parent, measure) if x), pts))
    return out, "columns"


def column_measure(h: str) -> str:
    """What a period column holds besides its period: 'POS % Share (Sep-24)' -> 'POS % Share',
    'Industry / Sep-24' -> 'Industry', a bare 'Sep-24' -> ''."""
    s = h.replace(" / ", " ")
    for _, (a, b) in sorted(periods.find_all(s), key=lambda x: -x[1][0]):
        s = s[:a] + " " + s[b:]
    s = re.sub(r"[\[\](){}]|\b(?:to|vs\.?)\b", " ", s)  # brackets and range words left by the period
    return re.sub(r"\s+", " ", s).strip(" -–/:,")


def column_measures(header: list[str], col_periods: dict) -> list[tuple[str, list[int]]]:
    """Period columns grouped by what they measure: [(measure, columns)], the plain one first.
    A table can give each period twice ('Sep'24 | POS % Share (Sep-24)'); one series per row
    would then hold two values per period. A measure is its own group when it repeats across
    periods, shares a period with another group, or there is no plain group; a one-off note on
    a single column ('Q4 FY26 (till Feb)') stays with the plain columns."""
    groups: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for j, p in col_periods.items():
        if p:
            m = column_measure(header[j])
            key = re.sub(r"[^a-z0-9%]", "", m.lower())
            groups.setdefault(key, []).append(j)
            names.setdefault(key, m)

    def keys(js):
        return {col_periods[j].key for j in js}

    plain = list(groups.get("", []))
    split = []
    for k, js in groups.items():
        if not k:
            continue
        clash = any(keys(js) & keys(o) for k2, o in groups.items() if k2 != k)
        if len(js) >= 2 or clash or "" not in groups:
            split.append((names[k], js))
        else:
            plain += js
    out = []
    for name, js in ([("", sorted(plain))] if plain else []) + split:
        out += _runs(name, sorted(js), header, col_periods)
    return out


def _runs(name: str, js: list[int], header: list[str], col_periods: dict) -> list[tuple[str, list[int]]]:
    """A group whose periods repeat is two tables side by side read as one grid ('Jan..Dec |
    label | Jan..Dec'): split it where a period comes round again. A later run is named by the
    non-period column just before it (the right-hand table's own label column), if any."""
    def gap(a: int, b: int) -> bool:  # a non-period column between two of the group's columns
        return any(not col_periods.get(k) and header[k] for k in range(a + 1, b))

    runs = [[]]
    for j in js:
        run = runs[-1]
        if col_periods[j].key in {col_periods[i].key for i in run}:
            # Cut where the tables meet: after the last non-period column in this run, if any
            # (the first table may lack a period the second has), else here.
            cut = next((n for n in range(len(run) - 1, 0, -1) if gap(run[n - 1], run[n])), len(run))
            runs[-1], moved = run[:cut], run[cut:]
            runs.append(moved)
        runs[-1].append(j)
    out = [(name, runs[0])]
    for n, run in enumerate(runs[1:], start=2):
        before = header[run[0] - 1] if run[0] - 1 not in col_periods or not col_periods[run[0] - 1] else ""
        label = column_measure(before) if before else ""
        out.append((" › ".join(x for x in (name, label or f"part {n}") if x), run))
    return out


def context_period(*texts: str) -> Period | None:
    """The one period a table or chart without a period axis is for: named in its title, else its
    section heading. Several candidates -> None, never a guess. Not the page text: a lone period
    there is usually commentary ("against ~116 Cr in Apr'25"), not the grid's date."""
    for t in texts:
        keys = {p.key: p for p, _ in periods.find_all(t or "")}
        if len(keys) == 1:
            return next(iter(keys.values()))
    return None


_NOT_A_MEASURE = re.compile(r"^\s*(page|pg|slide|s\.?\s*no|sl\.?\s*no|sr\.?\s*no|#|no\.?)\s*$", re.I)


def _numeric_share(cells: list[str]) -> float:
    vals = [c for c in cells if c and not is_blank(c)]
    return sum(parse_number(c) is not None for c in vals) / len(vals) if vals else 0.0


def category_series(labels: list[str], header: list[str], rows: list[list[str]], period: Period | None,
                    group_rows: bool = True) -> list[Series]:
    """A grid whose columns are not periods (segment x metric, product x value): one series per
    cell, named by its metric, with its category as the parent. The metric is on whichever axis
    names more known metrics; rows by default, as in most financial tables."""
    cols = [j for j in range(len(header)) if header[j] and not is_blank(header[j])
            and not _NOT_A_MEASURE.match(header[j])]
    col_hits = sum(1 for j in cols if match_metric(header[j]))
    row_hits = sum(1 for lab in labels if lab and match_metric(lab))
    metric_on_cols = col_hits > row_hits
    out, group = [], ""
    for lab, r in zip(labels, rows):
        cells = {j: r[j] for j in cols if j < len(r)}
        if group_rows and lab and all(not v or is_blank(v) for v in cells.values()):
            group = lab  # "Retail" above its products
            continue
        if not lab:
            continue
        for j, v in cells.items():
            if not v or is_blank(v):
                continue
            name, cat = (header[j], lab) if metric_on_cols else (lab, header[j])
            out.append(Series(name, " › ".join(x for x in (group, cat) if x), [(period, v)]))
    return out


_PERIODISH = re.compile(r"\b(?:fy|q[1-4]|h[12]|9m|ytd|qtd|mtd)|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                        r"[\s\-'’_]*\d{2}", re.I)


def periodish_headers(header: list[str], first_value_col: int) -> bool:
    """Column headers that look like periods the parser could not read ('FY_25 / 1-2', OCR-garbled
    quarters). Reading such a grid as categories would store every cell undated; better to flag it."""
    cols = [h for h in header[first_value_col:] if h and not is_blank(h)]
    return bool(cols) and sum(1 for h in cols if _PERIODISH.search(h)) >= max(2, len(cols) // 2)


def grid_is_numeric(header: list[str], rows: list[list[str]], first_value_col: int) -> bool:
    """Enough of a value grid to read as numbers: a named value column and mostly numbers in it.
    A contents list (section, page number) is not one."""
    cols = [j for j in range(first_value_col, len(header)) if header[j] and not _NOT_A_MEASURE.match(header[j])]
    if len(rows) < 2 or not cols:
        return False
    # Row labels must name categories: "1, 2, 3 ..." is an index whose real labels (often months)
    # were lost, and reading it would store "EMI › 1".
    labels = [" ".join(r[:first_value_col]).strip() for r in rows]
    named = sum(1 for lab in labels if re.search(r"[A-Za-z]{2,}", lab) and not periods.parse(lab))
    if named < 0.6 * len(labels):
        return False
    return _numeric_share([r[j] for r in rows for j in cols if j < len(r)]) >= 0.6


def _issue_cells(t: dict) -> tuple[set[tuple[str, str]], set[str]]:
    """(row, column) pairs named in tables.validate's total-mismatch issues, and malformed cells."""
    bad, malformed = set(), set()
    for msg in t.get("issues", []):
        m = re.match(r"total mismatch: '(.*?)' / '(.*?)'", msg)
        if m:
            bad.add(m.groups())
        m = re.match(r"malformed numbers: (.*)", msg)
        if m:
            malformed.update(x.strip() for x in m.group(1).split(","))
    return bad, malformed


def _in_bad(bad: set[tuple[str, str]], name: str, p: Period) -> bool:
    """Is this value the cell a total-mismatch issue names? Periods across: (row name, period
    column); periods down: (period row, series column)."""
    for row, col in bad:
        cp, rp = header_period(col), periods.parse(row)
        if (row == name and cp and cp.key == p.key) or (col == name and rp and rp.key == p.key):
            return True
    return False


def implied_by_total(series: list[Series], name: str, period: Period, parent: str = "") -> float | None:
    """Value a garbled cell must have for its column total to add up (one unknown only). Only
    series under the same parent count: a share column does not add into an amount column."""
    series = [s for s in series if s.parent == parent]
    total = next((s for s in series if re.search(r"\btotal\b", s.name, re.I)), None)
    if not total:
        return None
    tv = next((parse_number(v) for p, v in total.points if p and p.key == period.key), None)
    if tv is None:
        return None
    others = []
    for s in series:
        if s is total or s.name == name:
            continue
        v = next((parse_number(x) for p, x in s.points if p and p.key == period.key), None)
        if v is None:
            return None
        others.append(v)
    return tv - sum(others)


def digits_fit(printed: str, value: float) -> bool:
    """'11,25' could be a misread of 11,275: its digits appear in order in the value."""
    want = re.sub(r"\D", "", printed)
    have = str(int(round(abs(value))))
    it = iter(have)
    return bool(want) and all(ch in it for ch in want) and len(have) - len(want) <= 2


# ---------- charts ----------


_GROUPED = re.compile(r"[-−(]?[₹$]?(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d?(?:,\d{2})*,\d{3})(?:\.\d+)?\)?")


def rejoin_numbers(row: list[str], n: int) -> list[str] | None:
    """A CSV row with more fields than its header: numbers with thousands separators written
    without quotes ('Label,4,14,038,100%') were split at their commas. Rejoins them into at most
    n fields: the most fields that exactly one rejoining gives (a row may be shorter than the
    header), else None and the row is dropped. Percentages are never grouped."""
    if len(row) > 40:
        return None
    found: dict[int, list[list[str]]] = {}

    def go(i: int, acc: list[str]) -> None:
        if len(acc) > n:
            return
        if i == len(row):
            found.setdefault(len(acc), []).append(acc)
            return
        go(i + 1, acc + [row[i]])
        for k in range(i + 1, min(len(row), i + 6)):
            if not re.fullmatch(r"\d{2,3}(?:\.\d+)?\)?", row[k]):
                break
            joined = ",".join(row[i:k + 1])
            if _GROUPED.fullmatch(joined):
                go(k + 1, acc + [joined])

    go(0, [])
    return next((found[c][0] for c in sorted(found, reverse=True) if len(found[c]) == 1), None)


def chart_csvs(chart_desc: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """granite-vision chart2csv output -> [(chart title, header, rows)]."""
    out, title, buf = [], "", []

    def flush():
        nonlocal buf
        lines = [ln for ln in buf if ln.strip() and not ln.strip().startswith("```")]
        buf = []
        if len(lines) < 3:
            return
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
        rows = [[c.strip() for c in r] for r in rows if r]
        if rows:
            n = len(rows[0])
            rows = [rows[0]] + [r if len(r) <= n else rejoined for r in rows[1:]
                                if len(r) <= n or (rejoined := rejoin_numbers(r, n))]
        if len(rows) >= 3:
            out.append((title, rows[0], rows[1:]))

    for line in (chart_desc or "").splitlines():
        m = re.match(r"\s*#+\s*chart:\s*(.*)", line, re.I)
        if m:
            flush()
            title = m.group(1).strip()
        elif re.match(r"\s*#+\s*legend:", line, re.I):
            continue  # "### Legend: name | color | mark", the model's mapping notes, not data
        else:
            buf.append(line)
    flush()
    return out


# ---------- one page ----------


@dataclass
class PageOut:
    blocks: list[dict] = field(default_factory=list)  # kind, text, title, content, bbox, source, confidence
    series: list[dict] = field(default_factory=list)  # label, name, metric_id, kind, unit, ..., facts, block
    issues: list[tuple[str, str, str, str]] = field(default_factory=list)  # kind, ref, message, suggestion
    unmatched: list[str] = field(default_factory=list)


def _is_narrative(line: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", line)
    toks = line.split()
    nums = sum(1 for t in toks if parse_number(t) is not None or periods.parse(t))
    return len(words) >= 4 and nums <= 0.4 * max(1, len(toks))


# Text on a page besides tables and charts. "takeaway" is a short line saying what moved
# ("Spends up 12% MoM"); "footnote" qualifies a table ("* excludes ..."); "disclaimer" is
# boilerplate, searched last.
TEXT_KINDS = ("narrative", "takeaway", "footnote", "source", "disclaimer")
COMMENTARY_KINDS = TEXT_KINDS[:-1]  # what an answer may quote without being asked
_BULLET = re.compile(r"^[•●○◦▪■□➢➤►▶✓✔–—-]\s*")
_NOTE_START = re.compile(r"^(?:notes?\b|\*|#|\^|¹|²|³)", re.I)
_SOURCE_START = re.compile(r"^sources?\s*[:\-]", re.I)
_DISCLAIMER = re.compile(r"disclaimer|forward[- ]looking|not be relied|without (?:prior )?notice|strictly confidential|"
                         r"for internal (?:use|circulation)|does not constitute|no representation", re.I)
_MOVE = re.compile(r"\b(up|down|rose|risen|rise|rising|fell|fall|falling|increas\w*|decreas\w*|declin\w*|improv\w*|"
                   r"deteriorat\w*|grew|grow\w*|higher|lower|stable|flat|steady|at|vs\.?|versus|driven|due|compared|"
                   r"mom|qoq|yoy|peak\w*|highest|lowest|record|crossed|reached)\b", re.I)


def _numbers_outside_periods(line: str) -> list[re.Match]:
    spans = [s for _, s in periods.find_all(line)]
    return [m for m in _NUM_RE.finditer(line)
            if not any(a <= m.start() < b for a, b in spans) and parse_number(m.group().strip()) is not None]


def _is_takeaway(line: str) -> bool:
    """A short line stating a movement or a level: 'GNPA at 2.1%', 'Spends up 12% MoM'."""
    words = re.findall(r"[A-Za-z]{2,}", line)
    toks = line.split()
    if len(words) < 2 or len(toks) > 30 or not _MOVE.search(line):
        return False
    nums = _numbers_outside_periods(line)
    if len(nums) > 0.5 * len(toks):
        return False  # a table row that happens to contain "at"
    return bool(nums) or bool(metrics_in(line))


def _echoes_table(line: str, cell_words: set[str]) -> bool:
    """A line of page text that is a table or chart row read again from the text layer."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", line)]
    return bool(cell_words) and len(words) >= 1 and sum(w in cell_words for w in words) / len(words) >= 0.7


def text_kind(para: str, bulleted: bool) -> str:
    if _SOURCE_START.match(para):
        return "source"
    if _NOTE_START.match(para):
        return "footnote"
    if _DISCLAIMER.search(para):
        return "disclaimer"
    if bulleted or _is_takeaway(para):
        return "takeaway"
    return "narrative"


def narrative_blocks(text: str, cell_words: set[str] | None = None) -> list[tuple[str, str]]:
    """(kind, text) paragraphs from the page text: commentary, takeaways and notes, not chart
    labels or table rows. Wrapped prose lines join into one paragraph; a bullet, a note or a
    short takeaway line starts its own."""
    cell_words = cell_words or set()
    paras: list[tuple[list[str], bool]] = []  # (lines, bulleted)
    open_ = False
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or _echoes_table(s, cell_words):
            open_ = False
            continue
        bullet = bool(_BULLET.match(s)) and not re.match(r"^[-–—]\s*\d", s)  # "-2.1%" is a number
        body = _BULLET.sub("", s) if bullet else s
        prose, short = _is_narrative(body), _is_takeaway(body)
        if not (prose or short):
            open_ = False
            continue
        starts = bullet or _NOTE_START.match(body) or _SOURCE_START.match(body) or not prose
        if open_ and not starts:
            paras[-1][0].append(body)
        else:
            paras.append(([body], bullet))
        open_ = prose  # a wrapped sentence goes on; a one-line takeaway stands alone
    return [(text_kind(" ".join(lines), bulleted), " ".join(lines)) for lines, bulleted in paras]


# ---------- numbers stated in text ----------

_NUM_RE = re.compile(r"(?<![\w.])\(?[-−+]?[₹$]?\d[\d,]*(?:\.\d+)?\)?(?:\s?%)?")
_UNIT_AFTER = re.compile(r"^\s*(bps|basis points|crores?|cr\b\.?|lakhs?|lacs?|mn\b|million|bn\b|billion|'000)", re.I)
_LEVEL_WORD = re.compile(r"\b(to|at|of|was|were|is|are|stood|from|reached|touched|crossed|vs\.?|versus|against|"
                         r"remained|maintained)\s*[₹$]?\s*$", re.I)
_CHANGE_BEFORE = re.compile(r"\b(up|down|by|rose|rise|risen|fell|fall|grew|growth|increas\w*|decreas\w*|declin\w*|"
                            r"improv\w*|deteriorat\w*|higher|lower|change|jump\w*|drop\w*)\b", re.I)
# "~10% of Total Spends" is a share of spends, not spends; "aiming for ~300 Cr" is a target;
# "since Oct'25" starts a range and is not the period of the value.
_SHARE_AFTER = re.compile(r"^\s*(?:%\s*)?(?:of|share|contribution|penetration)\b", re.I)
_FORWARD = re.compile(r"\b(aim\w*|target\w*|expect\w*|plan\w*|guidance|budget\w*|forecast\w*|project\w*|"
                      r"estimat\w*|likely|should|will|would|could)\b", re.I)  # not "may": the month
_RANGE_START = re.compile(r"\b(since|from|after|till|until|up\s?to|before)\s*$", re.I)
_CHANGE_AFTER = re.compile(r"^\s*(?:%\s*)?(mom|qoq|yoy|m-o-m|q-o-q|y-o-y|growth|increase|decline|rise|fall|higher|"
                           r"lower|jump|drop|change)\b", re.I)


def metrics_in(text: str) -> list[tuple[int, str]]:
    """(position, metric id) of each known metric named in text, longest synonym first."""
    low, found, taken = (text or "").lower(), [], []
    for syn, mid in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        if len(syn) < 3:
            continue
        for m in re.finditer(rf"(?<![\w]){re.escape(syn)}(?![\w])", low):
            if not any(a < m.end() and m.start() < b for a, b in taken):
                taken.append(m.span())
                found.append((m.start(), mid))
    return sorted(found)


def text_numbers(para: str) -> list[dict]:
    """Numbers a line of commentary states, each tied to the metric named before it in its clause
    (the subject) and the nearest period. {name, metric_id, unit, period, printed, value, change,
    sure}. `sure` needs a known metric whose kind fits the number and a period; anything less is
    kept, flagged, and never computed with."""
    out = []
    for clause in re.split(
        r"(?<=[.;])\s{1,20}|,\s{1,20}(?=[A-Za-z])|\s{1,20}(?:and|while|whereas|but)\s{1,20}", para
    ):
        forward = bool(_FORWARD.search(clause))
        found_periods = [(p, s) for p, s in periods.find_all(clause) if not _RANGE_START.search(clause[:s[0]])]
        mets = metrics_in(clause)
        nums = _numbers_outside_periods(clause)
        for i, m in enumerate(nums):
            tok = m.group().strip()
            v = parse_number(tok)
            if v is None:
                continue
            if re.match(r"[kKlL]?\+", clause[m.end():m.end() + 2]):
                continue  # a bucket's name ('30+ DPD', '1.2L+ exposure'), not a value
            bare_int = re.fullmatch(r"\d+", tok)
            if bare_int and (1900 <= v <= 2100 or (v < 10 and not _UNIT_AFTER.match(clause[m.end():]))):
                continue  # a year, a count word or a footnote number
            u = _UNIT_AFTER.match(clause[m.end():])
            word = u.group(1).lower() if u else ""
            unit = "%" if "%" in tok else "bps" if word.startswith(("bps", "basis")) else unit_of(word) if word else ""
            before, after = clause[max(0, m.start() - 30):m.start()], clause[m.end():m.end() + 15]
            change = not _LEVEL_WORD.search(before) and bool(_CHANGE_BEFORE.search(before) or _CHANGE_AFTER.match(after))
            share = bool(_SHARE_AFTER.match(clause[m.end() - (1 if tok.endswith("%") else 0):]))
            mid = next((mid for pos, mid in reversed(mets) if pos < m.start()), None)
            if mid and not change and not share:
                pct_metric = METRICS[mid].kind == "percent"
                if pct_metric != (unit == "%") and unit != "bps":
                    mid = None  # "Spends 12%" is not a spends level; the kinds disagree
            elif share:
                mid = None
            # Its period lies between the numbers either side of it: in "60 Cr against ~116 Cr in
            # Apr'25", Apr'25 belongs to 116 only.
            lo = nums[i - 1].end() if i else 0
            hi = nums[i + 1].start() if i + 1 < len(nums) else len(clause)
            # One written after the number wins ("7,458 Cr in Feb-26"), else the last before it.
            after_ = [p for p, s in found_periods if m.end() <= s[0] and s[1] <= hi]
            before_ = [p for p, s in found_periods if lo <= s[0] and s[1] <= m.start()]
            period = after_[0] if after_ else before_[-1] if before_ else None
            if mid:
                name = METRICS[mid].name + (" (change)" if change else "")
            else:
                name = re.sub(r"\s+", " ", clause).strip()[:80]
            if forward:
                name = f"target/outlook: {name}"
            out.append({"name": name, "metric_id": "" if change or forward or not mid else mid, "unit": unit,
                        "period": period, "printed": tok, "value": v, "change": change,
                        "sure": bool(mid and period and not forward)})
    return out


def _plabel(p: Period | None) -> str:
    return p.label if p else "period not stated"


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.005 * abs(b), 0.05)


def same_as_table(out: PageOut, name: str, title: str, pts: list[tuple[Period, str]]) -> bool:
    """Is this chart series a table already on the page? granite-vision often transcribes a
    table and its chart both. Same metric (or name) and most shared periods agree -> skip it,
    reporting the periods where the two readings differ."""
    # Same metric id or same name: the table side may have dropped an id the chart side still
    # guesses (a "Credit Cost" amount is not the credit-cost % metric).
    keys = {match_metric(name, title), norm_label(name)} - {None, ""}
    chart = {p.key: parse_number(v) for p, v in pts}

    def same_name(s: dict) -> bool:
        if keys & ({s["metric_id"], norm_label(s["name"])} - {None, ""}):
            return True
        # One reading corrects the other's spelling ('Attributible' printed, 'Attributable' read).
        a, b = norm_label(name), norm_label(s["name"])
        return len(a) >= 6 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.9

    for s in out.series:
        if s["source"] == "chart" or not same_name(s):
            continue
        table = {f["period"].key: f for f in s["facts"] if f["period"]}  # undated values cannot match a chart point
        shared = [k for k in chart if k in table and chart[k] is not None]
        if not shared:
            continue
        agree = [k for k in shared if _close(chart[k], table[k]["value"])]
        if len(agree) / len(shared) < 0.6:
            continue  # same name, different numbers: another metric, keep both
        for k in shared:
            if k not in agree:
                f = table[k]
                out.issues.append(("chart_table_mismatch", f"{s['label']} @ {f['period'].label}",
                                   f"chart reads {chart[k]:g}, table {f['value']:g}", "table value kept"))
                f["confidence"] = min(f["confidence"], 0.6)
        for k in agree:
            table[k]["validated"] = True  # two independent readings agree
        return True
    return False


def add_text_series(out: PageOut, para: str, section: str, doc_title: str, block_idx: int) -> None:
    """Series for the numbers a paragraph states. Points of one metric share a series while their
    periods differ ("from 7,200 Cr in Jan-26 to 7,458 Cr in Feb-26"); an undated number gets its
    own, so a 'latest' lookup cannot hide it behind another."""
    open_: dict[tuple[str, str], dict] = {}
    for n in text_numbers(para):
        key = (n["name"], n["unit"])
        s = open_.get(key)
        if s is None or n["period"] is None or any(f["period"] is None or f["period"].key == n["period"].key
                                                   for f in s["facts"]):
            path = [x for x in (section, "in text", n["name"]) if x]
            s = {"label": " › ".join(path), "path": path, "name": n["name"], "metric_id": n["metric_id"],
                 "unit": n["unit"], "scale": SCALES.get(n["unit"]),
                 "kind": kind_of(n["metric_id"] or None, n["unit"], [n["printed"]], n["name"]),
                 "entity": entity_of(para, section, doc_title), "source": "text", "confidence": 0.8,
                 "block": block_idx, "facts": []}
            out.series.append(s)
            open_[key] = s
        if n["sure"]:
            conf, note = 0.8, "stated in the commentary"
        else:
            conf = 0.5
            note = "stated in the commentary; " + ("no period given with it" if not n["period"]
                                                   else "not a known metric, labelled by its sentence")
        s["confidence"] = min(s["confidence"], conf)
        # validated = the number is printed and its metric and period are clear; a loose reading
        # shows as unverified.
        s["facts"].append({"period": n["period"], "value": n["value"], "printed": n["printed"],
                           "validated": n["sure"], "confidence": conf, "note": note})


_ANCHOR_STOP = set("""the and for with from this that into over than per total month year quarter value values
change growth trend data figures amount""".split())


def _words(*texts: str) -> set[str]:
    return {w for t in texts for w in re.findall(r"[a-z]{4,}", (t or "").lower()) if w not in _ANCHOR_STOP}


def anchor_text(out: PageOut, section: str) -> None:
    """Tie each text block to the table or chart it talks about: the one whose title, headers or
    series names share the most words with it, or the page's only one. A tie leaves it unanchored."""
    targets = [i for i, b in enumerate(out.blocks) if b["kind"] in ("table", "chart")]
    if not targets:
        return
    common = _words(section)
    vocab = {}
    for i in targets:
        b = out.blocks[i]
        c = b.get("content") or {}
        names = [s["name"] for s in out.series if s["block"] == i]
        vocab[i] = _words(b.get("title", ""), *(c.get("header") or []), *names) - common
    for b in out.blocks:
        if b["kind"] not in TEXT_KINDS:
            continue
        words = _words(b["text"].split("\n", 1)[-1]) - common
        scored = sorted(((len(words & vocab[i]), i) for i in targets), reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            b["anchor"] = scored[0][1]
        elif len(targets) == 1:
            b["anchor"] = targets[0]


def page_facts(rec: dict, doc_title: str) -> PageOut:
    out = PageOut()
    page_text = rec.get("text", "")
    section = rec.get("heading", "") or ""
    numbers = printed_numbers(page_text)
    page_unit = unit_of(*[ln for ln in page_text.splitlines() if re.search(r"\b(all|figures|amount)\b.*\bin\b", ln, re.I)])
    context = f"{doc_title} | p.{rec['page']} | {section}".strip(" |")

    charts = chart_csvs(rec.get("chart_desc", ""))
    grids = [[t.get("header") or [], *t.get("rows", [])] for t in rec.get("tables", [])] + \
            [[h, *r] for _, h, r in charts]
    cell_words = {w.lower() for g in grids for row in g for c in row for w in re.findall(r"[A-Za-z]{2,}", c or "")}

    if section:
        out.blocks.append({"kind": "heading", "text": f"{context}\n{section}", "title": section,
                           "section_path": [section]})
    text_blocks = []
    for kind, para in narrative_blocks(page_text, cell_words):
        if norm_label(para) == norm_label(section):
            continue
        out.blocks.append({"kind": kind, "text": f"{context}\n{para}", "title": "", "section_path": [section]})
        text_blocks.append((len(out.blocks) - 1, kind, para))

    def add_series(name, parent, title, pts, source, base_conf, verify, block_idx, bad=(), malformed=(),
                   all_series=(), explained=None):
        explained = set() if explained is None else explained  # periods whose total gap is explained
        # Values printed with '%' are percentages whatever the table's title says ('Book (Cr)'
        # over a count column and its share column).
        printed_pct = bool(pts) and all("%" in v for _, v in pts if v)
        unit = "%" if printed_pct else unit_of(name, parent, title) or page_unit
        mid = match_metric(name, title, parent, section)
        if mid and METRICS[mid].kind == "percent" and unit != "%" and not any("%" in v for _, v in pts):
            rng = METRICS[mid].range
            vals = [x for _, v in pts if (x := parse_number(v)) is not None]
            if rng and vals and any(not (rng[0] <= x <= rng[1]) for x in vals):
                # "GNPA 426" in a loss table is the amount, not the GNPA % the name also means.
                mid = None
        if not mid and not re.fullmatch(r"total", norm_label(name)):
            out.unmatched.append(name)
        if re.fullmatch(r"[\d.,\s]+", name) and not source.startswith("native"):
            # Only a picture's reading can turn 'B1' into '81'; a text-layer label is as printed.
            out.issues.append(("numeric_label", f"{title} › {name}",
                               f"row label '{name}' is only digits: likely misread (B1 read as 81?)",
                               "correct the label in the Review tab"))
        same = {norm_label(section), norm_label(name)}
        path = [x for x in [section, title if title and norm_label(title) not in same else "", parent, name] if x]
        label = " › ".join(path)
        s = {"label": label, "path": path, "name": name, "metric_id": mid or "", "unit": unit,
             "scale": SCALES.get(unit), "kind": kind_of(mid, unit, [v for _, v in pts], f"{name} {title}"),
             "entity": entity_of(title, section, doc_title), "source": source, "confidence": base_conf,
             "block": block_idx, "facts": []}
        for p, printed in pts:
            v = parse_number(printed)
            note, validated, conf = "", True, base_conf
            if v is None:
                if p and (printed in malformed or is_malformed_number(printed)):
                    implied = implied_by_total(all_series, name, p, parent)
                    if implied is not None and digits_fit(printed, implied):
                        v, validated, conf = implied, False, 0.5
                        explained.add(p.key)
                        note = f"printed '{printed}' is garbled; value implied by the column total"
                        out.issues.append(("garbled_number", f"{label} @ {p.label}",
                                           f"'{printed}' is not a valid number", f"column total implies {implied:,.0f}"))
                if v is None:
                    continue
            elif verify:
                validated = on_page(v, numbers)
                if not validated and source == "chart":
                    conf = min(conf, 0.6)
                    note = "estimated from the chart (value not printed as text), approximate"
                elif not validated:
                    # Printed digits read from a pasted picture: usually exact, but the PDF has no
                    # text layer there to check them against.
                    conf = min(conf, 0.8)
                    note = "read from a picture of the table; not cross-checked"
            # A total that does not add up: distrust that total cell, unless a garbled cell in the
            # same column already explains the gap.
            if p and _in_bad(bad, name, p) and p.key not in explained:
                conf = min(conf, 0.6)
                validated = False
                note = note or "this total does not match the items above it"
            rng = METRICS[mid].range if mid else None
            if rng and not (rng[0] <= v <= rng[1]):
                out.issues.append(("out_of_range", f"{label} @ {_plabel(p)}", f"{v} is outside {rng}", ""))
                conf = min(conf, 0.5)
            s["facts"].append({"period": p, "value": v, "printed": printed, "validated": validated,
                               "confidence": conf, "note": note})
        if s["facts"]:
            out.series.append(s)

    def undated(what: str, title: str) -> None:
        out.issues.append(("undated", title or what, f"{what} has no period axis and names no single period",
                           "values stored as 'period not stated'"))

    for t in rec.get("tables", []):
        if t.get("score", 0) <= 0:
            continue
        title = t.get("title", "") or ""
        md_rows = [t["header"]] + t["rows"]
        md = "\n".join("| " + " | ".join(r) + " |" for r in md_rows)
        conf = float(t.get("score", 0))
        out.blocks.append({"kind": "table", "text": f"{context}\n{title}\n{md}".strip(), "title": title,
                           "content": {"header": t["header"], "rows": t["rows"]}, "bbox": t.get("bbox"),
                           "source": t.get("source", ""), "confidence": conf,
                           "section_path": [x for x in (section, title) if x]})
        bidx = len(out.blocks) - 1
        if t.get("low_confidence") or t.get("issues"):
            out.issues.append(("table_check", title or f"table at {t.get('bbox')}",
                               f"score {conf}; " + "; ".join(t.get("issues", [])[:4]), ""))
        series, orient = table_series(t, page_text)
        bad, malformed = _issue_cells(t)
        verify = not str(t.get("source", "")).startswith("native")
        if not orient and t["rows"] and conf >= 0.5:
            # No period axis (segment x metric): read it as categories, dated by its context.
            width = max(len(t["header"]), max(len(r) for r in t["rows"]))
            rows = [list(r) + [""] * (width - len(r)) for r in t["rows"]]
            header = list(t["header"]) + [""] * (width - len(t["header"]))
            nl = max(1, _label_cols(rows))
            if periodish_headers(header, nl):
                out.issues.append(("period_headers", title or f"table at {t.get('bbox')}",
                                   "column headers look like periods but could not be read: "
                                   + ", ".join(h for h in header[nl:nl + 4] if h), "no values stored from this table"))
            elif grid_is_numeric(header, rows, nl):
                period = context_period(title, section, doc_title)
                if period is None:
                    undated("table", title)
                labels = [" › ".join(c for c in r[:nl] if c).strip() for r in rows]
                series = category_series(labels, [""] * nl + header[nl:], rows, period)
        explained: set[str] = set()
        for s in series:
            add_series(s.name, s.parent, title, s.points, t.get("source", "table"), conf, verify, bidx,
                       bad, malformed, series, explained)

    for title, header, rows in charts:
        col_p = [header_period(h) for h in header[1:]]
        if sum(1 for p in col_p if p) >= max(2, len(col_p) // 2):
            # Periods across the top, one series per row: turn it so periods run down.
            header, rows = [header[0] or "Period", *(r[0] for r in rows if r)], \
                [[h, *(r[j + 1] if j + 1 < len(r) else "" for r in rows if r)] for j, h in enumerate(header[1:])]
        row_p = [periods.parse(r[0]) if r else None for r in rows]
        over_time = sum(1 for p in row_p if p) >= max(2, len(rows) // 2)
        if not over_time and not grid_is_numeric(header, rows, 1):
            continue  # not a chart with values (a contents list, a diagram's labels)
        if not over_time and (periodish_headers(header, 1) or
                              periodish_headers([""] + [r[0] for r in rows if r], 1)):
            out.issues.append(("period_headers", title or section, "chart periods could not be read", ""))
            continue
        new_cols = None
        if over_time:
            cols = {j: [(p, r[j]) for p, r in zip(row_p, rows) if p and j < len(r) and r[j] and not is_blank(r[j])]
                    for j in range(1, len(header))}
            new_cols = [j for j, pts in cols.items() if pts and not same_as_table(out, header[j], title, pts)]
            if not new_cols:
                # A table the vision model also transcribed as a "chart": its readings were
                # checked against the table above; a second, noisier copy as a passage only
                # competes with the table in search.
                continue
        # Pipe-separated like table blocks: a comma join made "46,08,807" and "25,670" one run of
        # digits ("Feb-26,46,08,807,25,670") that neither a model nor a parser can split back.
        md = "\n".join("| " + " | ".join(r) + " |" for r in [header] + rows)
        shown = title or f"{section} (chart)".strip()
        out.blocks.append({"kind": "chart", "text": f"{context}\nChart: {shown}\n{md}", "title": shown,
                           "content": {"header": header, "rows": rows, "x_axis": header[0] if header else "",
                                       "unit": unit_of(title) or page_unit},
                           "source": "chart", "confidence": 0.7, "section_path": [x for x in (section, shown) if x]})
        bidx = len(out.blocks) - 1
        if over_time:
            for j in new_cols:
                add_series(header[j], "", title, cols[j], "chart", 0.8, True, bidx)
            continue
        period = context_period(title, section, doc_title)
        if period is None:
            undated("chart", shown)
        for s in category_series([r[0] if r else "" for r in rows], [""] + header[1:], rows, period, group_rows=False):
            add_series(s.name, s.parent, title, s.points, "chart", 0.8, True, bidx)

    # Numbers stated only in commentary become facts too, so the answer can cite them.
    for bidx, kind, para in text_blocks:
        if kind in ("narrative", "takeaway"):
            add_text_series(out, para, section, doc_title, bidx)

    anchor_text(out, section)

    if out.unmatched:
        names = sorted(set(out.unmatched))
        out.issues.append(("unknown_metric", ", ".join(names[:12]),
                           f"{len(names)} labels not in metrics.yaml (stored under their own names)",
                           "add a synonym in metrics.yaml if one of these is a known metric"))
    return out
