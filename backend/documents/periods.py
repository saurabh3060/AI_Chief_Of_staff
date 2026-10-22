"""Reporting periods: parse how Indian financial documents write them into canonical keys.

Keys sort and compare across documents:
  M:2025-06     month (Jun-25, June 2025, Jun'25)
  Q:FY26-Q1     fiscal quarter (Q1FY26, Q1 FY2025-26, FY26 Q1)
  H:FY26-H1     fiscal half (H1 FY26, H1FY26)
  9M:FY26       nine months (9MFY26, 9M FY26)
  FY:FY26       fiscal year, April 2025 to March 2026 (FY26, FY2025-26, FY 2026)
  D:2025-09-30  point in time (30.09.2025, 30-Sep-25, Sep 30, 2025, 30th September 2025)
The Indian fiscal year runs April to March and is named after the year it ends in.
"""
import calendar
import re
from dataclasses import dataclass
from datetime import date

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTHS["sept"] = 9
_MON = (r"(january|february|march|april|may|june|july|august|september|october|november|december"
        r"|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)(?![a-z])\.?")
_YR = r"(\d{4}|\d{2})"
_SEP = r"[\s\-'’‘`/,.]*"

PARTIAL_RE = re.compile(r"\btill\b|\bupto\b|\bup to\b|\bqtd\b|\bytd\b|\bmtd\b|\bso far\b", re.I)


@dataclass(frozen=True)
class Period:
    key: str
    kind: str  # M, Q, H, 9M, FY, D
    start: date
    end: date
    partial: bool = False

    @property
    def label(self) -> str:
        """Short display form: Jun-25, Q1FY26, H1FY26, 9MFY26, FY26, 30-Sep-25."""
        if self.kind == "M":
            return f"{calendar.month_abbr[self.start.month]}-{self.start.year % 100:02d}"
        if self.kind == "D":
            return f"{self.end.day:02d}-{calendar.month_abbr[self.end.month]}-{self.end.year % 100:02d}"
        fy = self.key.split(":")[1].split("-")[0]
        suffix = {"Q": self.key[-2:], "H": self.key[-2:], "9M": "9M", "FY": ""}[self.kind]
        return f"{suffix}{fy}" + (" (partial)" if self.partial else "")


def _year(s: str) -> int:
    y = int(s)
    return y + 2000 if y < 100 else y


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def fiscal_year(d: date) -> int:
    """Calendar year the fiscal year containing d ends in (April starts a new one)."""
    return d.year + 1 if d.month >= 4 else d.year


def month(y: int, m: int) -> Period:
    return Period(f"M:{y}-{m:02d}", "M", date(y, m, 1), _month_end(y, m))


def fy(end_year: int) -> Period:
    return Period(f"FY:FY{end_year % 100:02d}", "FY", date(end_year - 1, 4, 1), date(end_year, 3, 31))


def quarter(end_year: int, q: int, partial: bool = False) -> Period:
    m0 = 4 + 3 * (q - 1)  # Q1 = Apr-Jun
    y0 = end_year - 1 if m0 <= 12 else end_year
    m0 = m0 if m0 <= 12 else m0 - 12
    start = date(y0, m0, 1)
    m1, y1 = (m0 + 2, y0) if m0 + 2 <= 12 else (m0 + 2 - 12, y0 + 1)
    return Period(f"Q:FY{end_year % 100:02d}-Q{q}", "Q", start, _month_end(y1, m1), partial)


def half(end_year: int, h: int) -> Period:
    if h == 1:
        return Period(f"H:FY{end_year % 100:02d}-H1", "H", date(end_year - 1, 4, 1), date(end_year - 1, 9, 30))
    return Period(f"H:FY{end_year % 100:02d}-H2", "H", date(end_year - 1, 10, 1), date(end_year, 3, 31))


def nine_months(end_year: int) -> Period:
    return Period(f"9M:FY{end_year % 100:02d}", "9M", date(end_year - 1, 4, 1), date(end_year - 1, 12, 31))


def day(y: int, m: int, d: int) -> Period | None:
    try:
        when = date(y, m, d)
    except ValueError:
        return None
    return Period(f"D:{when.isoformat()}", "D", when, when)


# (pattern, builder). Order matters: longer forms first so "Q1FY26" is not read as "FY26".
# Whitespace runs are bounded ({0,5}) so adjacent \s* around an optional separator cannot
# backtrack quadratically on long whitespace (py/polynomial-redos).
_FY = r"fy\s{0,5}'?\s{0,5}(?:(\d{4}|\d{2})\s{0,5}[-/]\s{0,5}(\d{2}|\d{4})|(\d{4}|\d{2}))"


def _fy_end(m: re.Match, first: int) -> int:
    """End year from an FY group: FY2025-26 -> 2026, FY26 -> 2026, FY2026 -> 2026."""
    a, b, c = m.group(first), m.group(first + 1), m.group(first + 2)
    return _year(b) if a else _year(c)


_PATTERNS: list[tuple[re.Pattern, callable]] = [
    (re.compile(rf"\bq\s{{0,5}}([1-4])\s{{0,5}}[-']?\s{{0,5}}{_FY}", re.I),
     lambda m: quarter(_fy_end(m, 2), int(m.group(1)))),
    (re.compile(rf"\b{_FY}\s{{0,5}}[-/]?\s{{0,5}}q\s{{0,5}}([1-4])\b", re.I),
     lambda m: quarter(_fy_end(m, 1), int(m.group(4)))),
    (re.compile(rf"\bh\s{{0,5}}([12])\s{{0,5}}[-']?\s{{0,5}}{_FY}", re.I),
     lambda m: half(_fy_end(m, 2), int(m.group(1)))),
    (re.compile(rf"\b9\s{{0,5}}m\s{{0,5}}[-']?\s{{0,5}}{_FY}", re.I),
     lambda m: nine_months(_fy_end(m, 1))),
    # 30.09.2025, 30/09/2025, 30-09-25 (day first, as written in India)
    (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b"),
     lambda m: day(_year(m.group(3)), int(m.group(2)), int(m.group(1)))),
    # 30-Sep-25, 30th September 2025, 30 Sep'25
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-]*{_MON}{_SEP}{_YR}\b", re.I),
     lambda m: day(_year(m.group(3)), MONTHS[m.group(2).lower()[:3]], int(m.group(1)))),
    # Sep 30, 2025
    (re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?,\s*(\d{{4}})\b", re.I),
     lambda m: day(int(m.group(3)), MONTHS[m.group(1).lower()[:3]], int(m.group(2)))),
    (re.compile(rf"\b{_MON}{_SEP}{_YR}\b", re.I),
     lambda m: month(_year(m.group(2)), MONTHS[m.group(1).lower()[:3]])),
    (re.compile(rf"\b{_FY}", re.I),
     lambda m: fy(_fy_end(m, 1))),
]


def find_all(text: str) -> list[tuple[Period, tuple[int, int]]]:
    """Every period written in text, with its character span, left to right."""
    found, taken = [], []
    # "Q1_FY26", "FY_25": spreadsheet exports join words with underscores; "FY’25": slides use
    # typographic apostrophes. Same length, so the spans still index the original text.
    text = (text or "").replace("_", " ").replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    for pat, build in _PATTERNS:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            p = build(m)
            if p is None:
                continue
            taken.append(m.span())
            found.append((p, m.span()))
    found.sort(key=lambda x: x[1][0])
    return found


def parse(text: str) -> Period | None:
    """A cell or header that is one period, such as 'Jun-25' or 'Q4FY26 (till Feb)'; else None.
    Text around the period may only be a short qualifier, not a metric name."""
    text = (text or "").strip()
    hits = find_all(text)
    if len(hits) != 1:
        return None
    p, (s, e) = hits[0]
    rest = re.sub(r"[\s()\[\]:,*]+", " ", text[:s] + " " + text[e:]).strip()
    partial = bool(PARTIAL_RE.search(rest))
    # Management tables commonly suffix period headers with scenario markers: A=actual,
    # P=projection and B=budget (for example "Julʼ26 P" or "Q2 FY27 B").
    qualifier = r"(?:month|period|quarter|year|as on|as at|ended|ending|a|p|b|actual|projection|projected|budget)?"
    if rest and not partial and not re.fullmatch(qualifier, rest, re.I):
        return None
    if partial:
        p = Period(p.key, p.kind, p.start, p.end, True)
    return p


def months_without_year(text: str) -> list[int]:
    """Months named in a question without a year ("in December"), outside full periods."""
    spans = [s for _, s in find_all(text)]
    out = []
    for m in re.finditer(rf"\b{_MON}", text or "", re.I):
        if any(s <= m.start() < e for s, e in spans):
            continue
        word = m.group(1).lower()
        if word in ("may",) and not re.search(r"\bin may\b|\bmay\s+(?:month|data)", text, re.I):
            continue  # "may" is usually the verb
        out.append(MONTHS[word[:3]])
    return out
