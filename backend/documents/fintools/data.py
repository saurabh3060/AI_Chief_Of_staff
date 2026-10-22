"""get_data: fetch metric values by name, dimension and period from PostgreSQL.

Rows come back in the item shape every other tool accepts, so a question like "total collections
cost for Q1" is get_data -> aggregate, with no number passing through the model.
"""
import re
from dataclasses import dataclass
from datetime import timedelta

from .. import periods
from ..periods import Period

from ._registry import tool

MAX_LABELS = 12
MAX_ROWS = 400

_RANGE_SEP = re.compile(r"\s*(?:-|–|—|to|through|thru|till|until)\s*", re.I)
_SPLIT = re.compile(r"\s*(?:,|;|\band\b|&|\bvs\.?\b|\bversus\b)\s*", re.I)
_FY_RANGE = re.compile(r"\bfy\s*'?(\d{2})\s*(?:-|–|—|to)\s*(?:fy\s*'?)?(\d{2})\b", re.I)


@dataclass
class PeriodSpec:
    text: str
    lo: Period | None = None
    hi: Period | None = None
    months: frozenset[int] | None = None

    def exact(self, r: dict) -> bool:
        if self.months is not None:
            return r["period_kind"] == "M" and int(r["period_start"][5:7]) in self.months
        return (r["period_kind"] == self.lo.kind and r["period_start"] >= self.lo.start.isoformat()
                and r["period_end"] <= self.hi.end.isoformat())

    def within(self, r: dict) -> bool:
        if self.months is not None or not r["period_start"]:
            return False
        return (r["period_start"] >= self.lo.start.isoformat() and r["period_end"] <= self.hi.end.isoformat()
                and r["period_kind"] != self.lo.kind)

    def expected(self) -> list[tuple[str, str]]:
        if self.months is not None:
            return []
        out, p = [], self.lo
        while p is not None and p.start <= self.hi.start and len(out) < 120:
            out.append((p.key, p.label))
            p = _next(p)
        return out


def _next(p: Period) -> Period | None:
    d = p.end + timedelta(days=1)
    fy = periods.fiscal_year(d)
    if p.kind == "M":
        return periods.month(d.year, d.month)
    if p.kind == "Q":
        return periods.quarter(fy, (d.month - 4) % 12 // 3 + 1)
    if p.kind == "H":
        return periods.half(fy, 1 if 4 <= d.month <= 9 else 2)
    if p.kind == "FY":
        return periods.fy(fy)
    return None


def parse_periods(spec) -> list[PeriodSpec]:
    if spec is None or spec == "" or spec == []:
        return []
    chunks = spec if isinstance(spec, (list, tuple)) else [spec]
    out = []
    for chunk in chunks:
        chunk = str(chunk).strip()
        one = _parse_one(chunk)
        if one:
            out.append(one)
            continue
        for part in filter(None, _SPLIT.split(chunk)):
            one = _parse_one(part)
            if not one:
                raise ValueError(f"can't read the period {part!r} (try e.g. 'FY26', 'Q1FY26', 'Jun-25', "
                                 f"'FY25-FY29', 'Jan-25 to Mar-25')")
            out.append(one)
    return out


def _parse_one(text: str) -> PeriodSpec | None:
    m = _FY_RANGE.fullmatch(text.strip())
    if m and int(m.group(2)) > int(m.group(1)):
        return PeriodSpec(text, periods.fy(2000 + int(m.group(1))), periods.fy(2000 + int(m.group(2))))
    hits = periods.find_all(text)
    rest = text
    for _, (s, e) in reversed(hits):
        rest = rest[:s] + " " + rest[e:]
    if len(hits) == 1 and not rest.strip(" ()"):
        p = hits[0][0]
        return PeriodSpec(text, p, p)
    if len(hits) == 2:
        (a, (_, a_end)), (b, (b_start, _)) = hits
        if a.kind == b.kind and _RANGE_SEP.fullmatch(text[a_end:b_start]) is not None:
            a, b = sorted((a, b), key=lambda p: p.start)
            return PeriodSpec(text, a, b)
    if not hits:
        months = periods.months_without_year(text)
        if len(months) == 1 and not re.sub(r"(?i)[a-z]+\.?", "", text).strip():
            return PeriodSpec(text, months=frozenset(months))
        if len(months) == 2 and re.search(r"–|—|-|\bto\b|\bthrough\b|\btill\b", text, re.I):
            a, b = months
            span = range(a, b + 1) if a <= b else [*range(a, 13), *range(1, b + 1)]
            return PeriodSpec(text, months=frozenset(span))
    return None


# ---------- metric matching ----------

def _norm_words(s: str) -> list[str]:
    from .. import facts
    return facts.norm_label(s).replace("/", " ").split()


def _word_hit(q: str, words: list[str]) -> bool:
    return any(w == q or (len(q) >= 3 and (w.startswith(q) or q.startswith(w) and len(w) >= 3)) for w in words)


async def _match_series(conn, metric: str, dimensions: list[str], doc_ids):
    """Series for one metric name.  Returns (series rows, candidates when too many labels tie)."""
    from .. import facts
    if doc_ids:
        rows = await conn.fetch(
            "SELECT series_id, label, metric_id, entity, path FROM series "
            "WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )
    else:
        rows = await conn.fetch(
            "SELECT series_id, label, metric_id, entity, path FROM series"
        )
    rows = [dict(r) for r in rows]
    for dim in dimensions:
        dw = _norm_words(dim)
        rows = [r for r in rows
                if all(_word_hit(w, _norm_words(f"{r['label']} {r['entity'] or ''} {r['path'] or ''}")) for w in dw)]

    mid = metric if any(r["metric_id"] == metric for r in rows) else facts.match_metric(metric)
    if mid:
        by_id = [r for r in rows if r["metric_id"] == mid]
        if by_id:
            return by_id, []

    q = _norm_words(metric)
    if not q:
        return [], []
    phrase = " ".join(q)
    scores: dict[str, float] = {}
    for r in rows:
        if r["label"] in scores:
            continue
        lw = _norm_words(r["label"])
        hit = sum(_word_hit(w, lw) for w in q)
        if hit * 2 < len(q) or hit == 0:
            continue
        scores[r["label"]] = hit + (10 if phrase in " ".join(lw) else 0) - 0.001 * len(lw)
    if not scores:
        return [], []
    top = max(scores.values())
    tied = [lab for lab, s in scores.items() if s >= top - 0.05]
    if len(tied) > MAX_LABELS:
        return [], sorted(tied, key=len)[:MAX_LABELS]
    return [r for r in rows if r["label"] in tied], []


@tool("get_data",
      "Look up reported values for one or more metrics, optionally narrowed by dimension (segment, "
      "product, entity, bucket) and period. Returns rows {label, period, value, as_printed, page}; "
      "pass them straight to aggregate, calculate_change, analyze_series etc.",
      {"metrics": {"type": "array", "items": {"type": "string"},
                   "description": "Metric names as the report prints them."},
       "periods": {"type": "string", "description": "Optional period or range."},
       "dimensions": {"type": "array", "items": {"type": "string"},
                      "description": "Optional dimension words."}},
      required=("metrics",), needs_con=True)
async def get_data(metrics, periods=None, dimensions=None, doc_ids=None, con=None) -> dict:
    if isinstance(metrics, str):
        metrics = [metrics]
    if not metrics:
        raise ValueError("metrics is empty")
    if isinstance(dimensions, str):
        dimensions = [dimensions]
    specs = parse_periods(periods)
    if con is None:
        from .. import store
        async with store.acquire() as conn:
            return await _get(conn, list(metrics), specs, list(dimensions or []), doc_ids)
    return await _get(con, list(metrics), specs, list(dimensions or []), doc_ids)


async def _get(conn, metrics: list[str], specs: list[PeriodSpec], dimensions: list[str], doc_ids) -> dict:
    rows, not_found, ambiguous, missing, rolled_up = [], [], {}, {}, []
    for metric in metrics:
        series, candidates = await _match_series(conn, metric, dimensions, doc_ids)
        if candidates:
            ambiguous[metric] = candidates
            continue
        if not series:
            not_found.append(metric)
            continue
        rows_f = await _facts(conn, [s["series_id"] for s in series])
        found: set[str] = set()
        for sid in dict.fromkeys(f["series"] for f in rows_f):
            mine = [f for f in rows_f if f["series"] == sid]
            if not specs:
                rows += mine
                continue
            picked = []
            for sp in specs:
                exact = [f for f in mine if sp.exact(f)]
                finer = [] if exact else [f for f in mine if sp.within(f)]
                if finer:
                    rolled_up.append(f"{mine[0]['label']} (p.{mine[0]['page']}): no {sp.text} value; "
                                     f"returned the {len(finer)} finer periods inside it")
                    found.add(sp.text)
                picked += exact or finer
                found |= {f["period_key"] for f in exact}
            rows += list({f["fact_id"]: f for f in picked}.values())
        gone = [lab for sp in specs if sp.text not in found for key, lab in sp.expected() if key not in found]
        if gone:
            missing[metric] = gone
    out = {"rows": [_public(r) for r in rows[:MAX_ROWS]], "count": len(rows)}
    groups = _groups(rows)
    if len(groups) > 1:
        out["series"] = groups
    if len(rows) > MAX_ROWS:
        out["truncated"] = True
    if not_found:
        out["not_found"] = not_found
    if ambiguous:
        out["ambiguous"] = {m: {"hint": "too many rows match; use a fuller name or add dimensions",
                                "candidates": c} for m, c in ambiguous.items()}
    if missing:
        out["missing_periods"] = missing
    if rolled_up:
        out["notes"] = rolled_up
    return out


async def _facts(conn, series_ids: list[int]) -> list[dict]:
    out = []
    for i in range(0, len(series_ids), 500):
        chunk = series_ids[i:i + 500]
        rows = await conn.fetch(
            """SELECT f.fact_id, s.series_id AS series, s.label, s.metric_id, s.kind, s.unit,
                      s.entity, s.source,
                      f.period_key, f.period_kind, f.period_start, f.period_end,
                      f.period_label AS period, f.partial, f.value, f.as_printed, f.page,
                      d.file_name AS file
                 FROM facts f JOIN series s ON s.series_id = f.series_id
                 LEFT JOIN documents d ON d.doc_id = s.doc_id
                 WHERE s.series_id = ANY($1::bigint[]) AND f.value IS NOT NULL""",
            chunk,
        )
        out += [dict(r) for r in rows]
    out.sort(key=lambda r: (r["label"], r["file"] or "", r["page"] or 0, r["series"],
                            r["period_start"] or ""))
    return out


def _groups(rows: list[dict]) -> list[dict]:
    out: dict[int, dict] = {}
    for r in rows:
        g = out.setdefault(r["series"], {"series": r["series"], "label": r["label"], "page": r["page"],
                                         "source": r["source"], "kind": r["kind"], "periods": []})
        g["periods"].append(r["period"])
    for g in out.values():
        ps = g.pop("periods")
        g["n"] = len(ps)
        g["span"] = f"{ps[0]} to {ps[-1]}" if len(ps) > 1 else ps[0]
        if len(set(ps)) < len(ps):
            g["warning"] = "several values for one period in this series"
    return list(out.values())


def _public(r: dict) -> dict:
    keep = ("label", "period", "value", "as_printed", "kind", "unit", "entity", "metric_id",
            "period_key", "period_start", "partial", "page", "file", "source", "series")
    return {k: r[k] for k in keep if r.get(k) not in (None, "", 0) or k in ("value", "page")}
