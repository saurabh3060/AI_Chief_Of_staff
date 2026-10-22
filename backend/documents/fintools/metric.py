"""analyze_metric: one call that looks a metric up and runs the arithmetic on it in code.

A small model chaining get_data -> analyze_series has to copy every row into the second call,
which is where it drops digits and signs. Here it names only the metric(s), the periods and the
operation; the values go from the store to the calculation without passing through the model,
and every number comes back already formatted the way the report prints it."""
import re

from ._registry import tool
from .analysis import analyze_series, rank_values
from .arith import aggregate, calculate_change, calculate_ratio
from .data import get_data

OPERATIONS = ("value", "change", "trend", "sum", "average", "max", "min", "share", "rank", "rank_latest")
_LAKH = re.compile(r"\d,\d\d,\d{3}")


def _indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *groups, tail]) if head else ",".join([*groups, tail])


class _Fmt:
    """Formats computed numbers like the report prints its inputs: lakh or thousand grouping, the
    inputs' decimal places, and a % sign for rates."""

    def __init__(self, rows: list[dict], lakh: bool = False):
        printed = [str(r.get("as_printed") or "") for r in rows]
        self.lakh = lakh or any(_LAKH.search(p) for p in printed)
        decs = [len(m.group(1)) for p in printed if (m := re.search(r"\.(\d+)", p.replace(",", "")))]
        self.decimals = max(decs, default=0)
        self.rate = bool(rows) and all(r.get("kind") == "percent" or str(r.get("as_printed", "")).endswith("%")
                                       for r in rows)

    def num(self, v, decimals: int | None = None, pct: bool = False) -> str | None:
        if v is None:
            return None
        d = self.decimals if decimals is None else decimals
        sign = "-" if v < 0 else ""
        whole, _, frac = f"{abs(v):.{d}f}".partition(".")
        whole = _indian(int(whole)) if self.lakh else f"{int(whole):,}"
        return f"{sign}{whole}{'.' + frac if frac else ''}{'%' if pct else ''}"

    def value(self, v) -> str | None:
        return self.num(v, pct=self.rate)

    def change(self, v) -> str | None:
        """A change of a rate is in percentage points, not percent."""
        s = self.num(v)
        return None if s is None else s + (" pp" if self.rate else "")


async def _uses_lakh(conn, files: set[str]) -> bool:
    """Whether the reports group digits Indian-style (1,06,532)."""
    if conn is None or not files:
        return False
    # Postgres regex (``~``) for the 1,06,532 pattern — SQLite's GLOB doesn't exist here.
    row = await conn.fetchrow(
        "SELECT 1 FROM facts f JOIN documents d ON d.doc_id = f.doc_id "
        "WHERE d.file_name = ANY($1::text[]) "
        "AND f.as_printed ~ '[0-9],[0-9][0-9],[0-9][0-9][0-9]' LIMIT 1",
        list(files),
    )
    return row is not None


def _short(label: str) -> str:
    return label.split(" › ")[-1]


async def _on_pages(conn, metric: str, pages: list[int], doc_ids) -> str | None:
    """The fullest label of a series on the given pages whose label contains every word of `metric`."""
    from .data import _norm_words, _word_hit
    if conn is None or not pages:
        return None
    q = _norm_words(metric)
    if doc_ids:
        rows = await conn.fetch(
            "SELECT label FROM series WHERE page = ANY($1::int[]) AND doc_id = ANY($2::text[])",
            list(pages), list(doc_ids),
        )
    else:
        rows = await conn.fetch(
            "SELECT label FROM series WHERE page = ANY($1::int[])",
            list(pages),
        )
    labels = [r[0] for r in rows if q and all(_word_hit(w, _norm_words(r[0])) for w in q)]
    best = [lab for lab in labels if all(_word_hit(w, _norm_words(_short(lab))) for w in q)] or labels
    return min(best, key=len) if best else None


_TOTAL = re.compile(r"(?i)^(?:grand\s+)?(?:sub\s*)?total\b")


async def rows_on_pages(conn, pages: list[int], doc_ids=None, limit: int = 60) -> list[tuple[str, str]]:
    if not pages:
        return []
    if doc_ids:
        rows = await conn.fetch(
            "SELECT label FROM series WHERE page = ANY($1::int[]) AND doc_id = ANY($2::text[]) "
            "ORDER BY page, series_id",
            list(pages), list(doc_ids),
        )
    else:
        rows = await conn.fetch(
            "SELECT label FROM series WHERE page = ANY($1::int[]) ORDER BY page, series_id",
            list(pages),
        )
    out = []
    for row in rows:
        label = row["label"]
        parts = label.split(" › ")
        out.append((" › ".join(parts[1:-1]) or parts[0] if len(parts) > 1 else "", parts[-1]))
    return list(dict.fromkeys(out))[:limit]


def is_total(row: str) -> bool:
    return bool(_TOTAL.match(row.strip()))


async def _exact(conn, label: str, periods, doc_ids) -> dict | None:
    from .data import _facts, _public, parse_periods
    if doc_ids:
        rows = await conn.fetch(
            "SELECT series_id FROM series WHERE (label = $1 OR label LIKE $2) "
            "AND doc_id = ANY($3::text[])",
            label, f"% › {label}", list(doc_ids),
        )
    else:
        rows = await conn.fetch(
            "SELECT series_id FROM series WHERE (label = $1 OR label LIKE $2)",
            label, f"% › {label}",
        )
    ids = [r[0] for r in rows]
    if not ids or " › " not in label and len(ids) > 1:
        return None
    facts_rows = await _facts(conn, ids)
    specs = parse_periods(periods)
    if specs:
        facts_rows = [r for r in facts_rows if any(sp.exact(r) for sp in specs)]
    return {"rows": [_public(r) for r in facts_rows]}


def _series_runs(rows: list[dict]) -> list[list[dict]]:
    runs: dict[int, list[dict]] = {}
    for r in rows:
        runs.setdefault(r.get("series", 0), []).append(r)
    for sid, run in runs.items():
        # One kind of period per run: a quarterly row that also carries its full-year total would
        # otherwise read as a jump between Q1 and Q2.
        kinds = [str(r.get("period_key", "")).split(":")[0] for r in run]
        main = max(set(kinds), key=kinds.count)
        runs[sid] = [r for r, k in zip(run, kinds) if k == main]
    for run in runs.values():
        run.sort(key=lambda r: str(r.get("period_start") or r.get("period_key") or ""))
    # Longest run first: the monthly trend, not a one-off mention of the same metric elsewhere.
    return sorted(runs.values(), key=len, reverse=True)


def _row(r: dict, f: _Fmt) -> dict:
    return {"label": r["label"], "period": r.get("period"), "value": f.value(r["value"]), "page": r.get("page")}


def _one_metric(run: list[dict], operation: str, f: _Fmt) -> dict:
    base = {"metric": run[0]["label"], "page": run[0].get("page"), "periods": [r.get("period") for r in run]}
    if operation == "value":
        return {**base, "values": [_row(r, f) for r in run]}
    if operation == "change":
        if len(run) < 2:
            raise ValueError(f"{run[0]['label']}: need two periods for a change, found {len(run)}")
        c = calculate_change(old=run[0], new=run[-1])
        out = {**base, "from": {"period": run[0].get("period"), "value": f.value(c["old"])},
               "to": {"period": run[-1].get("period"), "value": f.value(c["new"])},
               "change": f.change(c["abs_change"]), "direction": c["direction"]}
        if not f.rate and c.get("pct_change") is not None:
            out["pct_change"] = f.num(c["pct_change"], 1, pct=True)
        return out
    if operation == "trend":
        s = analyze_series(run)
        out = {**base, "trend": s["trend"],
               "first": {"period": run[0].get("period"), "value": f.value(s["first"]["value"])},
               "last": {"period": run[-1].get("period"), "value": f.value(s["last"]["value"])},
               "peak": {"at": [a.split(" (")[-1].rstrip(")") for a in s["peak"]["at"]], "value": f.value(s["peak"]["value"])},
               "trough": {"at": [a.split(" (")[-1].rstrip(")") for a in s["trough"]["at"]],
                          "value": f.value(s["trough"]["value"])},
               "total_change": f.change(s["total_change"]), "increases": s["increases"],
               "decreases": s["decreases"]}
        steps = [st for st in s["steps"] if st["change"] is not None]
        up = max(steps, key=lambda st: st["change"], default=None)
        down = min(steps, key=lambda st: st["change"], default=None)
        per = lambda x: x.split(" (")[-1].rstrip(")")  # noqa: E731
        if up and up["change"] > 0:
            out["largest_rise"] = {"from": per(up["from"]), "to": per(up["to"]), "change": f.change(up["change"])}
        if down and down["change"] < 0:
            out["largest_fall"] = {"from": per(down["from"]), "to": per(down["to"]), "change": f.change(down["change"])}
        if not f.rate and s.get("total_pct_change") is not None:
            out["total_pct_change"] = f.num(s["total_pct_change"], 1, pct=True)
        return out
    op = {"sum": "sum", "average": "avg", "max": "max", "min": "min"}[operation]
    a = aggregate(run, op)
    out = {**base, "operation": operation, "result": f.value(a["result"]) if op != "avg" else
           f.num(a["result"], max(f.decimals, 0), pct=f.rate)}
    if a.get("at"):
        out["at"] = [x.split(" (")[-1].rstrip(")") for x in a["at"]]
    return out


@tool("analyze_metric",
      "Look up a metric in the reports and compute on it in code. operation: value (the reported "
      "value(s)), change (first vs last period: difference and % change), trend (peak, trough, first, "
      "last, direction), sum / average / max / min over the periods, share (each metric's share of "
      "their total at one period — pass several metrics, e.g. every channel), rank (order several "
      "metrics by value, or by % change when two periods are given); rank_latest always orders the "
      "latest reported values. Every number in the result is "
      "final and formatted; copy it as-is.",
      {"metrics": {"type": "array", "items": {"type": "string"},
                   "description": "Metric or row names as the report prints them, e.g. ['Spends'] or "
                                  "['Retail', 'Corporate'] for share/rank."},
       "operation": {"type": "string", "enum": list(OPERATIONS)},
       "periods": {"type": "string", "description": "Optional: one period 'Dec-25', two to compare "
                   "'Nov-25, Dec-25', or a range 'Jun-25 to Feb-26'. Omit for every period."},
       "last_n": {"type": "integer", "description": "Optional: keep only the latest N periods "
                  "(\"last 6 months\" -> 6)."}},
      required=("metrics", "operation"), needs_con=True)
async def analyze_metric(metrics, operation: str = "value", periods=None, last_n: int | None = None,
                         doc_ids=None, prefer_pages=None, con=None) -> dict:
    """prefer_pages (set by the caller, not the model): pages of the passages retrieved for this
    question. A metric name matching too many rows is narrowed to the one on those pages."""
    if isinstance(metrics, str):
        metrics = [metrics]
    operation = (operation or "value").lower()
    if operation not in OPERATIONS:
        raise ValueError(f"operation must be one of {', '.join(OPERATIONS)}")
    per_metric: list[list[dict]] = []
    problems: dict = {}
    for m in metrics:
        got = None
        if con is not None:
            got = await _exact(con, m, periods, doc_ids)
        if not got:
            got = await get_data([m], periods=periods, doc_ids=doc_ids, con=con)
        if got.get("ambiguous") or (got.get("not_found") and prefer_pages):
            pinned = await _on_pages(con, m, list(prefer_pages or []), doc_ids)
            if pinned:
                got = await get_data([pinned], periods=periods, doc_ids=doc_ids, con=con)
        for k in ("not_found", "ambiguous", "missing_periods"):
            if got.get(k):
                problems.setdefault(k, [] if k == "not_found" else {})
                if k == "not_found":
                    problems[k] += got[k]
                else:
                    problems[k].update(got[k])
        rows = got["rows"]
        if prefer_pages and any(r.get("page") in prefer_pages for r in rows):
            # The same metric often recurs elsewhere (a plan table, a summary); the pages retrieved
            # for this question are the ones it is about.
            rows = [r for r in rows if r.get("page") in prefer_pages]
        runs = _series_runs(rows)
        if runs:
            per_metric.append(runs[0][-last_n:] if last_n else runs[0])
    if not per_metric:
        return {"error": "no values found", **problems}
    all_rows = [r for run in per_metric for r in run]
    lakh = await _uses_lakh(con, {r["file"] for r in all_rows if r.get("file")})
    f = _Fmt(all_rows, lakh=lakh)
    for run in per_metric:
        for r in run:
            r["label"] = _short(r["label"])

    if operation in ("share", "rank", "rank_latest"):
        if len(per_metric) < 2:
            raise ValueError(f"{operation} needs two or more metrics")
        if operation == "rank" and all(len(run) >= 2 for run in per_metric):
            changes = [{"label": run[0]["label"], "value": calculate_change(old=run[0], new=run[-1])["pct_change"]}
                       for run in per_metric]
            ranked = rank_values([c for c in changes if c["value"] is not None])
            return {"operation": "rank by % change", "from": per_metric[0][0].get("period"),
                    "to": per_metric[0][-1].get("period"),
                    "ranked": [{"rank": r["rank"], "metric": r["name"], "pct_change": f.num(r["value"], 1, pct=True)}
                               for r in ranked["ranked"]], **problems}
        latest = [run[-1] for run in per_metric]
        if operation in ("rank", "rank_latest"):
            ranked = rank_values(latest)
            return {"operation": "rank by value", "period": latest[0].get("period"),
                    "ranked": [{"rank": r["rank"], "metric": r["name"].split(" (")[0],
                                "value": f.value(r["value"])} for r in ranked["ranked"]], **problems}
        shares = calculate_ratio(latest)
        return {"operation": "share of total", "period": latest[0].get("period"),
                "total": f.value(shares["denominator"]),
                "shares": [{"metric": s["name"].split(" (")[0], "value": f.value(s["numerator"]),
                            "share": f"{round(s['result'], 1)}%"} for s in shares["results"]], **problems}

    results = [_one_metric(run, operation, f) for run in per_metric]
    out = {"results": results} if len(results) > 1 else results[0]
    return {**out, **problems}
