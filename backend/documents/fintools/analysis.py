"""Structure-aware analysis: calculate_contribution, analyze_series, rank_values, reconcile_bridge."""
from datetime import date

from ._common import (VALUE, clean, decimals, fmt, is_rate, item, items, name, num, pct, pct_change,
                      values_schema)
from ._registry import tool

_START_KEYS = ("start", "old", "from", "opening")
_END_KEYS = ("end", "new", "to", "closing")


def _start_end(x) -> tuple[float, float] | None:
    """A part given as {start, end} (or old/new, or a two-element [start, end])."""
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return num(x[0]), num(x[1])
    if isinstance(x, dict):
        s = next((x[k] for k in _START_KEYS if x.get(k) is not None), None)
        e = next((x[k] for k in _END_KEYS if x.get(k) is not None), None)
        if s is not None and e is not None:
            return num(s), num(e)
    return None


@tool("calculate_contribution",
      "How much each component contributes to a total, or to a total's change. Give parts with a "
      "value for share of a total, or parts with start and end for share of a change "
      "(\"how much of the Opex increase came from Technology?\").",
      {"parts": {"type": "array", "description": "Components: {label, value} for a level, or "
                 "{label, start, end} for a change.",
                 "items": {"type": "object", "properties": {
                     "label": {"type": "string"}, "value": {"type": "number"},
                     "start": {"type": "number"}, "end": {"type": "number"}}}},
       "total": {"description": "Optional reported total ({value} or {start, end}). If it differs "
                 "from the sum of parts, the gap is returned as a residual.",
                 "anyOf": [VALUE, {"type": "number"},
                           {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}}}]}},
      required=("parts",))
def calculate_contribution(parts, total=None) -> dict:
    if not parts:
        raise ValueError("parts is empty")
    labels = [str(p.get("label") or p.get("name") or f"part {i + 1}") if isinstance(p, dict) else f"part {i + 1}"
              for i, p in enumerate(parts)]
    spans = [_start_end(p) for p in parts]
    if all(spans):
        return _change_contribution(labels, spans, total)
    if any(spans):
        raise ValueError("give every part either a value, or start and end — not a mix")
    vals = [num(p) for p in parts]
    s = sum(vals)
    t = num(total) if total is not None else s
    rows = sorted(({"name": n, "value": clean(v), "share_pct": clean(pct(v, t)),
                    "formula": f"{fmt(v)} / {fmt(t)} * 100"} for n, v in zip(labels, vals)),
                  key=lambda r: -abs(r["value"]))
    out = {"mode": "level", "total": clean(t), "sum_of_parts": clean(s), "parts": rows,
           "largest": rows[0]["name"]}
    if total is not None:
        out["residual"] = clean(t - s)
        out["residual_share_pct"] = clean(pct(t - s, t))
    return out


def _change_contribution(labels, spans, total) -> dict:
    changes = [e - s for s, e in spans]
    t_start = sum(s for s, _ in spans)
    t_end = sum(e for _, e in spans)
    if total is not None:
        ts = _start_end(total)
        if ts is None:
            raise ValueError("with start/end parts, total must also have start and end")
        t_start, t_end = ts
    t_change = t_end - t_start
    rows = []
    for n, (s, e), c in zip(labels, spans, changes):
        rows.append({"name": n, "start": clean(s), "end": clean(e), "change": clean(c),
                     "pct_change": clean(pct_change(s, e)),
                     # share of the total's change; can exceed 100% when other parts moved the other way
                     "share_of_change_pct": clean(pct(c, t_change)),
                     # percentage points this part added to the total's growth rate
                     "contribution_pp": clean(pct(c, t_start)),
                     "formula": f"({fmt(e)} - {fmt(s)}) / {fmt(t_change)} * 100"})
    rows.sort(key=lambda r: -abs(r["change"]))
    out = {"mode": "change", "total_start": clean(t_start), "total_end": clean(t_end),
           "total_change": clean(t_change), "total_pct_change": clean(pct_change(t_start, t_end)),
           "parts": rows, "largest_driver": rows[0]["name"]}
    if total is not None:
        out["residual_change"] = clean(t_change - sum(changes))
    if t_change == 0:
        out["warnings"] = ["the total did not change; shares of change are undefined"]
    return out


def _date(it: dict) -> date | None:
    s = it.get("period_start")
    if s:
        return date.fromisoformat(str(s)[:10])
    key = str(it.get("period_key", ""))
    if not key:
        return None
    import periods  # the repo's period parser; imported lazily so pure-arithmetic use stays light
    p = periods.parse(it.get("period") or "")
    return p.start if p and p.key == key else None


def _years_between(a: dict, b: dict) -> float | None:
    da, db = _date(a), _date(b)
    if da and db and da != db:
        return (db - da).days / 365.25
    return None


@tool("analyze_series",
      "Describe a time series: first/last, peak and trough (with periods), each period-over-period "
      "change, whether it rose (or fell) every period, the longest run up/down, overall change and "
      "CAGR. Use for \"did X increase every year?\", \"when did it peak?\", \"what was the trend?\".",
      {"values": values_schema("The series in time order (get_data rows are sorted by period)."),
       "years": {"type": "number", "description": "Years between first and last value for CAGR. "
                 "Inferred from the periods if omitted."}},
      required=("values",))
def analyze_series(values, years: float | None = None) -> dict:
    its = items(values)
    if len(its) < 2:
        raise ValueError("a series needs at least two values")
    if all(it.get("period_key") for it in its):
        its.sort(key=lambda it: str(it.get("period_start") or it["period_key"]))
    rate = is_rate(its)
    vs = [it["value"] for it in its]
    steps = []
    for a, b in zip(its, its[1:]):
        d = b["value"] - a["value"]
        steps.append({"from": name(a), "to": name(b), "change": clean(d),
                      "pct_change": clean(pct_change(a["value"], b["value"])),
                      "direction": "up" if d > 0 else "down" if d < 0 else "flat"})
    dirs = [s["direction"] for s in steps]
    if all(d == "up" for d in dirs):
        trend = "increased every period"
    elif all(d == "down" for d in dirs):
        trend = "decreased every period"
    elif all(d == "flat" for d in dirs):
        trend = "flat"
    elif "down" not in dirs:
        trend = "never decreased (some flat periods)"
    elif "up" not in dirs:
        trend = "never increased (some flat periods)"
    else:
        trend = "mixed"
    peak, trough = max(vs), min(vs)
    out = {"n": len(its), "trend": trend,
           "first": {"name": name(its[0]), "value": clean(vs[0])},
           "last": {"name": name(its[-1]), "value": clean(vs[-1])},
           "peak": {"value": clean(peak), "at": [name(it) for it in its if it["value"] == peak]},
           "trough": {"value": clean(trough), "at": [name(it) for it in its if it["value"] == trough]},
           "total_change": clean(vs[-1] - vs[0]),
           "total_pct_change": clean(pct_change(vs[0], vs[-1])),
           "avg_change_per_period": clean((vs[-1] - vs[0]) / (len(vs) - 1)),
           "increases": dirs.count("up"), "decreases": dirs.count("down"),
           "longest_rise": _longest_run(its, dirs, "up"), "longest_fall": _longest_run(its, dirs, "down"),
           "latest_run": _latest_run(its, dirs), "steps": steps}
    if rate:
        out["note"] = "values are rates: changes are in percentage points"
    keys = [it.get("period_key") or it.get("period") for it in its]
    if all(keys) and len(set(keys)) < len(keys):
        out["warnings"] = ["several values share a period — this looks like more than one series; "
                           "pass the rows of one series (get_data's 'series' field)"]
    y = years if years else _years_between(its[0], its[-1])
    if vs[0] > 0 and vs[-1] > 0:
        if y is None:
            y = len(vs) - 1
            out["cagr_assumes"] = "one year between consecutive values (periods not given)"
        out["cagr_pct"] = clean(((vs[-1] / vs[0]) ** (1 / y) - 1) * 100) if y > 0 else None
        out["cagr_years"] = clean(y)
        out["cagr_formula"] = f"(({fmt(vs[-1])} / {fmt(vs[0])}) ** (1 / {fmt(y)}) - 1) * 100"
    else:
        out["cagr_pct"] = None
        out["cagr_note"] = "CAGR is undefined unless the first and last values are positive"
    return out


def _longest_run(its, dirs, want) -> dict | None:
    best, cur_start, best_span = 0, None, None
    for i, d in enumerate(dirs + [None]):
        if d == want:
            cur_start = i if cur_start is None else cur_start
        elif cur_start is not None:
            n = i - cur_start
            if n > best:
                best, best_span = n, (cur_start, i)
            cur_start = None
    if not best_span:
        return None
    a, b = best_span
    return {"periods": best, "from": name(its[a]), "to": name(its[b])}


def _latest_run(its, dirs) -> dict:
    last = dirs[-1]
    n = 0
    for d in reversed(dirs):
        if d != last:
            break
        n += 1
    return {"direction": last, "periods": n, "since": name(its[len(its) - 1 - n])}


@tool("rank_values",
      "Order values and pick the highest/lowest, top N or bottom N (\"which vintage had the highest "
      "COC?\"). Ties share a rank.",
      {"values": values_schema("The values to rank."),
       "order": {"type": "string", "enum": ["desc", "asc"],
                 "description": "desc (default): largest first. asc: smallest first."},
       "top_n": {"type": "integer", "description": "Return only the first N after ordering."}},
      required=("values",))
def rank_values(values, order: str = "desc", top_n: int | None = None) -> dict:
    its = items(values)
    if not its:
        raise ValueError("values is empty")
    desc = (order or "desc").lower() not in ("asc", "ascending", "lowest", "bottom")
    ranked = sorted(its, key=lambda it: it["value"], reverse=desc)
    rows, prev, rank = [], None, 0
    for i, it in enumerate(ranked):
        if it["value"] != prev:
            rank, prev = i + 1, it["value"]
        rows.append({"rank": rank, "name": name(it), "value": clean(it["value"])})
    ties = len({r["rank"] for r in rows}) < len(rows)
    first = [r["name"] for r in rows if r["rank"] == 1]
    last_rank = rows[-1]["rank"]
    out = {"order": "desc" if desc else "asc", "n": len(rows),
           "first": first, "last": [r["name"] for r in rows if r["rank"] == last_rank],
           "ranked": rows[:top_n] if top_n else rows}
    if ties:
        out["ties"] = True
    return out


_OUT_WORDS = ("-", "out", "outflow", "less", "minus", "subtract", "deduct", "decrease")


@tool("reconcile_bridge",
      "Walk a balance: opening + inflows - outflows = closing (book walk, ENR walk, movement or bridge "
      "analysis). Returns the running balance, the computed closing and any unexplained residual "
      "against the reported closing. Omit closing to derive it; omit opening to derive that instead.",
      {"opening": {"description": "Opening balance.", "anyOf": [VALUE, {"type": "number"}]},
       "movements": {"type": "array", "description": "Each movement with its own sign (negative = "
                     "outflow), or with sign '+'/'-'.",
                     "items": {"type": "object", "properties": {
                         "label": {"type": "string"}, "value": {"type": "number"},
                         "sign": {"type": "string", "enum": ["+", "-"]}}, "required": ["value"]}},
       "inflows": values_schema("Alternative to movements: amounts added."),
       "outflows": values_schema("Alternative to movements: amounts subtracted (give them positive)."),
       "closing": {"description": "Reported closing balance to reconcile against.",
                   "anyOf": [VALUE, {"type": "number"}]},
       "tolerance": {"type": "number", "description": "Allowed |residual|. Default: rounding of the "
                     "printed figures (half a unit in the last printed digit, per figure)."}})
def reconcile_bridge(opening=None, movements=None, inflows=None, outflows=None, closing=None,
                     tolerance: float | None = None) -> dict:
    moves = []
    for i, m in enumerate(movements or []):
        it = item(m, i)
        sign = str(m.get("sign") or m.get("type") or "").lower() if isinstance(m, dict) else ""
        if sign in _OUT_WORDS:
            it["value"] = -abs(it["value"])
        elif sign in ("+", "in", "inflow", "add", "plus", "increase"):
            it["value"] = abs(it["value"])
        moves.append(it)
    moves += [dict(it, value=abs(it["value"])) for it in items(inflows)]
    moves += [dict(it, value=-abs(it["value"])) for it in items(outflows)]
    if not moves:
        raise ValueError("give movements, or inflows/outflows")
    net = sum(m["value"] for m in moves)
    if opening is None and closing is None:
        raise ValueError("give opening, closing or both")
    if opening is None:
        c = item(closing)
        o = c["value"] - net
        return {"derived": "opening", "opening": clean(o), "net_movement": clean(net),
                "closing": clean(c["value"]),
                "formula": f"{fmt(c['value'])} - ({' + '.join(fmt(m['value']) for m in moves)})"}
    o = item(opening)
    bal, steps = o["value"], []
    for m in moves:
        bal += m["value"]
        steps.append({"name": name(m), "amount": clean(m["value"]), "balance": clean(bal)})
    out = {"opening": clean(o["value"]), "steps": steps, "net_movement": clean(net),
           "inflows": clean(sum(m["value"] for m in moves if m["value"] > 0)),
           "outflows": clean(sum(m["value"] for m in moves if m["value"] < 0)),
           "computed_closing": clean(bal),
           "formula": " + ".join([fmt(o["value"]), *(fmt(m["value"]) for m in moves)])}
    if closing is None:
        out["derived"] = "closing"
        return out
    c = item(closing)
    figs = [o, c, *moves]
    tol = tolerance if tolerance is not None else sum(0.5 * 10 ** -decimals(f) for f in figs)
    residual = c["value"] - bal
    out.update(closing=clean(c["value"]), residual=clean(residual), tolerance=clean(tol),
               reconciles=abs(residual) <= tol + 1e-9,
               residual_pct_of_closing=clean(pct(residual, c["value"])))
    return out
