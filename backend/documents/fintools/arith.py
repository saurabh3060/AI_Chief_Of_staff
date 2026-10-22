"""Arithmetic over values already in hand: aggregate, calculate_ratio, calculate_change,
compare_values, derive_missing_value. Every result carries the formula it was computed from, so
an answer can show its working and a verifier can trace each number."""
from ._common import VALUE, clean, fmt, is_rate, item, items, name, num, values_schema
from ._common import pct_change as rel_change
from ._registry import tool

_AVG = ("avg", "average", "mean")


@tool("aggregate",
      "Combine several values into one: sum, avg, min, max, count, median or weighted_avg. "
      "Use for totals over periods or rows, averages, and the smallest/largest value.",
      {"values": values_schema("The values to combine."),
       "op": {"type": "string", "enum": ["sum", "avg", "min", "max", "count", "median", "weighted_avg"]},
       "weights": {"type": "array", "items": {"type": "number"},
                   "description": "weighted_avg only: one weight per value (e.g. card counts)."}},
      required=("values", "op"))
def aggregate(values, op: str = "sum", weights=None) -> dict:
    its = items(values)
    if not its:
        raise ValueError("values is empty")
    op = (op or "sum").lower().replace(" ", "_")
    vs = [it["value"] for it in its]
    out = {"op": op, "n": len(its)}
    warnings = []
    if op == "sum":
        out["result"] = sum(vs)
        out["formula"] = " + ".join(fmt(v) for v in vs)
        if is_rate(its):
            warnings.append("these are percentages; a sum of rates is rarely meaningful — "
                            "consider avg or weighted_avg")
    elif op in _AVG:
        out["op"] = "avg"
        out["result"] = sum(vs) / len(vs)
        out["formula"] = f"({' + '.join(fmt(v) for v in vs)}) / {len(vs)}"
    elif op in ("min", "max"):
        pick = (min if op == "min" else max)(its, key=lambda it: it["value"])
        out["result"] = pick["value"]
        out["at"] = [name(it) for it in its if it["value"] == pick["value"]]
    elif op == "count":
        out["result"] = len(its)
    elif op == "median":
        s = sorted(vs)
        mid = len(s) // 2
        out["result"] = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    elif op in ("weighted_avg", "weighted_average", "wavg"):
        out["op"] = "weighted_avg"
        ws = [num(w) for w in (weights or [])]
        if len(ws) != len(vs):
            raise ValueError(f"weighted_avg needs one weight per value ({len(vs)} values, {len(ws)} weights)")
        if sum(ws) == 0:
            raise ValueError("weights sum to zero")
        out["result"] = sum(v * w for v, w in zip(vs, ws)) / sum(ws)
        out["formula"] = f"({' + '.join(f'{fmt(v)}*{fmt(w)}' for v, w in zip(vs, ws))}) / {fmt(sum(ws))}"
    else:
        raise ValueError(f"unknown op {op!r}")
    kinds = {it["period_key"].split(":")[0] for it in its if it.get("period_key")}
    if len(kinds) > 1 and op in ("sum", "avg", "weighted_avg"):
        warnings.append(f"values mix period kinds {sorted(kinds)} (e.g. months with a quarter)")
    out["result"] = clean(out["result"])
    out["inputs"] = [{"name": name(it), "value": clean(it["value"])} for it in its]
    if warnings:
        out["warnings"] = warnings
    return out


_SCALES = {"percent": 100.0, "ratio": 1.0, "bps": 10000.0, "multiple": 1.0}


@tool("calculate_ratio",
      "Divide A by B: a ratio, percentage, margin, rate or share (e.g. cost/income %, spend per card). "
      "Give a list as numerator and no denominator to get each item's share of their total (a mix).",
      {"numerator": {"description": "One value, or a list of values (each is divided).",
                     "anyOf": [VALUE, {"type": "number"}, {"type": "array", "items": VALUE}]},
       "denominator": {"description": "One value, or a list that is summed first. Omit for share of total.",
                       "anyOf": [VALUE, {"type": "number"}, {"type": "array", "items": VALUE}]},
       "scale": {"type": "string", "enum": list(_SCALES), "description": "percent (x100, default), "
                 "ratio (plain A/B), bps (x10000) or multiple (A/B as 'x times')."}},
      required=("numerator",))
def calculate_ratio(numerator, denominator=None, scale: str = "percent") -> dict:
    scale = (scale or "percent").lower()
    if scale not in _SCALES:
        raise ValueError(f"unknown scale {scale!r}")
    k = _SCALES[scale]
    nums = items(numerator)
    if not nums:
        raise ValueError("numerator is empty")
    if denominator is None or denominator == []:
        if len(nums) < 2:
            raise ValueError("give a denominator, or a list of numerators to get shares of their total")
        d, d_name = sum(it["value"] for it in nums), "total of the items"
    else:
        dens = items(denominator)
        d = sum(it["value"] for it in dens)
        d_name = name(dens[0]) if len(dens) == 1 else " + ".join(name(it) for it in dens)
    mult = f" * {int(k)}" if k != 1 else ""
    rows = [{"name": name(it), "numerator": clean(it["value"]),
             "result": clean(None if d == 0 else it["value"] / d * k),
             "formula": f"{fmt(it['value'])} / {fmt(d)}{mult}"} for it in nums]
    out = {"scale": scale, "denominator": clean(d), "denominator_name": d_name}
    if d == 0:
        out["warnings"] = ["denominator is zero"]
    if len(rows) == 1:
        out.update(result=rows[0]["result"], numerator_name=rows[0]["name"],
                   numerator=rows[0]["numerator"], formula=rows[0]["formula"])
    else:
        out["results"] = rows
    return out


def _change(old: dict, new: dict, rate: bool) -> dict:
    o, n = old["value"], new["value"]
    d = n - o
    out = {"from": name(old), "to": name(new), "old": clean(o), "new": clean(n),
           "abs_change": clean(d), "pct_change": clean(rel_change(o, n)),
           "direction": "up" if d > 0 else "down" if d < 0 else "flat",
           "formula": f"{fmt(n)} - {fmt(o)}"}
    if rate:
        # A rate moving 12% -> 15% is +3 percentage points (+300 bps), and +25% relative.
        out["pp_change"] = clean(d)
        out["bps_change"] = clean(d * 100)
    if o < 0:
        out["note"] = "pct_change is relative to the absolute size of a negative starting value"
    return out


@tool("calculate_change",
      "Change between two values: absolute change, % change and, for rates, percentage-point / bps "
      "change (MoM, QoQ, YoY growth, mix shift). Give old+new, or an ordered list to get every "
      "period-over-period change.",
      {"old": {"description": "Earlier / base value.", "anyOf": [VALUE, {"type": "number"}]},
       "new": {"description": "Later / compared value.", "anyOf": [VALUE, {"type": "number"}]},
       "values": values_schema("Instead of old/new: values in time order; each step is compared."),
       "kind": {"type": "string", "enum": ["amount", "count", "percent"],
                "description": "percent makes the change read in percentage points. Inferred if omitted."}})
def calculate_change(old=None, new=None, values=None, kind: str | None = None) -> dict:
    if values:
        its = items(values)
        if len(its) < 2:
            raise ValueError("values needs at least two entries")
        rate = is_rate(its, kind)
        steps = [_change(a, b, rate) for a, b in zip(its, its[1:])]
        out = {"steps": steps, "overall": _change(its[0], its[-1], rate)}
        return out
    if old is None or new is None:
        raise ValueError("give old and new, or values")
    o, n = item(old, 0), item(new, 1)
    return _change(o, n, is_rate([o, n], kind))


@tool("compare_values",
      "Compare two or more values against a baseline: difference, % difference, ratio, which is "
      "higher (actual vs plan, scenario A vs B, clean vs total, this period vs last).",
      {"values": values_schema("The values to compare (at least two)."),
       "baseline": {"type": "string", "description": "Label (or 0-based position) of the value the "
                    "others are measured against. Default: the second of two values (A vs B means "
                    "A relative to B), otherwise the first."}},
      required=("values",))
def compare_values(values, baseline=None) -> dict:
    its = items(values)
    if len(its) < 2:
        raise ValueError("compare_values needs at least two values")
    b = _pick(its, baseline, default=1 if len(its) == 2 else 0)
    base = its[b]
    rate = is_rate(its)
    comps = []
    for i, it in enumerate(its):
        if i == b:
            continue
        d = it["value"] - base["value"]
        c = {"name": name(it), "value": clean(it["value"]), "diff": clean(d),
             "pct_diff": clean(rel_change(base["value"], it["value"])),
             "ratio": clean(None if base["value"] == 0 else it["value"] / base["value"]),
             "vs_baseline": "higher" if d > 0 else "lower" if d < 0 else "equal",
             "formula": f"{fmt(it['value'])} - {fmt(base['value'])}"}
        if rate:
            c["pp_diff"] = clean(d)
        comps.append(c)
    hi = max(its, key=lambda it: it["value"])
    lo = min(its, key=lambda it: it["value"])
    return {"baseline": {"name": name(base), "value": clean(base["value"])}, "comparisons": comps,
            "highest": name(hi), "lowest": name(lo), "spread": clean(hi["value"] - lo["value"])}


def _pick(its: list[dict], ref, default: int) -> int:
    if ref is None or ref == "":
        return default
    if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
        i = int(ref)
        if not 0 <= i < len(its):
            raise ValueError(f"baseline position {i} out of range")
        return i
    ref_l = str(ref).lower()
    for i, it in enumerate(its):
        if ref_l in (it["label"].lower(), str(it.get("period", "")).lower(), name(it).lower()):
            return i
    for i, it in enumerate(its):
        if ref_l in name(it).lower():
            return i
    raise ValueError(f"baseline {ref!r} is not one of {[name(it) for it in its]}")


@tool("derive_missing_value",
      "Back-solve one unknown value algebraically. relation=sum: total minus the known parts "
      "(FY - Q1 - Q2 - Q3 = implied Q4). average: the missing member of an average. ratio: any one of "
      "numerator/denominator/ratio from the other two. growth: any one of old/new/pct_change.",
      {"relation": {"type": "string", "enum": ["sum", "average", "ratio", "growth"]},
       "total": {"description": "sum: the total.", "anyOf": [VALUE, {"type": "number"}]},
       "known": values_schema("sum/average: the known parts."),
       "average": {"type": "number", "description": "average: the stated average."},
       "count": {"type": "integer", "description": "average: how many members in all (default known + 1)."},
       "numerator": {"type": "number"}, "denominator": {"type": "number"},
       "ratio": {"type": "number", "description": "ratio: in the unit set by scale (percent default)."},
       "scale": {"type": "string", "enum": list(_SCALES)},
       "old": {"type": "number"}, "new": {"type": "number"},
       "pct_change": {"type": "number", "description": "growth: e.g. 12.5 for +12.5%."}},
      required=("relation",))
def derive_missing_value(relation: str = "sum", total=None, known=None, average=None, count=None,
                         numerator=None, denominator=None, ratio=None, scale: str = "percent",
                         old=None, new=None, pct_change=None) -> dict:
    relation = (relation or "sum").lower()
    if relation == "sum":
        if total is None:
            raise ValueError("sum needs total and known")
        t, ks = item(total), items(known)
        v = t["value"] - sum(k["value"] for k in ks)
        return {"relation": "sum", "missing": clean(v),
                "formula": " - ".join([fmt(t["value"]), *(fmt(k["value"]) for k in ks)]),
                "total": clean(t["value"]), "known": [{"name": name(k), "value": clean(k["value"])} for k in ks]}
    if relation in ("average", "avg", "mean"):
        ks = items(known)
        if average is None:
            raise ValueError("average needs average and known")
        n = int(count) if count else len(ks) + 1
        if n <= len(ks):
            raise ValueError(f"count {n} leaves no missing member among {len(ks)} known")
        if n - len(ks) != 1:
            raise ValueError("only one member may be missing")
        v = num(average) * n - sum(k["value"] for k in ks)
        return {"relation": "average", "missing": clean(v),
                "formula": f"{fmt(num(average))} * {n} - ({' + '.join(fmt(k['value']) for k in ks)})"}
    if relation == "ratio":
        k = _SCALES.get((scale or "percent").lower())
        if k is None:
            raise ValueError(f"unknown scale {scale!r}")
        given = {x: y for x, y in (("numerator", numerator), ("denominator", denominator), ("ratio", ratio))
                 if y is not None}
        if len(given) != 2:
            raise ValueError("ratio needs exactly two of numerator, denominator, ratio")
        if "ratio" not in given:
            d = num(denominator)
            return {"relation": "ratio", "solved": "ratio", "missing": clean(None if d == 0 else num(numerator) / d * k),
                    "formula": f"{fmt(num(numerator))} / {fmt(d)}" + (f" * {int(k)}" if k != 1 else "")}
        r = num(ratio) / k
        if "numerator" not in given:
            return {"relation": "ratio", "solved": "numerator", "missing": clean(r * num(denominator)),
                    "formula": f"{fmt(num(ratio))} / {int(k)} * {fmt(num(denominator))}"}
        return {"relation": "ratio", "solved": "denominator", "missing": clean(None if r == 0 else num(numerator) / r),
                "formula": f"{fmt(num(numerator))} / ({fmt(num(ratio))} / {int(k)})"}
    if relation == "growth":
        given = {x: y for x, y in (("old", old), ("new", new), ("pct_change", pct_change)) if y is not None}
        if len(given) != 2:
            raise ValueError("growth needs exactly two of old, new, pct_change")
        if "pct_change" not in given:
            o, n = num(old), num(new)
            return {"relation": "growth", "solved": "pct_change", "missing": clean(rel_change(o, n)),
                    "formula": f"({fmt(n)} - {fmt(o)}) / {fmt(abs(o))} * 100"}
        g = num(pct_change) / 100
        if "new" not in given:
            o = num(old)
            return {"relation": "growth", "solved": "new", "missing": clean(o + abs(o) * g),
                    "formula": f"{fmt(o)} + {fmt(abs(o))} * {fmt(num(pct_change))} / 100"}
        n = num(new)
        if g <= -1:
            raise ValueError("pct_change of -100% or less has no unique old value")
        return {"relation": "growth", "solved": "old", "missing": clean(n / (1 + g)),
                "formula": f"{fmt(n)} / (1 + {fmt(num(pct_change))} / 100)",
                "note": "assumes a positive old value"}
    raise ValueError(f"unknown relation {relation!r}")
