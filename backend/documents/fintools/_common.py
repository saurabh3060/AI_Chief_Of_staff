"""Value handling shared by every tool.

A value can be given as
  - a number: 562.0
  - a printed string: "1,272", "(40.4)", "3.5%", "₹ 1,234 Cr"
  - an item dict, e.g. a get_data row: {"label": ..., "period": ..., "value": ..., "kind": ...}
so the output of get_data feeds straight into any other tool."""
import math
import re

# JSON schema fragments for tool specs. The code also accepts bare numbers and printed strings.
VALUE = {"type": "object", "properties": {"label": {"type": "string"}, "period": {"type": "string"},
                                          "value": {"type": "number"}}, "required": ["value"]}


def values_schema(description: str) -> dict:
    return {"type": "array", "items": VALUE,
            "description": description + " Pass get_data rows as they are, or {label, period, value}."}


_STRIP = re.compile(r"^[₹$]|\s*(?:%|bps|x|cr\.?|crores?|lakhs?|mn|bn)$", re.I)


def num(x) -> float:
    if isinstance(x, bool):
        raise ValueError(f"not a number: {x!r}")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        if x.get("value") is None:
            raise ValueError(f"item has no value: {x}")
        return num(x["value"])
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("−", "-").replace("₹", "").strip()
        s = _STRIP.sub("", s).strip()
        neg = s.startswith("(") and s.endswith(")")  # accounting negative
        try:
            v = float(s.strip("()"))
        except ValueError:
            raise ValueError(f"not a number: {x!r}") from None
        return -v if neg else v
    raise ValueError(f"not a number: {x!r}")


def item(x, i: int = 0) -> dict:
    """Normalizes one value to {"label", "value", and "period"/"period_key"/"kind"/"unit"/"as_printed"
    when known}."""
    if isinstance(x, dict):
        out = {"label": str(x.get("label") or x.get("name") or f"item {i + 1}"), "value": num(x)}
        period = x.get("period") or x.get("period_label")
        if period:
            out["period"] = str(period)
        for k in ("period_key", "period_start", "kind", "unit", "as_printed"):
            if x.get(k):
                out[k] = x[k]
        return out
    return {"label": f"item {i + 1}", "value": num(x)}


def items(xs) -> list[dict]:
    if xs is None:
        return []
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    return [item(x, i) for i, x in enumerate(xs)]


def name(it: dict) -> str:
    return f"{it['label']} ({it['period']})" if it.get("period") else it["label"]


def is_rate(its: list[dict], kind: str | None = None) -> bool:
    """Percent-type values: their differences are percentage points, not percent."""
    if kind:
        return kind.lower() in ("percent", "pct", "%", "rate", "bps")
    return bool(its) and all(it.get("kind") == "percent" or it.get("unit") == "%"
                             or str(it.get("as_printed", "")).endswith("%") for it in its)


def clean(v: float | None) -> float | None:
    """Rounds float noise (0.1 + 0.2) away; None for division by zero / non-finite."""
    if v is None or not math.isfinite(v):
        return None
    r = round(v, 6)
    return int(r) if r == int(r) and abs(r) < 1e15 else r


def pct(part: float, whole: float) -> float | None:
    return None if whole == 0 else part / whole * 100


def pct_change(old: float, new: float) -> float | None:
    """Relative to |old| so a loss narrowing from -50 to -20 reads as +60%, not -60%."""
    return None if old == 0 else (new - old) / abs(old) * 100


def fmt(v: float) -> str:
    """A number as it appears inside a formula string."""
    c = clean(v)
    return "?" if c is None else (f"({c})" if c < 0 else str(c))


def decimals(it: dict) -> int:
    """Decimal places the value was printed with (as_printed when given, else the float itself)."""
    s = str(it.get("as_printed") or repr(float(it["value"])))
    m = re.search(r"\.(\d+)", s.replace(",", ""))
    if not m or (not it.get("as_printed") and set(m.group(1)) == {"0"}):
        return 0
    return len(m.group(1))
