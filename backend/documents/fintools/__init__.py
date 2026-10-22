"""Generic financial-analysis tools: code does the arithmetic, a model only chooses what to call.

  #  tool                    module      purpose
  1  get_data                data        values for metric(s) x dimensions x periods from library.db
  2  aggregate               arith       sum, avg, min, max, count, median, weighted_avg
  3  calculate_ratio         arith       A / B as percent / ratio / bps / multiple; shares of a total
  4  calculate_change        arith       absolute, % and percentage-point change; step-by-step
  5  compare_values          arith       values vs a baseline: diff, % diff, ratio, higher/lower
  6  calculate_contribution  analysis    each part's share of a total or of a total's change
  7  derive_missing_value    arith       back-solve one unknown (sum, average, ratio, growth)
  8  analyze_series          analysis    peak, trough, monotonicity, runs, overall change, CAGR
  9  rank_values             analysis    order, top/bottom N, ties
 10  reconcile_bridge        analysis    opening + movements = closing, with residual

Use from code:
    import fintools
    rows = fintools.get_data(["PBT"], periods="FY25-FY29")["rows"]
    fintools.analyze_series(rows)

Use from a tool-calling loop:
    params["tools"] = fintools.specs()            # or specs(["get_data", "aggregate"])
    result = fintools.dispatch(name, args, con)   # never raises; errors come back as {"error": ...}
    seen |= fintools.numbers_in(result)           # every number a tool produced, for answer checks

Every tool accepts values as numbers, printed strings ("1,272", "(40.4)", "3.5%") or item dicts
such as get_data rows, and returns plain JSON-able dicts that carry the formula they used.

Adding a tool: write a function in any module here, decorate it with @tool(name, description,
properties, required=...) from ._registry, and import the module below.
"""
from ._registry import TOOLS, Tool, dispatch, tool  # noqa: F401
from . import arith, analysis, data, metric  # noqa: F401  (importing registers the tools)
from .metric import analyze_metric  # noqa: F401
from .analysis import analyze_series, calculate_contribution, rank_values, reconcile_bridge  # noqa: F401
from .arith import aggregate, calculate_change, calculate_ratio, compare_values, derive_missing_value  # noqa: F401
from .data import get_data, parse_periods  # noqa: F401


def specs(names=None) -> list[dict]:
    """Tool-calling specs, all or the named subset (a small model chooses better from fewer)."""
    return [t.spec for n, t in TOOLS.items() if names is None or n in names]


def numbers_in(result) -> set[float]:
    """Every number anywhere in a tool result (nested dicts and lists included), rounded to 6
    places — what an answer checker can accept as produced by a tool."""
    out: set[float] = set()

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            out.add(round(float(x), 6))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(result)
    return out
