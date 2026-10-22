"""Question answering over the document library: hybrid search for context, then a tool-calling
loop where the chat model looks up exact numbers and combines them with a calculator.

  1. Extract {company, year, quarter, statement} from the question as JSON (grammar-constrained)
     and bake them into facts_view's WHERE clause before anything else — most wrong financial
     answers come from the right metric in the wrong period, and embeddings alone won't fix that.
     A guessed filter is only kept if it doesn't zero out the view (see _make_facts_view).
  2. Embed the question; hybrid-search (vector + FTS5) blocks (passages) and series for context.
  3. Build a prompt: instructions + a facts_view schema + matching metric definitions +
     the retrieved series and passages + the question.
  4. Send to the chat model with two tools: run_metric (a loose label/period lookup against
     facts_view, so it inherits the period/company scope from (1)) and calculate (a restricted
     Python arithmetic evaluator, see calc.py) — a total, share or growth is computed by calling
     run_metric for each row needed, then calculate on the numbers it returned, never SQL.
  5. A tool call is validated and run; its result goes back to the model as the tool result.
  6. Plain text (no tool call) is a candidate answer. Every number in it must trace back to a
     run_metric/calculate result seen this conversation (never from the model reading raw passage
     text) — an answer that fails this is bounced back once for a real lookup. Give up after
     _MAX_ROUNDS rounds total.
"""
import json
import re
import time
from collections.abc import Callable

import asyncio

import asyncpg
import httpx
import yaml

from . import calc
from . import config
from . import llm
from . import generic_metrics
from . import store
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")

METRICS = yaml.safe_load(config.METRICS_YAML.read_text(encoding="utf-8"))

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "run_metric",
            "description": ("Look up a metric's value(s) in facts_view by name, optionally for one "
                             "period. Matches the label loosely (every word in `label` must appear "
                             "somewhere in the row's label or metric_id) — the preferred tool for a "
                             "single number, since you don't need exact spelling. Leave period empty "
                             "to get every period at once for a trend or for feeding calculate."),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string",
                              "description": ("Best: a metric id from 'Relevant metric definitions' "
                                              "(e.g. 'cc_spends'), which is exact. Otherwise a label "
                                              "copied character-for-character from 'Series that "
                                              "exist', or a few words of it (e.g. 'casa ratio'). "
                                              "Don't add words that aren't in the label.")},
                    "period": {"type": "string",
                               "description": "A period label to filter to, e.g. 'Dec-25' or 'Q3FY26'. "
                                              "Leave empty for every period. If a call returns a "
                                              "note listing other periods, retry with one of those."},
                },
                "required": ["label"],
            },
        },
    },
    calc.TOOL_SPEC,
]

SCHEMA_DOC = """facts_view columns:
  file_name, title, doc_type, as_of_label  -- which report
  page                                     -- cite this as (p.6)
  label       -- the metric's printed name
  metric_id   -- canonical id shared across reports when known (may be blank)
  kind        -- amount | count | percent | ratio | bps
  unit, entity
  period_key    -- sorts correctly, e.g. 'M:2025-12', 'Q:FY26-Q3', 'FY:FY26'
  period_kind   -- M | Q | H | 9M | FY | D
  period_label  -- e.g. 'Dec-25', 'Q3FY26', 'FY26'
  partial       -- 1 if the period is not yet complete (e.g. quarter-to-date)
  value         -- the number, as a float
  as_printed    -- the value exactly as printed in the report; quote this in your answer
  source        -- how the value was read: "chart" or "vision" means read off a chart image by a
                   vision model, not the report's printed text. Prefix such a value with "≈"
                   in your answer (e.g. "≈ 1,234"); anything else is printed text, quote it plainly.

kind/unit are often blank — that never means the row is wrong. If a row's label plainly matches
what was asked (e.g. label "Retail" for a question about retail loans), use it; don't
refuse or ask the user to clarify just because kind/unit are blank.

Periods you ask run_metric for match loosely against period_label (e.g. period "FY26" matches
"Q1FY26", "Q2FY26", ...); leave period empty to get every period for that label back in one call.
For a trend, read the rows in period_key order and keep period_kind the same across rows you compare.

Never compute arithmetic yourself — always call calculate:
  total across rows: call run_metric once per row you need (or once with no period for every
                     period of one row), then calculate("100 + 200 + 300") on the values returned.
  each row's share:  calculate("100 * 100.0 / (100 + 200 + 300)") using values you already have.
  growth between two periods: run_metric for each period's value, then
                     calculate("(300 - 200) / 200 * 100") — never estimate a percentage by eye.

Match label with the fullest distinguishing phrase you have, not one generic word — a single word
like "channel" or "cost" can also match an unrelated row elsewhere in the report. Copy the phrase
as it appears in the series list below (e.g. "Deposit Mix", not just "Deposit").

The question's wording is often business jargon, not the report's exact label — translate it to
the term a report would print before you search (e.g. "paydown" of a book may be printed as a
repayment row in a book-walk breakdown, not a row literally called "paydown"). If a question
compares two business concepts, search for each one separately by its report-side term — do not
stop after finding only one side."""

SYSTEM_TEMPLATE = """You are a financial analyst answering questions about reports a user has uploaded.

You have two tools. run_metric looks up a metric's value(s) in facts_view by name, optionally for
one period — it matches loosely, so you don't need exact spelling. calculate evaluates one
arithmetic expression built from numbers you already have. Never compute arithmetic yourself, even
something that looks simple — always call calculate.

Decide internally which kind of question this is (don't announce it, just act on it):
  - a single number already sitting in a report -> run_metric, once
  - a total, difference, percentage change, ratio, share, ranking or trend across rows ->
    run_metric for each row/period you need (or once with no period, for every period of one row),
    then calculate on the values it returned
  - a "why", "what risks", or "how does the company describe..." question with no number needed ->
    answer straight from the passages below and cite the page; no tool call needed
  - a question needing both a number and an explanation -> call a tool for the number and use the
    passages for the explanation, then combine both in one answer with a page citation for each part
  - a follow-up about your own previous reply in this conversation (repeat, rephrase, clarify,
    summarize, "what did you just say") rather than a new financial lookup -> answer it directly
    from the conversation above; no tool call needed, and a lack of matching facts_view/passages
    below is not a reason to refuse this kind of question

{schema}
{filters}
If one of the metric definitions below clearly matches the question, call run_metric with its id
(e.g. "cc_spends") as label on your first call — an id is exact, wording can match several
different rows loosely. Always use a tool to get a specific number — never read one off the raw
passage text below by eye; a wide table there is plain extracted text and its columns are easy to
miscount. When you have the answer, reply in plain text (not a tool call): 2-4 sentences, state
the number(s) using as_printed, and cite the page as (p.6). For a trend across several periods,
summarize the shape once (start, high/low, latest) instead of listing every change, and never
restate a number twice. Every number you write must come from a tool result above, not from memory
or the passages. Use as_printed only for a value run_metric returned as-is from one row — it was
printed in the report exactly like that. A number calculate computed (a sum, difference,
percentage change, ratio or share) is calculated, not printed: report it as what you calculated
(e.g. "revenue grew 25%, calculated from the two figures above"), and never say the report states
a computed figure directly. Do not discuss citation formatting or your process — just answer. For a
new financial lookup, if nothing in facts_view or the passages below answers the question, say so plainly (e.g. "I couldn't
find that in the reports") rather than guessing.

Relevant metric definitions:
{metrics}

Series that exist (labels only — call a tool for their values):
{series}

Passages from the reports (raw extracted text, for narrative questions such as "why" only):
{passages}

THINK STEP BY STEP: work through metric/entity, period, which row(s) you actually need, and which
tool result matches the question's exact scope before answering — internally, not out loud.
"""

SYSTEM_TEMPLATE_INSIGHTS = """Think step by step before answering.
Answer like a data insights expert briefing the CEO.
Once a tool result gives you enough to answer, stop calling tools and answer in plain text."""

_MAX_ROUNDS = 8
_ROW_LIMIT = 50
_RESULT_CHARS = 4000

# --- 1. metadata filters, extracted before any lookup -----------------------------------------

FILTER_SYSTEM = """Extract these fields from the user question as JSON: company (an entity/company
name mentioned, or null), year (the single 4-digit calendar year a stated period implies, or null),
quarter (a quarter Q1-Q4, a half H1/H2, FY, 9M, or a month name/abbreviation stated, or null),
statement (a report or statement type mentioned, e.g. balance sheet, income statement, annual
report, or null). Only extract what is explicitly named or clearly implied by ONE stated period;
use null for a range of periods, a relative period ("last 6 months"), or anything not mentioned.
Never put the question itself, or a business term that is not a company, into a field.

Example: "What was the CASA ratio in Mar-24?" -> {"company": null, "year": 2024, "quarter": "Mar", "statement": null}
Example: "What was ABC Bank's NNPA in Q1FY25?" -> {"company": "ABC Bank", "year": 2024, "quarter": "Q1", "statement": null}
Example: "How did deposits change from Jan-24 to Jun-24?" -> {"company": null, "year": null, "quarter": null, "statement": null}
Example: "What was opex in Apr-24 vs Mar-24?" -> {"company": null, "year": null, "quarter": null, "statement": null}
"""

# A model this small sometimes still picks just one side of a two-period question (e.g. "Dec-25 vs
# Nov-25" -> quarter="Dec"), which would silently filter the other period out of facts_view before
# any lookup runs. Whenever the question itself names two or more distinct periods, the year/quarter
# guess is dropped regardless of what the model returned — a code-level backstop for the same
# "explicitly ONE stated period" rule already asked of the prompt.
_PERIOD_TOKEN_RE = re.compile(
    r"(?i)\b(?:q[1-4]|h[12]|fy|9m)[\s-]?(?:fy)?\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s.-]?\d{2,4}\b")
# A bare-year comparison ("2024 vs 2025") has no quarter/month token for _PERIOD_TOKEN_RE to catch,
# so it needs its own multi-year check.
_YEAR_TOKEN_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# A model this small can also mis-map an explicit month onto a fiscal quarter it doesn't belong to
# (e.g. "February 2026" -> quarter="Q2"), which then filters facts_view to the wrong period instead
# of the one actually asked about — and "keep a filter only if it doesn't zero out the view" doesn't
# catch this, since the wrong quarter is a real period elsewhere in the document. When the question
# names a month explicitly, the guess must be that month or it's discarded.
_MONTH_RE = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b")

FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "quarter": {"type": ["string", "null"]},
        "statement": {"type": ["string", "null"]},
    },
    "required": ["company", "year", "quarter", "statement"],
    "additionalProperties": False,
}

_QUARTER_RE = re.compile(r"(?i)^q([1-4])$")
_HALF_RE = re.compile(r"(?i)^h([12])$")

RESOLVE_SYSTEM = """Rewrite the user's latest question into one that stands alone, using only the
prior exchange below to fill in a company, metric, or period the latest question leaves implicit
(e.g. after "What was HDFC Bank's revenue in FY24?", "what about FY25?" becomes "What was HDFC
Bank's revenue in FY25?"). If the latest question is already self-contained, or isn't a new
financial lookup at all (e.g. it asks you to repeat, rephrase, or clarify your last answer), return
it unchanged, verbatim. Reply with only the question, nothing else — no quotes, no explanation."""


def _looks_like_a_question(text: str) -> bool:
    """RESOLVE_SYSTEM asks for a rewritten question, nothing else — but this model sometimes
    ignores that and answers the follow-up directly instead, fabricating numbers/reasons along
    the way (e.g. "What drove the single worst month?" -> "The single worst month was April 2025,
    driven by a significant drop of -142,857 units..."). That fabricated text then becomes the
    retrieval query, which is worse than just searching on the original unresolved question: it
    steers embedding search and metadata filters toward invented specifics instead of the real
    ones. A genuine rewritten question always ends in "?" and stays question-length; a fabricated
    answer is a longer declarative sentence — both observed failures were 190+ chars ending in a
    period, so this check is deliberately loose (only rules out the answer-shaped case), not a
    strict question-quality filter."""
    text = text.strip()
    return bool(text) and text.endswith("?") and len(text) <= 250


def _resolve_followup(question: str, history: list[dict] | None, model: str, no_cache: bool = False,
                      reasoning: bool = False) -> str:
    """Expands an elliptical follow-up ("what about FY25?") into a self-contained question using the
    prior exchange, so retrieval (filters, metric matching, embedding search) isn't run on a query
    that omits the company/metric/period it actually refers to. The original `question` — not this
    resolved one — is still what's sent as the user's turn to the chat model; this only steers
    retrieval. No-op on the first turn, if the rewrite call fails, or if the rewrite doesn't look
    like a question (see _looks_like_a_question) — in each case, fall back to the original question
    rather than risk retrieval on a fabricated one."""
    turns = [h for h in (history or []) if h.get("role") in ("user", "assistant") and h.get("content")]
    if not turns:
        logger.info("[DOC-QA] _resolve_followup: no history turns, using original question")
        return question
    try:
        logger.info("[DOC-QA] _resolve_followup: resolving with %d history turns, model=%s", len(turns), model)
        t0 = time.perf_counter()
        resolved = llm.chat(model, [
            {"role": "system", "content": RESOLVE_SYSTEM},
            *turns[-2:],
            {"role": "user", "content": question},
        ], temperature=0, max_tokens=1500 if reasoning else 80,
           chat_template_kwargs={"enable_thinking": reasoning},
           **({"cache_prompt": False} if no_cache else {}))
        resolved = _strip_think(resolved).strip('"')
        elapsed = (time.perf_counter() - t0) * 1000
        if resolved and _looks_like_a_question(resolved):
            logger.info("[DOC-QA] _resolve_followup: resolved in %.0fms -> '%s'", elapsed, resolved[:150])
            return resolved
        logger.info("[DOC-QA] _resolve_followup: resolved text not a valid question (%.0fms), using original", elapsed)
        return question
    except Exception:
        logger.warning("[DOC-QA] _resolve_followup: LLM call failed, using original question", exc_info=True)
        return question


def _extract_filters(question: str, model: str, no_cache: bool = False) -> dict:
    """{company, year, quarter, statement}, each None if not stated. Never raises — a bad or
    unparsable extraction just means no filter gets applied, not a broken request. Thinking stays
    off even when the user turns it on: with it on, granite-4.2 reasoned for ~5k chars and never
    emitted the JSON."""
    logger.info("[DOC-QA] _extract_filters: extracting from question, model=%s", model)
    t0 = time.perf_counter()
    try:
        msg = llm.chat_message(model, [
            {"role": "system", "content": FILTER_SYSTEM},
            {"role": "user", "content": question},
        ], temperature=0, max_tokens=150, chat_template_kwargs={"enable_thinking": False},
           response_format={"type": "json_schema",
                            "json_schema": {"name": "filters", "strict": True, "schema": FILTER_SCHEMA}},
           **({"cache_prompt": False} if no_cache else {}))
        result = json.loads(_strip_think(msg.get("content") or "") or "{}")
    except Exception:
        logger.warning("[DOC-QA] _extract_filters: LLM call failed (%.0fms), returning empty filters",
                       (time.perf_counter() - t0) * 1000, exc_info=True)
        return {}
    year = result.get("year")
    if isinstance(year, bool) or not isinstance(year, int) or not (1900 <= year <= 2100):
        result["year"] = None
    stated_month = _MONTH_RE.search(question)
    quarter = result.get("quarter")
    if stated_month and quarter and quarter.strip()[:3].lower() != stated_month.group(1)[:3].lower():
        logger.info("[DOC-QA] _extract_filters: dropping quarter=%s (doesn't match stated month %s)",
                    sanitize_log_input(str(quarter)), sanitize_log_input(stated_month.group(1)))
        result["quarter"] = None
    tokens = {re.sub(r"[\s.-]", "", t.lower()) for t in _PERIOD_TOKEN_RE.findall(question)}
    years = set(_YEAR_TOKEN_RE.findall(question))
    if tokens or len(years) >= 2:
        result["year"] = None
        result["quarter"] = None
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-QA] _extract_filters: result=%s  %.0fms", result, elapsed)
    return result


def _filter_clauses(filters: dict) -> list[tuple[str, list]]:
    """Candidate WHERE fragments for each extracted field. _make_facts_view keeps only the ones
    that don't zero out the result, so a wrong guess (e.g. a metric name read as a company) can't
    blank the whole view."""
    clauses = []
    company = filters.get("company")
    if company:
        c = f"%{company}%"
        clauses.append(("(s.entity LIKE ? OR d.file_name LIKE ? OR d.title LIKE ?)", [c, c, c]))
    year = filters.get("year")
    if year:
        y4, y2 = str(year), f"{year % 100:02d}"
        clauses.append(("(f.period_key LIKE ? OR f.period_key LIKE ?)", [f"%{y4}%", f"%{y2}%"]))
    quarter = filters.get("quarter")
    if quarter:
        q = str(quarter).strip()
        m, h = _QUARTER_RE.match(q), _HALF_RE.match(q)
        if m:
            clauses.append(("f.period_label LIKE ?", [f"%Q{m.group(1)}%"]))
        elif h:
            clauses.append(("f.period_label LIKE ?", [f"%H{h.group(1)}%"]))
        elif q.upper() in ("FY", "9M"):
            clauses.append(("f.period_kind = ?", [q.upper()]))
        else:
            clauses.append(("f.period_label LIKE ?", [f"%{q[:3]}%"]))
    statement = filters.get("statement")
    if statement:
        s = f"%{statement}%"
        clauses.append(("(d.doc_type LIKE ? OR d.title LIKE ?)", [s, s]))
    return clauses


def chat_models() -> list[str]:
    """Chat models from config.CHAT_MODELS that the llama.cpp router currently reports."""
    try:
        available = llm.models()
    except Exception:
        return list(config.CHAT_MODELS)
    return [m for m in config.CHAT_MODELS if m in available]


def passage_text(p: dict) -> str:
    return p.get("text", "")


def _matching_metrics(question: str) -> list[dict]:
    q = question.lower()
    return [m for m in METRICS if m["name"].lower() in q or any(s in q for s in m.get("synonyms", []))]


async def _fts_ids(conn: asyncpg.Connection, table: str, question: str, k: int,
                   doc_ids: list[str] | None = None) -> list[int]:
    """Postgres tsquery over blocks.tsv / series.tsv (replaces SQLite FTS5)."""
    terms = re.findall(r"[A-Za-z0-9]+", question)
    if not terms:
        return []
    tsquery_text = " | ".join(terms)
    id_col = {"blocks": "block_id", "series": "series_id"}[table]
    sql = (f"SELECT {id_col} FROM {table} "
           f"WHERE tsv @@ to_tsquery('english', $1)")
    args: list = [tsquery_text]
    if doc_ids:
        args.append(list(doc_ids))
        sql += " AND doc_id = ANY($2::text[])"
    sql += (f" ORDER BY ts_rank(tsv, to_tsquery('english', $1)) DESC "
            f"LIMIT ${len(args) + 1}")
    args.append(k)
    rows = await conn.fetch(sql, *args)
    return [r[id_col] for r in rows]


async def _vec_distances(table: str, qvec: list[float], k: int,
                         doc_ids: list[str] | None = None) -> dict[int, float]:
    """k-NN via the active vector backend; returns {row_id: distance}."""
    vec = store._vec()
    fn = vec.search_blocks if table == "blocks" else vec.search_series
    hits = await asyncio.to_thread(fn, qvec, k, doc_ids)
    return dict(hits)


_RRF_K = 60  # standard smoothing constant; keeps one method's rank-1 hit from dominating

_ROUTE_STOPWORDS = {
    "a", "about", "actually", "after", "all", "and", "are", "as", "at", "be", "because",
    "been", "before", "between", "by", "can", "could", "did", "does", "during", "each", "for",
    "from", "give", "given", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "level", "month", "monthly", "months", "much", "of", "on", "or", "our", "over",
    "page", "period", "quarter", "quarterly", "report", "show", "shown", "than", "that", "the",
    "their", "this", "to", "total", "trend", "was", "were", "what", "when", "which", "why",
    "with", "would", "year", "yearly", "actual", "back", "broad", "broadbased", "compare",
    "concentrated", "current", "dip", "driver", "driven", "erosion", "faster", "full", "grow",
    "growth", "hold", "pack", "pace", "performance", "project", "projection", "sharpest",
    "specifically", "trajectory", "apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar",
    "may", "nov", "oct", "sep", "q1", "q2", "q3", "q4", "current", "daily", "decline",
    "part", "rate", "run", "data", "issue", "quality",
}


def _route_normalize(text: str) -> str:
    raw = (text or "").lower().replace("non-term", "nonterm")
    raw = re.sub(r"\bdays?\s*4\s*(?:-|–|—|to)\s*10\b", "day4to10", raw)
    words = re.findall(r"[a-z0-9]+", raw)
    canonical = {
        "cards": "card", "spends": "spend", "writeoffs": "writeoff",
        "closures": "closure", "additions": "addition", "repayments": "repayment",
    }
    return " ".join(canonical.get(word, word) for word in words)


def _route_tokens(text: str) -> set[str]:
    return {token for token in _route_normalize(text).split()
            if token not in _ROUTE_STOPWORDS and not token.isdigit()
            and not re.fullmatch(r"(?:19|20)?\d{2}", token)}


async def _candidate_pages(conn: asyncpg.Connection, question: str, doc_ids: list[str] | None,
                           limit: int = 3) -> list[int]:
    """Infer likely source pages from reusable document metadata."""
    q_tokens = _route_tokens(question)
    if not q_tokens:
        return []
    descriptors: list[tuple[int, str, float]] = []

    if doc_ids:
        series_rows = await conn.fetch(
            "SELECT page,label,name,path FROM series WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )
    else:
        series_rows = await conn.fetch("SELECT page,label,name,path FROM series")
    for row in series_rows:
        descriptors.extend((
            (int(row["page"]), row["label"] or "", 1.0),
            (int(row["page"]), row["name"] or "", 1.35),
            (int(row["page"]), row["path"] or "", 0.8),
        ))
    if doc_ids:
        block_rows = await conn.fetch(
            "SELECT page,title,section,section_path FROM blocks WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )
    else:
        block_rows = await conn.fetch(
            "SELECT page,title,section,section_path FROM blocks"
        )
    for row in block_rows:
        descriptors.extend((
            (int(row["page"]), row["title"] or "", 1.1),
            (int(row["page"]), row["section"] or "", 0.9),
            (int(row["page"]), row["section_path"] or "", 0.7),
        ))
    source_names: list[str] = []
    if doc_ids:
        source_names = [row[0] for row in await conn.fetch(
            "SELECT file_name FROM documents WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )]
    if source_names:
        rule_rows = await conn.fetch(
            "SELECT source_page,metric_name,metric_family,aliases FROM metric_rules "
            "WHERE enabled = 1 AND (source_name = '' OR source_name = ANY($1::text[]))",
            source_names,
        )
    else:
        rule_rows = await conn.fetch(
            "SELECT source_page,metric_name,metric_family,aliases FROM metric_rules "
            "WHERE enabled = 1"
        )
    for row in rule_rows:
        aliases = []
        try:
            aliases = json.loads(row["aliases"] or "[]")
        except json.JSONDecodeError:
            pass
        for value, weight in ((row["metric_name"], 1.5), (row["metric_family"], 0.8),
                              *((alias, 1.6) for alias in aliases)):
            descriptors.append((int(row["source_page"]), value or "", weight))

    token_pages: dict[str, set[int]] = {}
    prepared = []
    for page_number, text_value, weight in descriptors:
        tokens = _route_tokens(text_value)
        if not tokens:
            continue
        prepared.append((page_number, text_value, tokens, weight))
        for token in tokens:
            token_pages.setdefault(token, set()).add(page_number)
    page_count = max(1, len({page_number for page_number, _, _, _ in prepared}))
    page_hits: dict[int, list[float]] = {}
    q_normalized = _route_normalize(question)
    q_words = q_normalized.split()
    q_phrases = {" ".join(q_words[i:i + size]) for size in (2, 3, 4)
                 for i in range(len(q_words) - size + 1)}
    for page_number, text_value, tokens, weight in prepared:
        overlap = q_tokens & tokens
        if not overlap:
            continue
        rarity = sum(page_count / max(1, len(token_pages.get(token, ()))) for token in overlap)
        coverage = len(overlap) / len(q_tokens)
        descriptor = _route_normalize(text_value)
        phrase_size = max((len(phrase.split()) for phrase in q_phrases if phrase in descriptor), default=0)
        exact_bonus = 3.0 * phrase_size
        if len(tokens) == 1 and next(iter(tokens)) in q_tokens and len(next(iter(tokens))) >= 3:
            exact_bonus += 5.0
        score = weight * (rarity + 3.0 * coverage + exact_bonus)
        page_hits.setdefault(page_number, []).append(score)
    page_scores = []
    for page_number, hits in page_hits.items():
        ordered = sorted(hits, reverse=True)
        score = ordered[0] + 0.45 * sum(ordered[1:4])
        page_scores.append((score, page_number))
    page_scores.sort(reverse=True)
    if not page_scores or page_scores[0][0] < 5.0:
        return []
    # Very weak separation is genuine ambiguity: retain both pages. A decisive top page still
    # keeps the runner-up as a fallback when it has independently meaningful evidence.
    top_score = page_scores[0][0]
    selected = [page_scores[0][1]]
    if len(page_scores) > 1 and page_scores[1][0] >= max(5.0, top_score * 0.45):
        selected.append(page_scores[1][1])
    if len(page_scores) > 2 and page_scores[2][0] >= max(5.0, top_score * 0.45):
        selected.append(page_scores[2][1])
    return selected[:limit]


async def _ranked_ids(conn: asyncpg.Connection, table: str, question: str, qvec: list[float],
                      k: int, doc_ids: list[str] | None = None) -> list[int]:
    """Vector + keyword hybrid, combined by Reciprocal Rank Fusion."""
    over = max(k * 3, 15)
    vec_distances = await _vec_distances(table, qvec, over, doc_ids)
    vec_ranked = sorted(vec_distances.items(), key=lambda kv: kv[1])
    fts_ranked = await _fts_ids(conn, table, question, over, doc_ids)
    scores: dict[int, float] = {}
    for rank, (i, _) in enumerate(vec_ranked):
        scores[i] = scores.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, i in enumerate(fts_ranked):
        scores[i] = scores.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def _rerank(question: str, rows: list[dict], text: Callable[[dict], str], k: int,
            candidate_limit: int | None = config.RERANK_CANDIDATES) -> list[dict]:
    """Cross-encoder rerank of RRF's fused candidates down to the top k. A no-op (RRF order kept)
    whenever the reranker model isn't loaded — config.RERANK_MODEL is optional, so this must
    degrade quietly rather than break search."""
    if len(rows) <= k:
        return rows
    if not config.RERANK_MODEL:
        return rows[:k]
    if candidate_limit is not None:
        rows = rows[:candidate_limit]
    try:
        scores = llm.rerank(question, [text(r)[:config.RERANK_CHARS] for r in rows])
    except Exception:
        return rows[:k]
    return [r for r, _ in sorted(zip(rows, scores), key=lambda rs: rs[1], reverse=True)][:k]


async def _search_blocks(conn: asyncpg.Connection, question: str, qvec: list[float],
                         doc_ids: list[str] | None, k: int, page: int | None = None) -> list[dict]:
    if page is not None:
        sql = ("SELECT b.block_id, b.doc_id, d.file_name, b.page, b.kind, b.text "
               "FROM blocks b JOIN documents d ON d.doc_id = b.doc_id WHERE b.page = $1")
        args: list = [page]
        if doc_ids:
            args.append(list(doc_ids))
            sql += " AND b.doc_id = ANY($2::text[])"
        sql += " ORDER BY b.block_id"
        rows = await conn.fetch(sql, *args)
        return _rerank(question, [dict(r) for r in rows], lambda r: r["text"] or "", k,
                       candidate_limit=None)
    over = k * 5 if doc_ids else max(k * 3, 15)
    ids = await _ranked_ids(conn, "blocks", question, qvec, over, doc_ids)
    if not ids:
        return []
    sql = ("SELECT b.block_id, b.doc_id, d.file_name, b.page, b.kind, b.text "
           "FROM blocks b JOIN documents d ON d.doc_id = b.doc_id "
           "WHERE b.block_id = ANY($1::bigint[])")
    args = [list(ids)]
    if doc_ids:
        args.append(list(doc_ids))
        sql += " AND b.doc_id = ANY($2::text[])"
    rows = await conn.fetch(sql, *args)
    by_id = {r["block_id"]: dict(r) for r in rows}
    ordered = [by_id[i] for i in ids if i in by_id]
    return _rerank(question, ordered, lambda r: r["text"] or "", k)


async def _search_series(conn: asyncpg.Connection, question: str, qvec: list[float],
                         doc_ids: list[str] | None, k: int, page: int | None = None) -> list[dict]:
    if page is not None:
        sql = ("SELECT s.series_id, s.doc_id, d.file_name, s.page, s.label, s.metric_id, "
               "s.kind, s.unit, s.entity "
               "FROM series s JOIN documents d ON d.doc_id = s.doc_id WHERE s.page = $1")
        args: list = [page]
        if doc_ids:
            args.append(list(doc_ids))
            sql += " AND s.doc_id = ANY($2::text[])"
        sql += " ORDER BY s.series_id"
        rows = await conn.fetch(sql, *args)
        ordered = [dict(r) for r in rows]
    else:
        over = k * 5 if doc_ids else max(k * 3, 15)
        ids = await _ranked_ids(conn, "series", question, qvec, over, doc_ids)
        if not ids:
            return []
        sql = ("SELECT s.series_id, s.doc_id, d.file_name, s.page, s.label, s.metric_id, "
               "s.kind, s.unit, s.entity "
               "FROM series s JOIN documents d ON d.doc_id = s.doc_id "
               "WHERE s.series_id = ANY($1::bigint[])")
        args = [list(ids)]
        if doc_ids:
            args.append(list(doc_ids))
            sql += " AND s.doc_id = ANY($2::text[])"
        rows = await conn.fetch(sql, *args)
        by_id = {r["series_id"]: dict(r) for r in rows}
        ordered = [by_id[i] for i in ids if i in by_id]
    out = _rerank(question, ordered, lambda r: r["label"] or "", k,
                  candidate_limit=None if page is not None else config.RERANK_CANDIDATES)
    for s in out:
        facts = await conn.fetch(
            "SELECT period_label, as_printed, value FROM facts "
            "WHERE series_id = $1 ORDER BY period_start",
            s["series_id"],
        )
        s["facts"] = [dict(f) for f in facts]
    return out


async def _search_blocks_preferred(conn: asyncpg.Connection, question: str, qvec: list[float],
                                   doc_ids: list[str] | None, k: int,
                                   preferred_pages: list[int]) -> list[dict]:
    if not preferred_pages:
        return await _search_blocks(conn, question, qvec, doc_ids, k)
    preferred: list[dict] = []
    for index, page_number in enumerate(preferred_pages):
        page_k = max(1, k - 1) if index == 0 else 1
        preferred.extend(await _search_blocks(conn, question, qvec, doc_ids, page_k, page=page_number))
    broad = await _search_blocks(conn, question, qvec, doc_ids, k)
    combined = preferred + broad
    seen_ids: set[int] = set()
    selected: list[dict] = []
    for row in combined:
        if row["block_id"] not in seen_ids:
            seen_ids.add(row["block_id"])
            selected.append(row)
        if len(selected) >= k:
            break
    return selected


async def _search_series_preferred(conn: asyncpg.Connection, question: str, qvec: list[float],
                                   doc_ids: list[str] | None, k: int,
                                   preferred_pages: list[int]) -> list[dict]:
    if not preferred_pages:
        return await _search_series(conn, question, qvec, doc_ids, k)
    preferred: list[dict] = []
    for index, page_number in enumerate(preferred_pages):
        page_k = max(2, k - 2) if index == 0 else min(2, k)
        preferred.extend(await _search_series(conn, question, qvec, doc_ids, page_k, page=page_number))
    broad = await _search_series(conn, question, qvec, doc_ids, k)
    combined = preferred + broad
    seen_ids: set[int] = set()
    selected: list[dict] = []
    for row in combined:
        if row["series_id"] not in seen_ids:
            seen_ids.add(row["series_id"])
            selected.append(row)
        if len(selected) >= k:
            break
    return selected


async def _make_facts_view(conn: asyncpg.Connection, doc_ids: list[str] | None,
                           filters: dict | None = None) -> list[str]:
    """Create a session-scoped TEMP VIEW ``facts_view`` with filters inlined as literals.

    Postgres TEMP views live on the connection; since answer_events uses a single
    acquired connection for the whole request, downstream queries can reference it.
    Filters that would zero out the view are discarded.
    """
    conds: list[tuple[str, list, str]] = []
    if doc_ids:
        marks = ",".join(f"${i + 1}" for i in range(len(doc_ids)))
        conds.append((f"s.doc_id IN ({marks})", list(doc_ids), ""))
    fixed = len(conds)
    described = list(zip(_filter_clauses(filters or {}), _describe_filters(filters or {})))
    for (clause, params), desc in described:
        trial = conds + [(clause, params, desc)]
        # Renumber '?' placeholders across the combined clauses to $1..$N for Postgres.
        n = 0
        renumbered_parts = []
        for c, _, _ in trial:
            parts = c.split("?")
            renumbered = []
            for i, p in enumerate(parts):
                renumbered.append(p)
                if i < len(parts) - 1:
                    n += 1
                    renumbered.append(f"${n}")
            renumbered_parts.append("".join(renumbered))
        where = " AND ".join(renumbered_parts)
        args = [p for _, ps, _ in trial for p in ps]
        count = await conn.fetchval(
            "SELECT count(*) FROM facts f JOIN series s ON s.series_id = f.series_id "
            "JOIN documents d ON d.doc_id = s.doc_id WHERE " + where,
            *args,
        )
        if count and count > 0:
            conds = trial

    inlined = [_inline(c, ps) for c, ps, _ in conds]
    where = ("WHERE " + " AND ".join(inlined)) if inlined else ""
    await conn.execute("DROP VIEW IF EXISTS facts_view")
    await conn.execute(f"""
        CREATE TEMP VIEW facts_view AS
        SELECT f.fact_id, d.file_name, d.title, d.doc_type, d.as_of_label, s.page, s.label,
               s.metric_id, s.kind, s.unit, s.entity, f.period_key, f.period_kind, f.period_label,
               f.partial, f.value, f.as_printed, s.source, f.period_start, f.period_end
        FROM facts f JOIN series s ON s.series_id = f.series_id
        JOIN documents d ON d.doc_id = s.doc_id
        {where}
    """)
    return [desc for _, _, desc in conds[fixed:]]


def _sql_lit(v) -> str:
    """Inline a Python value as a Postgres literal (used only when baking filters
    into the TEMP VIEW definition, which cannot carry bound parameters)."""
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def _inline(clause: str, params: list) -> str:
    for p in params:
        clause = clause.replace("?", _sql_lit(p), 1)
    return clause


def _describe_filters(filters: dict) -> list[str]:
    out = []
    if filters.get("company"):
        out.append(f"company~{filters['company']}")
    if filters.get("year"):
        out.append(f"year~{filters['year']}")
    if filters.get("quarter"):
        out.append(f"period~{filters['quarter']}")
    if filters.get("statement"):
        out.append(f"statement~{filters['statement']}")
    return out


def _strip_think(text: str) -> str:
    """Granite's chat template sometimes leaks a reasoning block into content even with
    enable_thinking off; keep only what follows the last close tag, as the template itself does
    for older turns."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


# --- 2. every figure must trace back to a tool result ------------------------------------------

_NUM_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_CITATION_RE = re.compile(r"(?i)\(?p(?:age)?\.?\s?\d+\)?")


def _normalize_num(tok: str) -> str:
    return tok.replace(",", "").replace("%", "").strip()


def _looks_like_figure(tok: str) -> bool:
    """Skips small bare numbers ("2-4 sentences", "6 months") that are prose, not report figures.
    A bare "100" is also skipped even though it clears the digit-count bar: showing a share/growth
    calculation almost always writes "x / y * 100" to turn a ratio into a percentage, and that 100
    is the model's own arithmetic, never a value that needs a tool result behind it."""
    if tok == "100":
        return False
    digits = re.sub(r"\D", "", tok)
    return len(digits) >= 3 or bool(re.search(r"\.\d", tok)) or "%" in tok


def _record_seen(seen: set[str], result_json: str) -> None:
    """Every numeric value in a tool result is fair game, not just as_printed/value — calculate's
    "result" field legitimately returns a sum/share/growth figure the model built from other
    numbers it already had, and that's still a number a tool produced, not one the model invented.
    run_metric returns a JSON array of row dicts; calculate returns a single {"result": ...} (or
    {"error": ...}) dict — both shapes are scanned the same way."""
    try:
        parsed = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return
    rows = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for v in row.values():
            if isinstance(v, bool) or v in (None, ""):
                continue
            try:
                seen.add(round(float(_normalize_num(str(v))), 6))
            except ValueError:
                pass


def _parse_nums(text: str) -> set[float]:
    out = set()
    for tok in _NUM_TOKEN_RE.findall(text):
        try:
            out.add(float(_normalize_num(tok)))
        except ValueError:
            pass
    return out


def _unverified_numbers(answer: str, seen: set[float], question: str = "") -> list[str]:
    """A number the model merely echoed from the question (e.g. "in February 2026" -> "2026") isn't
    a claim that needs a tool result behind it — it was never a report figure to begin with. Only a
    number NOT already present in the question is checked against `seen`. Compared as floats, not
    strings: a number at the end of a sentence ("...2026.") pulls the sentence's period into the
    token itself ("2026."), which would never string-match the question's "2026"."""
    question_nums = _parse_nums(question)
    text = _CITATION_RE.sub("", answer)
    bad = []
    for tok in _NUM_TOKEN_RE.findall(text):
        if not _looks_like_figure(tok):
            continue
        try:
            f = float(_normalize_num(tok))
        except ValueError:
            continue
        if any(abs(f - q) <= 0.01 for q in question_nums):
            continue
        # Sign-tolerant: a subtraction/growth result is routinely narrated either way ("declined by
        # 99,756" vs. calculate's -99756) — same figure, and the sign is prose framing, not a
        # different number a tool never produced.
        if not any(abs(f - s) <= max(abs(s) * 0.001, 0.01) or abs(f + s) <= max(abs(s) * 0.001, 0.01)
                  for s in seen):
            bad.append(tok)
    return bad


def _dump_rows(rows: list, max_chars: int = _RESULT_CHARS) -> str:
    """json.dumps(rows), dropping trailing rows (never slicing the string mid-object) when the full
    dump would exceed max_chars. A blind [:max_chars] slice can cut off inside a value, handing the
    model invalid JSON and making _record_seen's parse fail silently — which empties `seen` and
    makes every number in a later answer look unverified even when it's exactly right."""
    full = json.dumps(rows, default=str)
    if len(full) <= max_chars:
        return full
    out: list[str] = []
    total = 2  # "[" + "]"
    for r in rows:
        s = json.dumps(r, default=str)
        added = len(s) + (1 if out else 0)  # +1 for the joining comma
        if total + added > max_chars:
            break
        out.append(s)
        total += added
    return "[" + ",".join(out) + "]"


def _resolve_metric_id(label: str) -> str | None:
    """A metrics.yaml name/synonym resolves `label` to its canonical id regardless of which words
    the model chose — it doesn't have to spell "cc_spends" for this to be exact. Every word in
    `label` must be accounted for by the candidate (query words ⊆ candidate words), not merely
    overlapping — "proprietary cards sourced" must NOT resolve via the synonym "cards sourced"
    just because it contains it; "proprietary" is the word that actually distinguishes this row,
    and metrics.yaml has no id for a specific product/channel breakdown."""
    q_words = set(re.findall(r"[a-z0-9]+", (label or "").lower()))
    if not q_words:
        return None
    for m in METRICS:
        for cand in (m["name"].lower(), *m.get("synonyms", [])):
            if q_words <= set(re.findall(r"[a-z0-9]+", cand)):
                return m["id"]
    return None


async def _run_metric(conn: asyncpg.Connection, label: str, period: str = "",
                      preferred_pages: list[int] | None = None) -> str:
    """label resolves to a canonical metric_id (via metrics.yaml or an id typed directly) when
    possible — that's exact. Otherwise it matches by words, against label only: matching loose
    words against metric_id too let unrelated metrics that merely share a word root (cc_spends vs
    cc_spend_per_card_day) collide. A phrase-match bonus on top of the per-word score is required
    because hierarchical labels share a common prefix (e.g. every row under "... Channel & Product
    ..." contains the word "Product"), so word-only scoring can't tell "Product Mix" apart from
    "Channel Mix" — both match every word equally. When several distinct labels still tie for the
    top score, that's usually a breakdown family under one category (e.g. Proprietary/Consumer/
    Distribution/Premium/Total all under "Product Mix") rather than duplicate rows for one metric
    — the model asked for the mix, not any single row, so every tied row is returned instead of
    arbitrarily keeping only the shortest label (which is reliably "Total", silently discarding the
    breakdown). A large tie count instead means the words matched too broadly (e.g. one generic
    word) — the tightest (shortest, canonical) single match is kept there so the model isn't
    flooded with unrelated rows."""
    label = (label or "").strip()
    words = re.findall(r"[A-Za-z0-9]+", label)
    if not words:
        return json.dumps({"error": "label is required"})
    preferred_pages = [int(value) for value in (preferred_pages or [])]
    page_order = (f"CASE WHEN page = ANY(ARRAY[{','.join(map(str, preferred_pages))}]::int[]) "
                  f"THEN 0 ELSE 1 END"
                  if preferred_pages else "0::int")
    best = None
    resolved = _resolve_metric_id(label) or label

    same_label = await conn.fetchrow(
        f"SELECT label FROM facts_view WHERE LOWER(TRIM(label)) = LOWER($1) "
        f"ORDER BY {page_order}, LENGTH(label) LIMIT 1",
        label,
    )
    if same_label:
        best = same_label
    elif period:
        best = await conn.fetchrow(
            f"SELECT label FROM facts_view WHERE LOWER(metric_id) = LOWER($1) "
            f"AND period_label LIKE $2 ORDER BY {page_order}, LENGTH(label) ASC LIMIT 1",
            resolved, f"%{period}%",
        )
    if not best:
        best = await conn.fetchrow(
            f"SELECT label FROM facts_view WHERE LOWER(metric_id) = LOWER($1) "
            f"ORDER BY {page_order}, LENGTH(label) ASC LIMIT 1",
            resolved,
        )
    if best and preferred_pages:
        if period:
            present = await conn.fetchrow(
                "SELECT 1 FROM facts_view WHERE label = $1 "
                "AND page = ANY($2::int[]) AND period_label LIKE $3 LIMIT 1",
                best["label"], preferred_pages, f"%{period}%",
            )
        else:
            present = await conn.fetchrow(
                "SELECT 1 FROM facts_view WHERE label = $1 "
                "AND page = ANY($2::int[]) LIMIT 1",
                best["label"], preferred_pages,
            )
        if not present:
            canonical = next((metric for metric in METRICS if metric["id"] == resolved), None)
            canonical_words = sorted(_route_tokens(canonical["name"])) if canonical else []
            if canonical_words:
                # Build: (label LIKE $1)+(label LIKE $2)+... and (label LIKE $1 OR ... )
                score_terms = [f"(label LIKE ${i + 1})::int" for i in range(len(canonical_words))]
                any_terms   = [f"label LIKE ${i + 1}"       for i in range(len(canonical_words))]
                likes = [f"%{w}%" for w in canonical_words]
                pages_arg_idx = len(canonical_words) + 1
                candidate = await conn.fetchrow(
                    f"SELECT label, ({' + '.join(score_terms)}) AS score FROM facts_view "
                    f"WHERE page = ANY(${pages_arg_idx}::int[]) AND ({' OR '.join(any_terms)}) "
                    f"GROUP BY label ORDER BY score DESC, LENGTH(label) ASC LIMIT 1",
                    *likes, preferred_pages,
                )
                if candidate and candidate["score"] > 0:
                    best = candidate
    if best:
        best_labels = [best["label"]]
    else:
        score_terms = [f"(label LIKE ${i + 1})::int" for i in range(len(words))]
        any_terms   = [f"label LIKE ${i + 1}"       for i in range(len(words))]
        phrase_idx = len(words) + 1
        like_params = [f"%{w}%" for w in words]
        phrase_param = f"%{' '.join(words).lower()}%"
        score_sql = f"({' + '.join(score_terms)}) + (10 * (LOWER(label) LIKE ${phrase_idx})::int)"
        where_any = " OR ".join(any_terms)
        candidates: list = []
        if period:
            period_idx = phrase_idx + 1
            candidates = await conn.fetch(
                f"SELECT label, ({score_sql}) AS score, MIN({page_order}) AS route_rank "
                f"FROM facts_view WHERE ({where_any}) AND period_label LIKE ${period_idx} "
                f"GROUP BY label ORDER BY route_rank, score DESC, LENGTH(label) ASC",
                *like_params, phrase_param, f"%{period}%",
            )
        if not candidates:
            candidates = await conn.fetch(
                f"SELECT label, ({score_sql}) AS score, MIN({page_order}) AS route_rank "
                f"FROM facts_view WHERE {where_any} "
                f"GROUP BY label ORDER BY route_rank, score DESC, LENGTH(label) ASC",
                *like_params, phrase_param,
            )
        if not candidates:
            return json.dumps([])
        if preferred_pages and any(c["route_rank"] == 0 for c in candidates):
            candidates = [c for c in candidates if c["route_rank"] == 0]
        top_score = candidates[0]["score"]
        tied = [c["label"] for c in candidates if c["score"] == top_score]
        best_labels = tied if 1 < len(tied) <= 8 else [candidates[0]["label"]]

    if period:
        rows = await conn.fetch(
            "SELECT file_name, page, label, metric_id, entity, unit, period_key, period_label, "
            "partial, value, as_printed, source FROM facts_view "
            "WHERE label = ANY($1::text[]) AND period_label LIKE $2 "
            "ORDER BY label, period_key LIMIT $3",
            best_labels, f"%{period}%", _ROW_LIMIT,
        )
    else:
        rows = await conn.fetch(
            "SELECT file_name, page, label, metric_id, entity, unit, period_key, period_label, "
            "partial, value, as_printed, source FROM facts_view "
            "WHERE label = ANY($1::text[]) ORDER BY label, period_key LIMIT $2",
            best_labels, _ROW_LIMIT,
        )
    if preferred_pages:
        preferred_rows = [row for row in rows if int(row["page"]) in preferred_pages]
        if preferred_rows:
            rows = preferred_rows
    if not rows and period and len(best_labels) == 1:
        rollup = await _period_rollup(conn, best_labels[0], period)
        if rollup:
            return _dump_rows(rollup)
    if not rows and period:
        have = [r["period_label"] for r in await conn.fetch(
            "SELECT DISTINCT period_label, period_key FROM facts_view "
            "WHERE label = ANY($1::text[]) ORDER BY period_key LIMIT 24",
            best_labels,
        )]
        return json.dumps({"rows": [],
                           "label": best_labels[0] if len(best_labels) == 1 else best_labels,
                           "note": f"No row for period '{period}'. Periods this label has: "
                                   f"{', '.join(have) or 'none'}."})
    return _dump_rows([_trust_value(dict(r)) for r in rows])


def _trust_value(row: dict) -> dict:
    """as_printed is the extractor's copy of the printed text, and occasionally drops a digit
    ("12,34" for 12,345) while `value` is right. A model told to quote as_printed then reports a
    wrong figure, so show the stored value whenever the two disagree."""
    try:
        printed = float(_normalize_num(str(row.get("as_printed") or "")).replace("(", "-").replace(")", ""))
    except ValueError:
        return row
    value = row.get("value")
    if isinstance(value, (int, float)) and abs(printed - value) > max(abs(value) * 0.001, 0.01):
        row["as_printed"] = f"{value:,.2f}".rstrip("0").rstrip(".")
    return row


async def _period_rollup(conn: asyncpg.Connection, label: str, period: str) -> list[dict]:
    """A quarter's constituent months (etc.) when the label isn't published at that period."""
    bounds = await conn.fetchrow(
        "SELECT MIN(period_start) AS start, MAX(period_end) AS end FROM facts_view "
        "WHERE LOWER(period_label) = LOWER($1) AND period_start IS NOT NULL AND period_start <> ''",
        period,
    )
    if not bounds or not bounds["start"]:
        return []
    rows = await conn.fetch(
        "SELECT file_name, page, label, metric_id, entity, unit, period_key, period_label, partial, "
        "value, as_printed, source FROM facts_view WHERE label = $1 AND period_start >= $2 "
        "AND period_end <= $3 ORDER BY period_key LIMIT $4",
        label, bounds["start"], bounds["end"], _ROW_LIMIT,
    )
    if not rows:
        return []
    note = (f"No row for '{period}' directly — {label} is only reported at finer periods. "
           f"Above are all {len(rows)} periods inside {period}'s window instead; sum their "
           f"values with calculate for the {period} total (only if this is a flow figure that "
           f"should be summed, not a running balance where you'd want the last period's value).")
    # The note goes last, not first: a long instructional sentence sitting before the data rows
    # pushes the first real row out of the primacy position, and a model this size has been
    # observed to skip exactly that row when building a multi-value sum from the result.
    return [*(dict(r) for r in rows), {"note": note}]


def _format_metrics(metrics: list[dict]) -> str:
    if not metrics:
        return "(none matched by name)"
    return "\n".join(f"- {m['id']}: {m['name']} ({m['kind']}" +
                     (f", usual range {m['range'][0]}-{m['range'][1]}" if m.get("range") else "") + ")"
                     for m in metrics)


def _format_series(series: list[dict]) -> str:
    """Identity only, no values: which label/page/periods exist. Get the actual number with
    run_metric — a period picked by eye from a long list here is exactly how a value gets misread."""
    if not series:
        return "(none matched)"
    lines = []
    for s in series:
        have = [f["period_label"] for f in s["facts"]]
        span = f"{have[0]} to {have[-1]}" if have else "no periods"
        lines.append(f"- \"{s['label']}\" (metric_id={s['metric_id'] or 'n/a'}, unit={s['unit'] or s['kind']}), "
                     f"{s['file_name']} p.{s['page']}, periods {span}")
    return "\n".join(lines)


def _format_passages(passages: list[dict]) -> str:
    """Compact retrieved evidence. Numeric answers still have to use run_metric/calculate, but
    table/chart headers and row names help the model choose the correct series and interpret it."""
    if not passages:
        return "(none matched)"
    return "\n".join(
        f"p.{p['page']} ({p['file_name']}) [{p['kind']}]: "
        f"{passage_text(p)[:1200 if p['kind'] in ('table', 'chart') else 800]}"
        for p in passages)


async def _page_issues(conn: asyncpg.Connection, doc_ids: list[str] | None, page: int | None) -> str:
    if page is None:
        return ""
    if doc_ids:
        rows = await conn.fetch(
            "SELECT kind, ref, message, suggestion FROM issues "
            "WHERE page = $1 AND doc_id = ANY($2::text[]) ORDER BY issue_id",
            page, list(doc_ids),
        )
    else:
        rows = await conn.fetch(
            "SELECT kind, ref, message, suggestion FROM issues "
            "WHERE page = $1 ORDER BY issue_id",
            page,
        )
    if not rows:
        return ""
    lines = ["Ingestion validation flags for this page:"]
    for row in rows:
        line = f"- {row['kind']} — {row['ref']}: {row['message']}"
        if row["suggestion"]:
            line += f"; suggested review: {row['suggestion']}"
        lines.append(line)
    return "\n".join(lines) + "\n"


async def _scope_filters_text(doc_ids: list[str] | None, conn: asyncpg.Connection,
                              applied_filters: list[str]) -> str:
    scope = ""
    if doc_ids:
        names = [r[0] for r in await conn.fetch(
            "SELECT file_name FROM documents WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )]
        scope = f"Only these documents are in scope: {', '.join(names)}.\n"
    return (f"{scope}facts_view is already filtered to {', '.join(applied_filters)} — you don't "
           f"need to filter for that yourself unless you want to narrow further.\n" if applied_filters
           else scope)


async def _system_prompt(passages: list[dict], series: list[dict], metrics: list[dict],
                         doc_ids: list[str] | None, conn: asyncpg.Connection,
                         applied_filters: list[str]) -> str:
    filters = await _scope_filters_text(doc_ids, conn, applied_filters)
    return SYSTEM_TEMPLATE.format(schema=SCHEMA_DOC, filters=filters, metrics=_format_metrics(metrics),
                                  series=_format_series(series), passages=_format_passages(passages))


async def _context_block(passages: list[dict], series: list[dict], metrics: list[dict],
                         doc_ids: list[str] | None, conn: asyncpg.Connection,
                         applied_filters: list[str]) -> str:
    filters = await _scope_filters_text(doc_ids, conn, applied_filters)
    return (f"{SCHEMA_DOC}\n{filters}\n"
           f"Relevant metric definitions:\n{_format_metrics(metrics)}\n\n"
           f"Series that exist (labels only — call a tool for their values):\n{_format_series(series)}\n\n"
           f"Passages from the reports (raw extracted text, for narrative questions such as \"why\" only):\n"
           f"{_format_passages(passages)}")


async def _dispatch_tool(conn: asyncpg.Connection, call: dict, seen: set[float],
                         preferred_pages: list[int] | None = None) -> tuple[str, dict, str]:
    """Returns (name, args, result) — the trace an answer's numbers came from, for the UI."""
    name = call["function"].get("name")
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    logger.info("[DOC-QA] _dispatch_tool: %s(%s)  preferred_pages=%s", name, str(args)[:200], preferred_pages)
    t0 = time.perf_counter()
    if name == "run_metric":
        result = await _run_metric(conn, args.get("label", ""), args.get("period", ""), preferred_pages)
    elif name == "calculate":
        result = json.dumps(calc.calculate(args.get("expression", "")))
    else:
        result = json.dumps({"error": f"no such tool: {name}"})
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-QA] _dispatch_tool: %s -> %d chars result, %.0fms  result_preview='%s'",
                name, len(result), elapsed, result[:300])
    _record_seen(seen, result)
    return name, args, result


def _plaintext_tool_request(content: str) -> tuple[str, dict] | None:
    """Recover a tool request that a small model emitted as answer text.

    This is deliberately syntax-based and metric-agnostic: it accepts only a literal
    ``calculate("...")`` request or a JSON object accompanying the word ``run_metric``.
    It does not infer a metric, period, page, or formula from the user's question.
    """
    calculate_match = re.fullmatch(
        r"\s*calculate\(\s*(['\"])(.*?)\1\s*\)\s*\.?\s*", content or "", re.DOTALL | re.IGNORECASE
    )
    if calculate_match:
        return "calculate", {"expression": calculate_match.group(2)}
    if "run_metric" not in (content or "").lower():
        return None
    object_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if object_match:
        try:
            payload = json.loads(object_match.group())
        except json.JSONDecodeError:
            return None
        label = str(payload.get("label", "")).strip()
        period = str(payload.get("period") or payload.get("period_label") or "").strip()
    else:
        label_match = re.search(r'(?i)\blabel\s+["“]([^"”]+)["”]', content)
        if not label_match:
            return None
        label = label_match.group(1).strip()
        period_match = re.search(r"(?i)\bperiod_label\s+([A-Za-z0-9'._-]+)", content)
        period = period_match.group(1).strip(".')") if period_match else ""
    if not label:
        return None
    return "run_metric", {"label": label, "period": period}


async def answer_events(question: str, history: list[dict] | None = None, doc_ids: list[str] | None = None,
                        model: str | None = None, reasoning: bool = False, style: str = "full",
                        no_cache: bool = False, page: int | None = None, page_title: str = ""):
    """Yields ("stage"|"passages"|"series"|"tools"|"warning"|"answer", value). "tools" carries the
    run_metric/calculate calls (with their arguments and raw results) that produced the answer's
    numbers, one list yielded once before the final "answer". "warning" is yielded only when an
    answer still has numbers unverified against those results after a retry.

    style: "full" (default) uses SYSTEM_TEMPLATE — schema, tool rules and retrieved context all in
    the system message. "insights" uses the bare two-line SYSTEM_TEMPLATE_INSIGHTS instead; the
    retrieved schema/metrics/series/passages still have to reach the model, so they ride in the
    user turn instead of the system message.

    no_cache: llama-server's cache-reuse (models.ini) makes generation at temperature=0 non-
    reproducible across a *sequence* of different questions — CPU batched float accumulation order
    shifts with whatever's already cached, which can flip a near-tied argmax. True asks the server
    to skip cache reuse for every call in this question, so eval runs are actually deterministic;
    left off by default since it makes the interactive app slower for no benefit there."""
    model = model or config.CHAT_MODEL
    from core.db import get_pool
    pool = get_pool()
    if pool is None:
        raise RuntimeError("PostgreSQL pool not initialised (POSTGRES_DSN missing).")
    conn = await pool.acquire()
    try:
        async for event in _answer_events_impl(
            conn, question, history, doc_ids, model, reasoning, style,
            no_cache, page, page_title,
        ):
            yield event
    finally:
        await pool.release(conn)


async def _answer_events_impl(conn, question, history, doc_ids, model, reasoning, style,
                              no_cache, page, page_title):

    logger.info("[DOC-QA] ═══ answer_events START ═══  question='%s'  model=%s  style=%s  "
                "reasoning=%s  page=%s  doc_ids=%s  history=%d",
                sanitize_log_input(question[:150]), model, style, reasoning, page, doc_ids, len(history or []))
    t_total = time.perf_counter()

    logger.info("[DOC-QA] Step 1: candidate_pages + generic_metrics.resolve()")
    t0 = time.perf_counter()
    preferred_pages = [int(page)] if page is not None else await _candidate_pages(conn, question, doc_ids)
    logger.info("[DOC-QA]   preferred_pages=%s  (%.0fms)", preferred_pages, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    resolved = await generic_metrics.resolve(conn, question, doc_ids=doc_ids, page=page, limit=12)
    logger.info("[DOC-QA]   generic_metrics.resolve() -> %d rows  (%.0fms)", len(resolved), (time.perf_counter() - t0) * 1000)

    deterministic_group=any(str(row.get("period_key","")).startswith("G:") for row in resolved)
    if (page is None and preferred_pages and not generic_metrics._extremum_directions(question)
            and not deterministic_group):
        before = len(resolved)
        resolved = [row for row in resolved if int(row["source_page"]) == preferred_pages[0]]
        logger.info("[DOC-QA]   filtered resolved to preferred page %s: %d -> %d rows",
                    preferred_pages[0], before, len(resolved))
    if resolved:
        result_json = json.dumps(resolved, ensure_ascii=False, default=str)
        logger.info("[DOC-QA]   DETERMINISTIC PATH: returning generic_metrics answer (%d chars)", len(result_json))
        yield "stage", "Using validated calculations…"
        yield "passages", []
        yield "series", []
        yield "tools", [{"tool": "generic_metrics", "args": {"question": question},
                         "result": result_json}]
        answer = generic_metrics.deterministic_answer(resolved)
        logger.info("[DOC-QA] ═══ answer_events END (deterministic) ═══  %.0fms  answer='%s'",
                    (time.perf_counter() - t_total) * 1000, (answer or "")[:200])
        yield "answer", answer
        return

    logger.info("[DOC-QA] Step 2: generic_metrics.reviewed_metric_matches()")
    t0 = time.perf_counter()
    reviewed = await generic_metrics.reviewed_metric_matches(
        conn, question, doc_ids=doc_ids, page=page, limit=3
    )
    logger.info("[DOC-QA]   reviewed_metric_matches() -> %d rows  (%.0fms)",
                len(reviewed) if reviewed else 0, (time.perf_counter() - t0) * 1000)
    if reviewed:
        logger.info("[DOC-QA]   REVIEWED PATH: returning reviewed metric answer")
        yield "stage", "Checking business metric validation…"
        yield "passages", []
        yield "series", []
        yield "tools", [{"tool": "generic_metrics_review", "args": {"question": question},
                         "result": json.dumps(reviewed, ensure_ascii=False, default=str)}]
        answer = generic_metrics.reviewed_metric_answer(reviewed)
        logger.info("[DOC-QA] ═══ answer_events END (reviewed) ═══  %.0fms  answer='%s'",
                    (time.perf_counter() - t_total) * 1000, (answer or "")[:200])
        yield "answer", answer
        return

    logger.info("[DOC-QA] Step 3: resolve followup + extract filters")
    yield "stage", "Reading the question…"
    t0 = time.perf_counter()
    search_question = _resolve_followup(question, history, model, no_cache=no_cache, reasoning=reasoning)
    logger.info("[DOC-QA]   search_question='%s'  (%.0fms)", sanitize_log_input(search_question[:150]), (time.perf_counter() - t0) * 1000)

    if page is not None:
        search_question = f"{page_title or 'Source page'} (page {page}): {search_question}"
        logger.info("[DOC-QA]   page-prefixed search_question='%s'", sanitize_log_input(search_question[:150]))
    else:
        t0 = time.perf_counter()
        preferred_pages = await _candidate_pages(conn, search_question, doc_ids)
        logger.info("[DOC-QA]   re-computed preferred_pages=%s  (%.0fms)", preferred_pages, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    filters = _extract_filters(search_question, model, no_cache=no_cache)
    logger.info("[DOC-QA]   filters=%s  (%.0fms)", filters, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    applied = await _make_facts_view(conn, doc_ids, filters)
    logger.info("[DOC-QA]   facts_view created, applied_filters=%s  (%.0fms)", applied, (time.perf_counter() - t0) * 1000)

    logger.info("[DOC-QA] Step 4: embed + hybrid search (passages + series)")
    yield "stage", "Searching the reports…"
    t0 = time.perf_counter()
    qvec = llm.embed(["search_query: " + search_question])[0]
    logger.info("[DOC-QA]   embedding done: dim=%d  (%.0fms)", len(qvec), (time.perf_counter() - t0) * 1000)

    passage_k = max(config.PASSAGES_K, 4) if page is not None else config.PASSAGES_K
    series_k = max(config.SERIES_K, 10) if page is not None else config.SERIES_K
    logger.info("[DOC-QA]   passage_k=%d  series_k=%d", passage_k, series_k)

    t0 = time.perf_counter()
    if page is not None:
        passages = await _search_blocks(conn, search_question, qvec, doc_ids, passage_k, page=page)
    else:
        passages = await _search_blocks_preferred(conn, search_question, qvec, doc_ids, passage_k, preferred_pages)
    logger.info("[DOC-QA]   passages search: %d results  (%.0fms)", len(passages), (time.perf_counter() - t0) * 1000)
    for i, p in enumerate(passages):
        logger.info("[DOC-QA]     passage[%d]: page=%s kind=%s file=%s text='%s'",
                    i, p.get("page"), p.get("kind"), p.get("file_name"), (p.get("text") or "")[:80])

    t0 = time.perf_counter()
    if page is not None:
        series = await _search_series(conn, search_question, qvec, doc_ids, series_k, page=page)
    else:
        series = await _search_series_preferred(conn, search_question, qvec, doc_ids, series_k, preferred_pages)
    logger.info("[DOC-QA]   series search: %d results  (%.0fms)", len(series), (time.perf_counter() - t0) * 1000)
    for i, s in enumerate(series):
        logger.info("[DOC-QA]     series[%d]: page=%s label='%s' metric_id=%s facts=%d",
                    i, s.get("page"), s.get("label"), s.get("metric_id"), len(s.get("facts", [])))

    page_issues = await _page_issues(conn, doc_ids, page)
    if page_issues:
        logger.info("[DOC-QA]   page_issues: %d chars", len(page_issues))
    yield "passages", passages
    yield "series", series

    t0 = time.perf_counter()
    precomputed = await generic_metrics.search(conn, question, doc_ids, page=page, limit=12)
    if page is None and preferred_pages:
        precomputed = [row for row in precomputed if int(row["source_page"]) in preferred_pages]
    logger.info("[DOC-QA]   precomputed metrics: %d rows  (%.0fms)", len(precomputed), (time.perf_counter() - t0) * 1000)

    metrics = {m["id"]: m for m in _matching_metrics(search_question)}
    for s in series:
        metrics.update({m["id"]: m for m in METRICS if m["id"] == s["metric_id"]})
    logger.info("[DOC-QA] Step 5: build prompt  style=%s  metrics=%d  precomputed=%d",
                style, len(metrics), len(precomputed))

    if style == "insights":
        system = SYSTEM_TEMPLATE_INSIGHTS
        context = await _context_block(passages, series, list(metrics.values()), doc_ids, conn, applied)
        hint = f"Source page: p.{page} — {page_title}.\n" if page is not None else ""
        precomputed_block = ("\n\nPRECOMPUTED AND VALIDATED BUSINESS METRICS. Use only a matching stored result for "
                             "calculation questions; do not calculate again:\n"
                             + generic_metrics.context(precomputed)) if precomputed else ""
        user_content = f"{context}{precomputed_block}\n\n{hint}{page_issues}Question: {question}"
    else:
        system = await _system_prompt(passages, series, list(metrics.values()), doc_ids, conn, applied)
        hint = f"Source page: p.{page} — {page_title}.\n" if page is not None else ""
        precomputed_block = ("\n\nPRECOMPUTED AND VALIDATED BUSINESS METRICS. Use only a matching stored result for "
                             "calculation questions; do not calculate again:\n"
                             + generic_metrics.context(precomputed)) if precomputed else ""
        user_content = hint + page_issues + question + precomputed_block

    messages = [{"role": "system", "content": system}]
    for h in history or []:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_content})
    logger.info("[DOC-QA]   prompt built: system=%d chars, user=%d chars, total_messages=%d",
                len(system), len(user_content), len(messages))

    # Thinking on gives the model room to reason through which tool to call and how to combine
    # results; off for the filter pass above, where a fast, single JSON guess is all that's needed.
    think_kwargs = {"enable_thinking": reasoning}
    if reasoning:
        think_kwargs["reasoning_effort"] = "low"
    params = {"temperature": 0, "max_tokens": 3000 if reasoning else 2000, "tools": TOOL_SPECS,
             "tool_choice": "auto", "chat_template_kwargs": think_kwargs}
    if no_cache:
        params["cache_prompt"] = False
    seen: set[float] = set()
    verify_retries = 2
    tool_trace: list[dict] = []
    if precomputed:
        result_json = json.dumps(precomputed, ensure_ascii=False, default=str)
        _record_seen(seen, result_json)
        tool_trace.append({"tool": "generic_metrics", "args": {"question": question}, "result": result_json})
    seen_calls: set[tuple[str, str]] = set()

    logger.info("[DOC-QA] Step 6: entering tool-calling loop (max %d rounds)", _MAX_ROUNDS)

    for round_num in range(_MAX_ROUNDS):
        logger.info("[DOC-QA]   ── Round %d/%d ──  messages=%d  tool_trace=%d  seen_values=%d",
                    round_num + 1, _MAX_ROUNDS, len(messages), len(tool_trace), len(seen))
        t0 = time.perf_counter()
        try:
            msg = llm.chat_message(model, messages, **params)
        except httpx.HTTPStatusError as e:
            logger.warning("[DOC-QA]   Round %d: HTTPStatusError %s (%.0fms) — asking model to retry",
                           round_num + 1, e.response.status_code, (time.perf_counter() - t0) * 1000)
            messages.append({"role": "user", "content": (
                "That tool call was cut off and invalid. Call the tool again with a shorter, "
                "simpler query.")})
            continue
        round_ms = (time.perf_counter() - t0) * 1000
        calls = msg.get("tool_calls") or []
        content = _strip_think(msg.get("content") or "")
        logger.info("[DOC-QA]   Round %d result: content=%d chars, tool_calls=%d, %.0fms",
                    round_num + 1, len(content), len(calls), round_ms)
        if content:
            logger.info("[DOC-QA]   Round %d content preview: '%s'", round_num + 1, content[:300])

        if not calls:
            if not content and params["chat_template_kwargs"].get("enable_thinking"):
                logger.info("[DOC-QA]   Round %d: empty content with thinking on — retrying without thinking",
                            round_num + 1)
                params["chat_template_kwargs"] = think_kwargs = {"enable_thinking": False}
                params["max_tokens"] = 2000
                continue
            if not content:
                logger.warning("[DOC-QA]   Round %d: empty content, no tool calls — giving up", round_num + 1)
                yield "tools", tool_trace
                yield "answer", "I couldn't find that in the reports."
                logger.info("[DOC-QA] ═══ answer_events END (no content) ═══  %.0fms",
                            (time.perf_counter() - t_total) * 1000)
                return
            plaintext_request = _plaintext_tool_request(content)
            if plaintext_request:
                name, args = plaintext_request
                logger.info("[DOC-QA]   Round %d: plaintext tool request detected: %s(%s)",
                            round_num + 1, name, str(args)[:150])
                key = (name, json.dumps(args, sort_keys=True))
                if key not in seen_calls:
                    seen_calls.add(key)
                    call = {"function": {"name": name, "arguments": json.dumps(args)}}
                    dispatched_name, dispatched_args, result = await _dispatch_tool(
                        conn, call, seen, preferred_pages
                    )
                    tool_trace.append({"tool": dispatched_name, "args": dispatched_args, "result": result})
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": (
                        "The tool request you wrote as plain text has now been executed. Its result is:\n"
                        f"{result}\nAnswer the original question directly in plain text."
                    )})
                    continue
                else:
                    logger.info("[DOC-QA]   Round %d: duplicate plaintext tool call, skipping", round_num + 1)
            bad = _unverified_numbers(content, seen, question)
            if bad and verify_retries > 0:
                verify_retries -= 1
                logger.info("[DOC-QA]   Round %d: unverified numbers %s — asking model to retry (retries_left=%d)",
                            round_num + 1, bad, verify_retries)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    f"You wrote {', '.join(bad)}, which doesn't match any run_metric/calculate "
                    "result yet. Keep your narrative conclusion, but look up or remove only the "
                    "unsupported numeric claim(s), then answer the original question directly.")})
                continue
            yield "tools", tool_trace
            if bad:
                logger.warning("[DOC-QA]   Round %d: FINAL answer has unverified numbers: %s", round_num + 1, bad)
                yield "warning", (f"Not matched to a tool result after a retry: {', '.join(bad)}. "
                                  "Treat those number(s) with caution.")
            logger.info("[DOC-QA] ═══ answer_events END (answer) ═══  %.0fms  round=%d  answer='%s'",
                        (time.perf_counter() - t_total) * 1000, round_num + 1, content[:200])
            yield "answer", content
            return

        # Process tool calls
        logger.info("[DOC-QA]   Round %d: processing %d tool call(s)", round_num + 1, len(calls))
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        yield "stage", "Looking up numbers…"
        for call_idx, call in enumerate(calls):
            key = (call["function"].get("name", ""), call["function"].get("arguments") or "{}")
            if key in seen_calls:
                logger.info("[DOC-QA]   Round %d call[%d]: DUPLICATE %s — returning error",
                            round_num + 1, call_idx, key[0])
                name = key[0]
                args = json.loads(key[1]) if key[1] else {}
                result = json.dumps({"error": (
                    "Already called with these exact arguments; repeating it won't return anything "
                    "new. If you need to compare it against another row, call run_metric for that "
                    "other row's label, then calculate on the numbers both calls returned.")})
            else:
                seen_calls.add(key)
                name, args, result = await _dispatch_tool(conn, call, seen, preferred_pages)
            tool_trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})

    # Round budget exhausted while the model kept calling tools instead of answering
    logger.warning("[DOC-QA] Step 7: round budget EXHAUSTED (%d rounds) — forcing synthesis pass", _MAX_ROUNDS)
    yield "stage", "Finishing up…"
    messages.append({"role": "user", "content": (
        "Stop calling tools. Answer the question now using only the run_metric/calculate results "
        "already returned above. If none of them answer it, say you couldn't find it.")})
    t0 = time.perf_counter()
    try:
        content = _strip_think(llm.chat(model, messages, temperature=0, max_tokens=2000,
                                        chat_template_kwargs=think_kwargs))
        logger.info("[DOC-QA]   synthesis pass: %d chars  (%.0fms)  content='%s'",
                    len(content), (time.perf_counter() - t0) * 1000, content[:200])
    except httpx.HTTPStatusError as e:
        logger.error("[DOC-QA]   synthesis pass: HTTPStatusError %s  (%.0fms)",
                     e.response.status_code, (time.perf_counter() - t0) * 1000)
        content = ""
    if "<tool_call" in content or "<function=" in content:
        logger.warning("[DOC-QA]   synthesis pass: model emitted fake tool call in plain text — treating as no answer")
        content = ""
    yield "tools", tool_trace
    if not content:
        logger.warning("[DOC-QA] ═══ answer_events END (exhausted, no answer) ═══  %.0fms",
                       (time.perf_counter() - t_total) * 1000)
        yield "answer", "I couldn't find that in the reports after several lookups."
        return
    bad = _unverified_numbers(content, seen, question)
    if bad:
        logger.warning("[DOC-QA]   synthesis answer has unverified numbers: %s", bad)
        yield "warning", f"Not matched to a tool result: {', '.join(bad)}. Treat those number(s) with caution."
    logger.info("[DOC-QA] ═══ answer_events END (exhausted, synthesized) ═══  %.0fms  answer='%s'",
                (time.perf_counter() - t_total) * 1000, content[:200])
    yield "answer", content
