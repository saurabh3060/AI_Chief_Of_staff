"""Isolated retrieval pipeline: embed search only, no facts_view, no metadata filters, no
run_sql/run_metric against the library (see embedqa.py for that fuller hybrid pipeline). The one
tool here is a plain calculator, since the system prompt requires arithmetic to go through a tool
rather than the model guessing.

  1. Embed the question.
  2. Hybrid-search the blocks: BM25-style ``ts_rank`` over the ``blocks.tsv`` tsvector column
     + cosine over Weaviate's ``DocLibraryBlocks`` collection, fused by Reciprocal Rank Fusion,
     then cut to the top k by the cross-encoder reranker when one is loaded.
  3. Build one prompt: system prompt (fixed) + retrieved passages + question.
  4. Ask the chat model; if it calls calculate, evaluate and loop once more; else return the text.

Storage: PostgreSQL (row data + full-text) via asyncpg; Weaviate for vectors.  No SQLite.
"""
import asyncio
import json
import re

import asyncpg

from . import calc
from . import config
from . import fintools
from . import glossary
from . import llm
from . import periods
from . import generic_metrics
from . import store


def _strip_think(text: str) -> str:
    """Granite's chat template sometimes leaks a reasoning block into content even with
    enable_thinking off; keep only what follows the last close tag, as the template itself does
    for older turns."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()

SYSTEM_PROMPT = """You answer financial questions using only the provided context, in the voice of
a data analyst briefing a CEO — a verdict in 1-3 sentences, not a data dump.

If the question is instead a follow-up about your own previous reply in this conversation
(repeat, rephrase, clarify, summarize, "what did you just say") rather than a new financial
lookup, answer it directly from the conversation above — do not treat missing context/evidence
as a reason to refuse that kind of question.

Context may contain TEXT, TABLES, and CHART DATA.

Follow this process:

1. Identify:
   - metric/entity
   - period
   - requested operation

2. Find the minimum relevant evidence.

3. Prefer evidence in this order:
   structured table/chart values > explicit text values > narrative explanation.
   A value from CHART context is read off a chart, not printed — prefix it with "≈" in your
   answer (e.g. "≈58%"); a TEXT/TABLE value is printed as-is, state it plainly, no "≈".

4. If the context has a "Computed from the reports by code" block, it already answers the
   calculation: take the numbers, periods and names from it exactly as written. Otherwise, if
   the answer needs a difference, trend, peak/low, total, average, share or ranking, call
   analyze_metric with the metric name(s) as the context prints them; for any other arithmetic,
   use calculate. Never do arithmetic yourself.
   Name the measure from the computed result; do not mistake a page or section heading for the
   measure. Include its first value, last value, and total change when those are provided.

5. For "why", "driver", or "trend" questions:
   - trend: describe the shape once — start, peak/trough, latest — each anchored to its number
     (e.g. "rose from 120 in Jan to a peak of 180 in Jun, then eased to 150 by Sep before
     recovering to 170 in Dec"). Never list every period-over-period change one by one.
   - why/driver: combine the numeric evidence with the supporting text/chart driver, and say
     whose explanation it is — the report's own stated reason, or your inference from the numbers.

6. If the context includes the report's own narrative callout or claim about a number, check it
   against the underlying data before repeating it — say plainly whether it holds up, holds up
   but is incomplete, or isn't fully explained by what you were given.

7. Check:
   - same entity/scope
   - same unit/currency
   - correct period
   - % versus percentage points

8. Answer in 1-3 sentences:
   - verdict first — a direct "Yes —", "No, not by itself.", or the headline number/name, not a
     restated question,
   - the key number(s) behind it, with period and "≈" if approximate,
   - one clause of explanation only if it adds something non-obvious (a driver, a caveat, a
     comparison),
   - source/page when available.

Never invent missing values or reasons, and never add a claim you weren't given just because a
confident answer would sound better with it. If the context only partly answers the question,
say exactly what's missing or out of scope, after answering the part you can from what's there —
not a blanket "insufficient evidence" in place of using what you do have."""

TOOL_SPECS = [fintools.TOOLS["analyze_metric"].spec, calc.TOOL_SPEC]

# Wording that asks for arithmetic over reported values. A 3B model often answers such questions
# from the passages by eye instead of calling a tool (and llama-server ignores tool_choice
# "required" for granite's XML tool format), so for these code runs analyze_metric up front from a
# JSON plan the model fills in, and the answer is written from its results.
ANALYTIC_RE = re.compile(
    r"(?i)\b(?:vs\.?|versus|compared?|comparison|change[sd]?|trend(?:ed|s)?|grew|grow(?:th|ing)?|"
    r"increase[ds]?|decrease[ds]?|decline[ds]?|rose|rise|fell|fall|drop(?:ped)?|average|avg|mean|"
    r"sum|share|proportion|peak|highest|lowest|best|worst|sharpest|fastest|slowest|largest|"
    r"smallest|most|least|rank|mix|breakdown|broken down|efficien(?:cy|t)|shrink(?:ing|s)?|"
    r"expand(?:ing|s)?|contract(?:ing|s)?|net trend|"
    r"month to month|over the (?:last|past))\b")
_NONE = "none of these"

_LAST_N_RE = re.compile(r"(?i)\b(?:last|past|latest|previous)\s+(\d{1,2})\s+(?:months?|quarters?|years?|periods?)\b")
_MANY_RE = re.compile(r"(?i)\b(?:share|proportion|split|each|every|which|what)\b.*\b(?:channels?|products?|"
                      r"segments?|buckets?|categories|bands?|heads?|cohorts?)\b|\beach\b")


def _operation(question: str, n_periods: int) -> str:
    """The calculation a question's wording asks for — fixed English patterns, not report content."""
    q = question.lower()
    if re.search(r"\b(?:share|proportion|split|mix)\b", q):
        return "share"
    category = re.search(r"\b(?:channels?|products?|segments?|buckets?|categories|bands?|heads?|cohorts?)\b", q)
    if re.search(r"\b(?:fastest|slowest|rank)\b", q) or (
            re.search(r"\b(?:which|what|where|specific)\b", q) and category
            and not re.search(r"\b(?:month|quarter|year|period)\b", q)):
        # "Where is it highest?" asks for the latest cross-sectional level. Growth/rate-of-change
        # wording still uses rank, which compares the first and last selected periods.
        if (re.search(r"\b(?:where|specific|best|worst|highest|lowest|current|latest|broken down|breakdown)\b", q)
                and not re.search(r"\b(?:change|grew|growth|increase|decrease|decline|fastest|slowest)\b", q)):
            return "rank_latest"
        return "rank"
    if re.search(r"\b(?:average|avg|mean)\b", q):
        return "average"
    if re.search(r"\bsum\b|\bcumulative\b", q):
        return "sum"
    if n_periods == 2 and re.search(r"\b(?:vs\.?|versus|compared?|change|from)\b", q):
        return "change"
    return "trend"


# "from X to Y" / "between X and Y" wording (the gap between the two words is bounded) or
# "through" / "till"; the question is capped before matching (py/polynomial-redos).
_MAX_REGEX_INPUT_LEN = 10000
_RANGE_WORDING_RE = re.compile(
    r"(?i)\bfrom\b[^\n]{0,300}\bto\b|\bbetween\b[^\n]{0,300}\band\b|\bthrough\b|\btill\b"
)


def _periods(question: str) -> tuple[str | None, int]:
    """The periods the question names, as analyze_metric's `periods` string, and how many."""
    found = [question[s:e] for _, (s, e) in periods.find_all(question)]
    if not found:
        return None, 0
    if len(found) == 2 and _RANGE_WORDING_RE.search(question[:_MAX_REGEX_INPUT_LEN]) and not re.search(
            r"(?i)\bvs\.?\b|\bversus\b", question):
        return f"{found[0]} to {found[1]}", 2
    return ", ".join(found), len(found)


# A per-unit row ("Spends/Card/Day", "Cost per account"): only wanted when the question asks for one.
_PER_UNIT_RE = re.compile(r"(?i)/\s*(?:card|day|month|week|year|account|customer|user|unit|head)s?\b|\bper\b")


_STEM_STOP = {"the", "this", "that", "with", "from", "what", "which", "where", "does", "show",
              "about", "over", "past", "month", "months", "year", "years", "page", "total"}


def _stems(text: str) -> set[str]:
    """Short financial tokens such as CIF, B3 and NCA are meaningful and must survive matching."""
    return {w[:5] for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in _STEM_STOP}


async def _planned_lookup(conn, question: str, model: str, pages: list[int], doc_ids) -> dict | None:
    """For a question that needs arithmetic, code reads the operation and periods off the wording,
    the model only picks which row(s) from the ones stored on the retrieved pages (a JSON enum, so
    it can't invent a name; thinking off, since with it on granite never emits the JSON), and
    code runs analyze_metric. None when the question isn't analytic or no plan could be made."""
    if not ANALYTIC_RE.search(question):
        return None
    stored = await fintools.metric.rows_on_pages(conn, pages, doc_ids)
    if not stored:
        return None
    period_text, n_periods = _periods(question)
    m = _LAST_N_RE.search(question)
    last_n = int(m.group(1)) if m else None
    operation = _operation(question, n_periods)
    if operation == "rank" and n_periods == 2 and " to " in (period_text or ""):
        period_text = period_text.replace(" to ", ", ")  # rank compares the two endpoints
    many = operation in ("share", "rank", "rank_latest")
    named: list[str] = []
    if many:
        # Share/rank compare the rows of one table or chart (its channels, products...), so the
        # model picks the table and code takes its rows, leaving out the total row.
        groups = list(dict.fromkeys(g for g, row in stored if g and not fintools.metric.is_total(row)))
        choices, ask = groups, ("Pick the table or chart whose rows the question compares "
                                "(e.g. the one listing the channels, products or buckets it names).")
        listing = "\n".join(f"- {g}: " + ", ".join(r for gg, r in stored if gg == g) for g in groups)
        # Only a table the question itself names ("each channel" -> "Channel Mix"): computing a
        # share over whatever table was retrieved turned "What's our market share?" into a
        # made-up figure from an unrelated cost table.
        choices = [g for g in groups if _stems(g.split(" › ")[-1]) & _stems(question)]
        # Common finance abbreviation: COC means cost of collection. Resolve the terminology,
        # not a document-specific value, so this works on any report using that standard label.
        if re.search(r"(?i)\b(?:collection\s+cost|cost\s+of\s+collection)\b", question):
            coc_choices = [g for g in groups if re.search(r"(?i)\bCOC\b", g)]
            choices = coc_choices or choices
        # Questions about efficiency/rates should compare a percentage/rate table rather than an
        # absolute balance or cost table when both are present on the same page.
        if re.search(r"(?i)\b(?:efficien(?:cy|t)|rate|ratio|percent|percentage|%)\b", question):
            rate_choices = [g for g in choices if re.search(r"(?i)(?:%|rate|ratio|resolution)", g)]
            choices = rate_choices or choices
        listing = "\n".join(f"- {g}: " + ", ".join(r for gg, r in stored if gg == g) for g in choices)
    else:
        choices = [f"{g} › {r}" if g else r for g, r in stored]
        if not _PER_UNIT_RE.search(question):
            choices = [c for c in choices if not _PER_UNIT_RE.search(c.split(" › ")[-1])] or choices
        # Rows whose own name the question uses; the model only chooses among these when there
        # are several (given the whole list plus "none", it declined even an exact name match).
        # A complete printed row name in the question is stronger than a shared stem. This keeps
        # "total CIF" on the CIF row instead of the neighbouring Net Addition row.
        qnorm = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
        exact_named = [c for c in choices
                       if " ".join(re.findall(r"[a-z0-9]+", c.split(" › ")[-1].lower())) in qnorm]
        named = list(dict.fromkeys(exact_named or [
            c for c in choices if _stems(c.split(" › ")[-1]) & _stems(question)]))
        if len(named) > 1:
            # Prefer rows matched by a word few row names share ("proprietary" over "sourcing",
            # which also names a channel and a text line).
            counts: dict[str, int] = {}
            for c in choices:
                for s in _stems(c.split(" › ")[-1]):
                    counts[s] = counts.get(s, 0) + 1
            score = {c: sum(1 for s in _stems(c.split(" › ")[-1]) & _stems(question) if counts[s] == 1)
                     for c in named}
            best = max(score.values())
            if best > 0:
                named = [c for c in named if score[c] == best]
        if named:
            choices = named
        ask = (f"Pick the one row that holds the measure the question asks about, or \"{_NONE}\" if "
               "no row is that measure.")
        listing = "\n".join(f"- {c}" for c in choices)
    if not choices:
        return None
    schema = {"type": "object", "additionalProperties": False, "required": ["pick"],
              "properties": {"pick": {"type": "string", "enum": choices + ([] if many else [_NONE])}}}
    if len(choices) == 1 and (many or named):
        pick = choices[0]
    else:
        pick = _pick_with_model(model, ask, listing, question, schema)
    if pick not in choices:
        return None
    if many:
        rows = [r for g, r in stored if g == pick and not fintools.metric.is_total(r)]
        if operation == "rank_latest":
            # Some tables show an aggregate beside its children (B3 beside B3-SP/B3-Agency).
            # Compare one hierarchy level: if the aggregate exists, omit its hyphenated children.
            row_names = {r.casefold() for r in rows}
            rows = [r for r in rows if "-" not in r or r.rsplit("-", 1)[0].casefold() not in row_names]
        metrics = [f"{pick} › {r}" for r in rows]
    else:
        metrics = [pick]
    args = {"metrics": metrics, "operation": operation}
    if period_text and not last_n:
        args["periods"] = period_text
    if last_n:
        args["last_n"] = last_n
    result = await fintools.dispatch(
        "analyze_metric", {**args, "doc_ids": doc_ids, "prefer_pages": pages}, conn)
    return {"tool": "analyze_metric", "args": args, "result": result}


def _pick_with_model(model: str, ask: str, listing: str, question: str, schema: dict) -> str | None:
    try:
        msg = llm.chat_message(model, [
            {"role": "system", "content": f"{ask} Reply as JSON: {{\"pick\": \"...\"}}."},
            {"role": "user", "content": f"Stored on the pages found for this question:\n{listing}\n\n"
                                        f"Question: {question}"},
        ], temperature=0, max_tokens=120, chat_template_kwargs={"enable_thinking": False},
            response_format={"type": "json_schema", "json_schema": {"name": "pick", "strict": True, "schema": schema}})
        return json.loads(_strip_think(msg.get("content") or "") or "{}").get("pick")
    except Exception:
        return None


def _computed_brief(question: str, result: dict) -> str:
    """A deterministic reading aid for small models. It contains no report-specific constants:
    every name and number comes from analyze_metric's result."""
    if result.get("metric") and result.get("first") and result.get("last"):
        first, last = result["first"], result["last"]
        bits = [f"{result['metric']} moved from {first.get('value')} in {first.get('period')} to "
                f"{last.get('value')} in {last.get('period')}"]
        if result.get("total_change") is not None:
            bits.append(f"total change {result['total_change']}")
        if result.get("total_pct_change") is not None:
            bits.append(f"{result['total_pct_change']}")
        if result.get("increases") is not None:
            bits.append(f"{result['increases']} increases and {result.get('decreases', 0)} decreases")
        return "; ".join(bits) + "."
    ranked = result.get("ranked") or []
    if ranked:
        top = ranked[0]
        brief = f"The highest latest value is {top.get('metric')} at {top.get('value')} in {result.get('period')}."
        if re.search(r"(?i)\b(?:cost|expense)\b", question) and re.search(
                r"(?i)\b(?:worst|breakdown|broken down|inefficien)\b", question):
            brief += " For this cost-efficiency question, the highest cost percentage is the worst result."
        return brief
    return ""


_MAX_ROUNDS = 4

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
    driven by a sharp decline in CIF due to reduced export activity..."). That fabricated text
    then becomes the search query, which is worse than just searching on the original unresolved
    question: it steers retrieval toward invented specifics instead of the real ones. A genuine
    rewritten question always ends in "?" and stays question-length; a fabricated answer is a
    longer declarative sentence — both observed failures were 190+ chars ending in a period, so
    this check is deliberately loose (only rules out the answer-shaped case), not a strict
    question-quality filter."""
    text = text.strip()
    return bool(text) and text.endswith("?") and len(text) <= 250


def _resolve_followup(question: str, history: list[dict] | None, model: str, reasoning: bool = False) -> str:
    """Expands an elliptical follow-up ("what about FY25?") into a self-contained question using the
    prior exchange, so the passage search below isn't run on a query that omits the company/metric/
    period it actually refers to. The original `question` is still what's sent as the user's turn
    to the chat model; this only steers search. No-op on the first turn, if the call fails, or if
    the rewrite doesn't look like a question (see _looks_like_a_question) — in each case, fall back
    to the original question rather than risk retrieval on a fabricated one."""
    turns = [h for h in (history or []) if h.get("role") in ("user", "assistant") and h.get("content")]
    if not turns:
        return question
    try:
        resolved = llm.chat(model, [
            {"role": "system", "content": RESOLVE_SYSTEM},
            *turns[-2:],
            {"role": "user", "content": question},
        ], temperature=0, max_tokens=1500 if reasoning else 80,
           chat_template_kwargs={"enable_thinking": reasoning})
        resolved = _strip_think(resolved).strip('"')
        return resolved if resolved and _looks_like_a_question(resolved) else question
    except Exception:
        return question


_RRF_K = 60  # standard smoothing constant; keeps one method's rank-1 hit from dominating

_rerank_available = True  # cleared after the first failure, see _rerank


async def _fts_ids(conn: asyncpg.Connection, question: str, k: int,
                   doc_ids: list[str] | None = None) -> list[int]:
    """Best-matching block ids by Postgres full-text search over ``blocks.tsv``.
    Each word becomes an OR'd lexeme; punctuation is stripped so Postgres tsquery
    operators cannot leak in from the question."""
    terms = re.findall(r"[A-Za-z0-9]+", question)
    if not terms:
        return []
    tsquery_text = " | ".join(terms)
    sql = ("SELECT block_id FROM blocks "
           "WHERE tsv @@ to_tsquery('english', $1)")
    args: list = [tsquery_text]
    if doc_ids:
        marks = ",".join(f"${i + 2}" for i in range(len(doc_ids)))
        sql += f" AND doc_id IN ({marks})"
        args.extend(doc_ids)
    sql += (f" ORDER BY ts_rank(tsv, to_tsquery('english', $1)) DESC "
            f"LIMIT ${len(args) + 1}")
    args.append(k)
    rows = await conn.fetch(sql, *args)
    return [r["block_id"] for r in rows]


async def _vec_ids(qvec: list[float], k: int,
                   doc_ids: list[str] | None = None) -> list[int]:
    """Vector search via the active backend (Weaviate or FAISS)."""
    vec = store._vec()
    hits = await asyncio.to_thread(vec.search_blocks, qvec, k, doc_ids)
    return [bid for bid, _ in hits]


async def _fused_ids(conn: asyncpg.Connection, question: str, qvec: list[float], k: int,
                     doc_ids: list[str] | None = None) -> list[int]:
    """Vector + keyword hybrid, combined by Reciprocal Rank Fusion."""
    vec_ranked = await _vec_ids(qvec, k, doc_ids)
    fts_ranked = await _fts_ids(conn, question, k, doc_ids)
    scores: dict[int, float] = {}
    for ranked in (vec_ranked, fts_ranked):
        for rank, i in enumerate(ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def _rerank(question: str, rows: list[dict], k: int) -> list[dict]:
    """Cross-encoder rerank of RRF's fused candidates down to the top k. config.RERANK_MODEL is
    optional, so with no reranker loaded this has to degrade quietly to RRF order — and stay
    degraded, since retrying a request that will fail costs time on every later question."""
    global _rerank_available
    if len(rows) <= k or not _rerank_available or not config.RERANK_MODEL:
        return rows[:k]
    rows = rows[:config.RERANK_CANDIDATES]
    try:
        scores = llm.rerank(question, [(r["text"] or "")[:config.RERANK_CHARS] for r in rows])
    except Exception:
        _rerank_available = False
        return rows[:k]
    return [r for r, _ in sorted(zip(rows, scores), key=lambda rs: rs[1], reverse=True)][:k]


async def search(conn: asyncpg.Connection, question: str, k: int = config.PASSAGES_K,
                 doc_ids: list[str] | None = None, page: int | None = None) -> list[dict]:
    """Top-k blocks for the question by hybrid search, optionally scoped to doc_ids."""
    if page is not None:
        sql = ("SELECT b.block_id, b.doc_id, d.file_name, b.page, b.kind, b.text "
               "FROM blocks b JOIN documents d ON d.doc_id = b.doc_id "
               "WHERE b.page = $1")
        args: list = [page]
        if doc_ids:
            marks = ",".join(f"${i + 2}" for i in range(len(doc_ids)))
            sql += f" AND b.doc_id IN ({marks})"
            args.extend(doc_ids)
        sql += " ORDER BY b.block_id"
        rows = await conn.fetch(sql, *args)
        return _rerank(question, [dict(r) for r in rows], k)

    qvec = llm.embed(["search_query: " + question])[0]
    over = k * 5 if doc_ids else max(k * 3, 15)
    ids = await _fused_ids(conn, question, qvec, over, doc_ids)
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
    return _rerank(question, [by_id[i] for i in ids if i in by_id], k)


def format_context(passages: list[dict]) -> str:
    if not passages:
        return "(no passages found)"
    return "\n\n".join(f"[{(p['kind'] or 'text').upper()}] p.{p['page']} ({p['file_name']}):\n{p['text']}"
                       for p in passages)


async def page_issues(conn: asyncpg.Connection, doc_ids: list[str] | None, page: int | None) -> str:
    if page is None:
        return ""
    sql = "SELECT kind, ref, message, suggestion FROM issues WHERE page = $1"
    args: list = [page]
    if doc_ids:
        marks = ",".join(f"${i + 2}" for i in range(len(doc_ids)))
        sql += f" AND doc_id IN ({marks})"
        args.extend(doc_ids)
    sql += " ORDER BY issue_id"
    rows = await conn.fetch(sql, *args)
    if not rows:
        return ""
    lines = ["\nIngestion validation flags for this page:"]
    for row in rows:
        line = f"- {row['kind']} — {row['ref']}: {row['message']}"
        if row["suggestion"]:
            line += f"; suggested review: {row['suggestion']}"
        lines.append(line)
    return "\n".join(lines)


async def answer_events(question: str, history: list[dict] | None = None, doc_ids: list[str] | None = None,
                        model: str = config.CHAT_MODEL, k: int = config.PASSAGES_K, reasoning: bool = False,
                        page: int | None = None, page_title: str = ""):
    """Async generator yielding ("stage"|"passages"|"tools"|"answer", value)."""
    async with store.acquire() as conn:
        resolved = await generic_metrics.resolve(conn, question, doc_ids=doc_ids, page=page, limit=12)
        if resolved:
            yield "stage", "Using validated calculations…"
            yield "passages", []
            yield "tools", [{"tool": "generic_metrics", "args": {"question": question},
                             "result": json.dumps(resolved, ensure_ascii=False)}]
            yield "answer", generic_metrics.deterministic_answer(resolved)
            return
        reviewed = await generic_metrics.reviewed_metric_matches(
            conn, question, doc_ids=doc_ids, page=page, limit=3
        )
        if reviewed:
            yield "stage", "Checking business metric validation…"
            yield "passages", []
            yield "tools", [{"tool": "generic_metrics_review", "args": {"question": question},
                             "result": json.dumps(reviewed, ensure_ascii=False)}]
            yield "answer", generic_metrics.reviewed_metric_answer(reviewed)
            return
        yield "stage", "Searching the reports…"
        search_question = _resolve_followup(question, history, model, reasoning=reasoning)
        glossary_matches = glossary.match(search_question)
        retrieval_question = glossary.expand(search_question, glossary_matches)
        if page is not None:
            retrieval_question = f"{page_title or 'Source page'} (page {page}): {retrieval_question}"
        passage_k = max(k, 4) if page is not None or glossary_matches else k
        passages = await search(conn, retrieval_question, k=passage_k,
                                doc_ids=doc_ids, page=page)
        if page is None:
            guided: list[dict] = []
            categories = list(dict.fromkeys(
                e["category"] for e in glossary_matches if e["kind"] == "formula" and e["category"]))[:2]
            for category in categories:
                guided.extend(await search(conn, category, k=1, doc_ids=doc_ids))
            combined = guided + passages
            passages = list({p["block_id"]: p for p in combined}.values())[:max(passage_k, len(guided) + k)]
        yield "passages", passages

        pages = sorted({p["page"] for p in passages})
        precomputed = await generic_metrics.search(conn, question, doc_ids, page=page, limit=12)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history or []:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        tool_trace: list[dict] = []
        computed = ""
        if precomputed:
            computed = ("\n\nPRECOMPUTED AND VALIDATED BUSINESS METRICS. For calculation questions, use only "
                        "matching results from this block; never recalculate or substitute a raw passage value:\n"
                        + generic_metrics.context(precomputed))
            tool_trace.append({"tool": "generic_metrics", "args": {"question": question},
                               "result": json.dumps(precomputed, ensure_ascii=False)})
        planned = None if precomputed else await _planned_lookup(conn, question, model, pages, doc_ids)
        prompt_passages = passages
        if planned and "error" not in planned["result"]:
            tool_trace.append({**planned, "result": json.dumps(planned["result"], ensure_ascii=False)})
            brief = _computed_brief(question, planned["result"])
            computed = ("\n\nAuthoritative computation from the reports (use this result; do not substitute "
                        "numbers from another table):\n"
                        + json.dumps(planned["result"], ensure_ascii=False)
                        + (f"\nDirect reading of that computation: {brief}" if brief else ""))
            groups = {m.split(" › ")[0] for m in planned["args"].get("metrics", []) if " › " in m}
            relevant = [p for p in passages if any(g.lower() in (p.get("text") or "").lower() for g in groups)]
            if relevant:
                prompt_passages = relevant
        hint = f"\nSource page: p.{page} — {page_title}." if page is not None else ""
        issues = await page_issues(conn, doc_ids, page)
        glossary_hint = glossary.context(glossary_matches)
        messages.append({"role": "user", "content": f"Context:\n{format_context(prompt_passages)}{computed}"
                                                    f"{hint}{issues}{glossary_hint}\n\nQuestion: {question}"})

        # "low": full thinking spent all 3000 tokens deliberating and returned no answer.
        think_kwargs = {"enable_thinking": reasoning}
        if reasoning:
            think_kwargs["reasoning_effort"] = "low"
        max_tokens = 3000 if reasoning else 500

        for _ in range(_MAX_ROUNDS):
            available_tools = [] if precomputed else TOOL_SPECS
            msg = llm.chat_message(model, messages, tools=available_tools, temperature=0, max_tokens=max_tokens,
                                   chat_template_kwargs=think_kwargs)
            calls = msg.get("tool_calls") or []
            content = _strip_think(msg.get("content") or "")
            if not calls and not content and think_kwargs.get("enable_thinking"):
                think_kwargs, max_tokens = {"enable_thinking": False}, 500
                continue
            if not calls:
                yield "tools", tool_trace
                yield "answer", content
                return
            yield "stage", "Calculating…"
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for call in calls:
                name = call["function"].get("name", "")
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "analyze_metric":
                    result = await fintools.dispatch(
                        name, {**args, "doc_ids": doc_ids, "prefer_pages": pages}, conn)
                else:
                    name, result = "calculate", calc.calculate(args.get("expression", ""))
                tool_trace.append({"tool": name, "args": args, "result": json.dumps(result, ensure_ascii=False)})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})

        text = _strip_think(llm.chat(model, messages + [
            {"role": "user", "content": "Give your final answer now, without calling any tool."}], temperature=0,
            max_tokens=max_tokens, chat_template_kwargs=think_kwargs))
        yield "tools", tool_trace
        yield "answer", text


async def answer(question: str, history: list[dict] | None = None, doc_ids: list[str] | None = None,
                 model: str = config.CHAT_MODEL, k: int = config.PASSAGES_K, reasoning: bool = False,
                 page: int | None = None, page_title: str = "") -> tuple[str, list[dict]]:
    """Async convenience wrapper over ``answer_events``, for scripts/CLI."""
    passages: list[dict] = []
    text = ""
    async for kind, val in answer_events(question, history, doc_ids=doc_ids, model=model, k=k,
                                          reasoning=reasoning, page=page, page_title=page_title):
        if kind == "passages":
            passages = val
        elif kind == "answer":
            text = val
    return text, passages


if __name__ == "__main__":
    import sys
    from core.db import init_db, close_db

    async def _cli():
        await init_db()
        try:
            q = " ".join(sys.argv[1:]) or input("Question: ")
            reply, found = await answer(q)
            print(f"\n--- {len(found)} passage(s) ---")
            for p in found:
                print(f"p.{p['page']} ({p['file_name']}) [{p['kind']}]")
            print("\n--- answer ---")
            print(reply)
        finally:
            await close_db()

    asyncio.run(_cli())
