"""Database-driven calculation rules for any page and any number of PDFs.

Business input is the six-column table used in the project:
  # | Metric family | Pipeline-ready metric | Definition / formula | Applies to | Example

Supported formulas are compiled into a safe arithmetic expression containing fact references,
for example ``RR(CB,m) - RR(CB,m-1)`` or ``A(p,m) / B(p,m) * 100``.  Unsupported or ambiguous
rows fail closed and are never materialized.  No eval() or generated Python is used.

CLI examples:
  pbpaste | python generic_metrics.py import - --page 18 --source Report.pdf --dry-run
  python generic_metrics.py import metrics.xlsx --page-column Page --apply
  python generic_metrics.py materialize --all
  python generic_metrics.py list
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import metric_bindings
from . import periods as period_parser
from . import store
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")

# ── Regex safety (CodeQL py/polynomial-redos) ─────────────────
# User question text reaches the patterns below, so every pattern here uses
# bounded repetition (no open-ended \s*, \w*, .+? next to another quantifier)
# and the text is capped at MAX_REGEX_INPUT_LEN before matching. Real questions
# are a few hundred characters; the bounds only matter for hostile input.
MAX_REGEX_INPUT_LEN = 10000
_MAX_WORD_LEN = 20       # longest word-like run matched after a month stem such as "apr"
_MAX_GAP_LEN = 5         # longest run of whitespace between the parts of a phrase
_MAX_WINDOW_LEN = 200    # longest period-window phrase captured by "between X and Y"
# "ytd", "year to date" and an Apr-to-Nov style range.
_YTD_RANGE_RE = re.compile(
    r"(?i)\bytd\b|year.to.date|apr\w{0,%d}\s{0,%d}(?:-|–|to)\s{0,%d}nov"
    % (_MAX_WORD_LEN, _MAX_GAP_LEN, _MAX_GAP_LEN)
)
# "between <window> and <window>"; the second window ends at ? . ! or end of line.
_COMPARISON_WINDOWS_RE = re.compile(
    r"(?i)\bbetween\s{1,%(gap)d}([^\n]{1,%(win)d}?)\s{1,%(gap)d}(?:and|vs\.?|versus)"
    r"\s{1,%(gap)d}([^\n]{1,%(win)d}?)(?:[?.!]|$)" % {"gap": _MAX_GAP_LEN, "win": _MAX_WINDOW_LEN}
)
# A half-year token such as "H1" / "h 2" (the caller strips the text first).
_HALF_YEAR_RE = re.compile(r"(?i)h\s{0,%d}([12])" % _MAX_GAP_LEN)
# Lead-in phrase of a "what is X" style question; X is whatever follows it.
_ASKED_PREFIX_RE = re.compile(r"\b(?:what is|what was|what were|give me|show me|tell me)\s")


ENGINE_VERSION = 1
BUSINESS_WORKBOOKS = {
    "01_Book_Growth_Analysis_4786.pdf": "business_metrics_source/01_Book_Growth_Analysis_Metrics.xlsx",
    "02_Overall_and_Segment_wise_MAB_5144.pdf": "business_metrics_source/02_Overall_Segment_MAB_Metrics.xlsx",
    "03_Interest_Rate_Sensitivity_Deposits_Maturity_4354.pdf": "business_metrics_source/03_IRS_Deposits_Maturity_Metrics.xlsx",
    "04_New_Originations_Month_Wise_1924.pdf": "business_metrics_source/04_New_Originations_Metrics.xlsx",
    "05_Disb_MIS_3363.pdf": "business_metrics_source/05_Disb_MIS_Metrics.xlsx",
    "06_WB_Balance_Sheet_9623.pdf": "business_metrics_source/06_WB_Balance_Sheet_Metrics.xlsx",
    "07_Product_Wise_9556.pdf": "business_metrics_source/07_Product_Wise_Metrics.xlsx",
    "08_SMA_and_Collection_Efficiency_5659.pdf": "business_metrics_source/08_SMA_Collection_Efficiency_Metrics.xlsx",
    "09_Net_Advances_Insights_7222.pdf": "business_metrics_source/RBL_Net_Advances_Metrics_Pagewise.xlsx",
    "10_Credit_Portfolio_Overview_2564.pdf": "business_metrics_source/10_Credit_Portfolio_Overview_Metrics.xlsx",
    "CC12.pdf": "page*_metrics.tsv",
}


def document_family(file_name: str, title: str = "") -> str:
    """Stable identity for recurring reports; dates, delivery order and copy suffixes are ignored."""
    stem = Path(file_name or "").stem.lower().replace("liablity", "liability")
    stem_words = re.sub(r"[^a-z0-9]+", " ", stem).strip()
    # Filename-specific identities take precedence over broad words in a report title.
    filename_families = (
        (r"^01 book growth analysis\b", "book_growth_analysis"),
        (r"^02 overall and segment wise mab\b", "overall_segment_mab"),
        (r"^03 interest rate sensitivity deposits maturity\b", "interest_rate_sensitivity"),
        (r"^04 new originations month wise\b", "new_originations"),
        (r"^05 disb mis\b", "retail_disbursement_mis"),
        (r"^06 wb balance sheet\b", "wb_balance_sheet"),
        (r"^07 product wise\b", "product_wise_monitoring"),
        (r"^08 sma and collection efficiency\b", "sma_collection_efficiency"),
        (r"^10 credit portfolio overview\b", "credit_portfolio_overview"),
        (r"\bcollection daily refresh\b", "collection_daily_refresh"),
        (r"\bnet advances insights\b", "net_advances_insights"),
        (r"\boverall advances\b", "overall_advances"),
        (r"\bliability summary\b", "liability_summary"),
        (r"\bretail assets collections efficiency\b", "retail_assets_collections_efficiency"),
        (r"^cc\d+$", "credit_cards_business_update"),
    )
    for pattern, family in filename_families:
        if re.search(pattern, stem_words):
            return family
    text = re.sub(r"[^a-z0-9]+", " ", f"{stem} {title or ''}".lower()).strip()
    known = (
        (r"^wb\b|\bwholesale banking\b", "wholesale_banking_review"),
        (r"\btreasury\b", "treasury_review"),
        (r"\bhl\s+lap\b", "mortgages_hl_lap_review"),
        (r"\bmortgages?\b(?:\s+lago\b)?", "mortgages_lago_review"),
        (r"\bbbg\b", "bbg_review"),
        (r"\bagri\b", "agri_review"),
        (r"\brbg\b", "rbg_review"),
        (r"\brvf\b|\brural vehicle finance\b", "rvf_review"),
        (r"^cc\d+\b|\bcredit cards?\b", "credit_cards_business_update"),
        (r"\boverall advances\b|\boverview of advances\b", "overall_advances"),
        (r"^mfi\b|\bjlg update\b", "mfi_review"),
    )
    for pattern, family in known:
        if re.search(pattern, text):
            return family
    # Scheduled dashboard exports keep their descriptive stem but change delivery order/suffix.
    base = re.sub(r"^\d{1,2}[\s_-]+", "", stem)
    base = re.sub(r"[_\s-]+\d{4}$", "", base)
    base = re.sub(r"\s*\(\d+\)$", "", base)
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


def document_pattern(file_name: str) -> str:
    family = document_family(file_name)
    return f"family:{family}" if family else ""


_DOCUMENT_EVIDENCE_CACHE: dict = {}
_DOCUMENT_EVIDENCE_CACHE_TS: float = 0.0
_DOCUMENT_EVIDENCE_CACHE_TTL: float = 60.0  # seconds


async def _document_evidence_catalog(conn) -> tuple[dict, dict]:
    """Cache reusable document/row tokens; invalidate after TTL expires.

    Adapted from the SQLite reference which checks file mtime; here we use a simple
    time-based TTL since PostgreSQL has no local file to stat.
    """
    global _DOCUMENT_EVIDENCE_CACHE_TS
    now = time.monotonic()
    cached = _DOCUMENT_EVIDENCE_CACHE.get("value")
    if cached and (now - _DOCUMENT_EVIDENCE_CACHE_TS) < _DOCUMENT_EVIDENCE_CACHE_TTL:
        return cached[0], cached[1]

    documents = {row["doc_id"]: dict(row) for row in await conn.fetch(
        "SELECT doc_id, file_name, title, as_of_key FROM documents WHERE status='ready'"
    )}
    evidence: dict[str, dict] = {
        doc_id: {"tokens": _norm(f"{row['file_name']} {row['title']}"), "rows": []}
        for doc_id, row in documents.items()
    }
    for row in await conn.fetch("SELECT doc_id, label, name, path FROM series"):
        if row["doc_id"] not in evidence:
            continue
        words = _norm(" ".join(str(row[key] or "") for key in ("label", "name", "path")))
        evidence[row["doc_id"]]["tokens"].update(words)
        evidence[row["doc_id"]]["rows"].append(words)
    for row in await conn.fetch(
        "SELECT doc_id, metric_name, metric_family, entity, search_text FROM calculated_metrics "
        "WHERE calculation_status='validated'"
    ):
        if row["doc_id"] not in evidence:
            continue
        words = _norm(" ".join(str(row[key] or "") for key in
                                ("metric_name", "metric_family", "entity", "search_text")))
        evidence[row["doc_id"]]["tokens"].update(words)
        evidence[row["doc_id"]]["rows"].append(words)

    _DOCUMENT_EVIDENCE_CACHE["value"] = (documents, evidence)
    _DOCUMENT_EVIDENCE_CACHE_TS = now
    return documents, evidence


async def infer_document_scope(conn, question: str) -> list[str] | None:
    """Infer an unambiguous report family from portfolio words in an unscoped UI question.

    Page numbers and filenames are delivery details users should not need to know.  When several
    dated reports can answer a portfolio question, use the newest matching document.  Generic
    questions with no portfolio cue remain unscoped rather than being guessed.
    """
    text = re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()

    # A few report identities are also common metric vocabulary.  Honour an explicit report name
    # before evidence scoring so "Overall Advances report" cannot be pulled into a Wholesale
    # review merely because that review contains many rows mentioning advances.
    explicit_report_families = (
        (r"\boverall advances\b", "overall_advances"),
        (r"\bcollection daily refresh\b", "collection_daily_refresh"),
        (r"\bliabilit(?:y|ies) summary\b", "liability_summary"),
        (r"\bretail assets collections efficiency\b", "retail_assets_collections_efficiency"),
    )
    explicit_family = next((family for pattern, family in explicit_report_families
                            if re.search(pattern, text)), None)
    if explicit_family:
        candidates = [dict(row) for row in await conn.fetch(
            "SELECT doc_id, file_name, title, as_of_key FROM documents WHERE status='ready'"
        ) if document_family(row["file_name"], row["title"]) == explicit_family]
        if candidates:
            return [max(candidates, key=lambda row: (str(row["as_of_key"] or ""),
                                                     row["file_name"]))["doc_id"]]

    # FY27 Wholesale questions refer to the forward-looking balance-sheet pack, not to the dated
    # monthly Wholesale review.  The two reports share almost every portfolio word, so their
    # business/report identity is a stronger signal than raw token overlap in this narrow case.
    if re.search(r"\bfy\s*27\b", text) and re.search(r"\b(?:wholesale banking|wb)\b", text):
        candidates = [dict(row) for row in await conn.fetch(
            "SELECT doc_id, file_name, title, as_of_key FROM documents WHERE status='ready'"
        ) if document_family(row["file_name"], row["title"]) == "wb_balance_sheet"]
        if candidates:
            return [max(candidates, key=lambda row: (str(row["as_of_key"] or ""),
                                                     row["file_name"]))["doc_id"]]

    # Prefer the document whose own structured series/calculation catalogue covers the complete
    # question.  This lets newly ingested reports route without adding filename-specific code and
    # prevents a partner/entity shared by reports (for example RFL) from selecting the wrong one.
    concepts = _question_concepts(question)
    if len(concepts) >= 2:
        try:
            documents, evidence = await _document_evidence_catalog(conn)
            asked_periods = _intended_periods(question)
            period_docs: set[str] = set()
            if asked_periods:
                period_docs.update(row[0] for row in await conn.fetch(
                    "SELECT DISTINCT doc_id FROM facts WHERE period_label = ANY($1::text[])",
                    sorted(asked_periods),
                ))
                period_docs.update(row[0] for row in await conn.fetch(
                    "SELECT DISTINCT doc_id FROM calculated_metrics "
                    "WHERE period_label = ANY($1::text[]) OR comparison_period_label = ANY($1::text[])",
                    sorted(asked_periods),
                ))
            ranked = []
            for doc_id, item in evidence.items():
                overlap = len(item["tokens"] & concepts)
                if overlap < 2 or overlap / len(concepts) < 0.75:
                    continue
                best_overlap = max((len(words & concepts) for words in item["rows"]), default=0)
                exact = any(concepts <= words for words in item["rows"])
                title_overlap = len(_norm(
                    f"{documents[doc_id]['file_name']} {documents[doc_id]['title']}"
                ) & concepts)
                score = (overlap * 10 + best_overlap * 6 + title_overlap * 5 + int(exact) * 30
                         + int(doc_id in period_docs) * 45)
                ranked.append((score, int(exact), title_overlap, doc_id))
            ranked.sort(reverse=True)
            if ranked:
                best = ranked[0]
                runner_up = ranked[1] if len(ranked) > 1 else None
                if runner_up is None or best[:3] > runner_up[:3]:
                    return [best[3]]
        except Exception:
            # Gracefully degrade if evidence tables are missing or queries fail.
            pass

    family_patterns = (
        (r"\bhl lap\b", {"mortgages_hl_lap_review"}),
        (r"\b(?:mortgages?|mortgage total)\b", {"mortgages_lago_review"}),
        (r"\b(?:plap|phl|smlap|ahl)\b", {"mortgages_lago_review", "mortgages_hl_lap_review"}),
        (r"\bwholesale banking\b|\bwb\b", {"wholesale_banking_review"}),
        (r"\btreasury\b", {"treasury_review"}),
        (r"\bmicrofinance\b|\bmfi\b", {"mfi_review"}),
        (r"\bbbg\b", {"bbg_review"}),
        (r"\brbg\b", {"rbg_review"}),
        (r"\brvf\b|\brural vehicle finance\b", {"rvf_review"}),
        (r"\bagri\b", {"agri_review"}),
    )
    families = next((values for pattern, values in family_patterns if re.search(pattern, text)), None)
    if not families:
        return None
    candidates = [dict(row) for row in await conn.fetch(
        "SELECT doc_id, file_name, title, as_of_key FROM documents WHERE status='ready'"
    ) if document_family(row["file_name"], row["title"]) in families]
    if not candidates:
        return None

    def recency(row):
        match = re.search(r"\b(20\d{2})[-_ ](\d{2})[-_ ](\d{2})\b", row["file_name"] or "")
        filename_date = "".join(match.groups()) if match else ""
        as_of = re.sub(r"[^0-9]", "", str(row["as_of_key"] or ""))
        return filename_date or as_of, row["file_name"]

    newest = max(candidates, key=recency)
    return [newest["doc_id"]]


PERIOD_VARIABLES = {"m", "t", "v", "q", "d", "qe"}
DIMENSION_VARIABLES = {"p", "s", "r", "x", "k", "b", "g", "group", "segment"}
HEADER_ALIASES = {
    "number": {"#", "number", "no", "id"},
    "family": {"metric family", "family"},
    "name": {"pipeline-ready metric", "pipeline ready metric", "metric", "metric name"},
    "formula": {"definition / formula", "definition/formula", "definition", "formula"},
    "executable_formula": {"executable formula", "engine formula", "compiled formula"},
    "applies": {"applies to", "applies"},
    "example": {"example", "validation example"},
    "output_type": {"output type", "unit", "result unit"},
    "page": {"page", "source page"},
    "aliases": {"aliases", "query aliases"},
    "required_all": {"required terms", "required all", "routing required terms"},
    "required_any": {"required any", "routing required any"},
    "excluded": {"excluded terms", "exclude terms", "routing excluded terms"},
    "period_scope": {"period scope", "routing period", "period kind"},
    "result_period": {"result period", "materialization period", "output period"},
    "result_entity": {"result entity", "output entity", "display entity"},
    "priority": {"priority", "routing priority"},
    "intent_group": {"intent group", "exclusive group", "routing group"},
    "answer_template": {"answer template", "response template"},
    "success_condition": {"success condition", "interpretation condition"},
    "true_text": {"true text", "success text"},
    "false_text": {"false text", "failure text"},
}
SYMBOL_TERMS = {
    "na": ("net addition",),
    "writeoff": ("write off", "write-off"),
}
REF_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]*)\)")
BARE_FACT_SYMBOLS = {
    "OS", "SMA", "NPA", "n_OS", "n_SMA", "n_NPA", "M", "M3", "MI",
    "D_LM", "D_LY", "Stab", "FF", "RB", "Norm", "Dm", "Col", "OP", "Res",
}


def _norm(text: str) -> set[str]:
    text=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",text or "")
    text=re.sub(r"(?i)\bmonth[ -]on[ -]month\b", "mom", text)
    text=re.sub(r"(?i)\bquarter[ -]on[ -]quarter\b", "qoq", text)
    text=re.sub(r"(?i)\byear[ -]on[ -]year\b", "yoy", text)
    text=re.sub(r"(?i)\byo\s+y\b", "yoy", text)
    text=re.sub(r"(?i)\bmo\s+m\b", "mom", text)
    text=re.sub(r"(?i)\bqo\s+q\b", "qoq", text)
    text=re.sub(r"(?i)\bachv?\b|\bachievement\b", "achievement", text)
    text=re.sub(r"(?i)\bq[1-4]\b","quarter",text or "")
    text=re.sub(r"(?i)\bactuals\b","actual",text)
    text=re.sub(r"(?i)\bavg\b","average",text)
    text=re.sub(r"(?i)\b(?:written|wrote)\s+off\b","write off",text)
    canonical = {
        "amt": "amount", "indi": "individual",
        "accounts": "account", "advances": "advance", "balances": "balance",
        "deposits": "deposit", "rates": "rate", "recoveries": "recovery",
        "totals": "total", "buckets": "bucket", "differences": "diff",
        "difference": "diff", "products": "product", "segments": "segment", "three": "3",
    }
    return {canonical.get(x, x) for x in re.findall(r"[a-z0-9]+", text.lower())
            if x not in {"a", "an", "and", "by", "for", "from", "in", "is", "of", "on", "or",
                         "s", "the", "to"}}


_QUESTION_NOISE = {
    "across", "all", "answer", "current", "data", "figure", "give", "latest", "metric",
    "number", "overall", "please", "under", "vs",
    "report", "result", "show", "tell", "value", "what", "when", "where", "which", "who",
    "why", "how", "many", "much", "was", "were", "does", "did",
}
_TIME_WORDS = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun",
    "june", "jul", "july", "aug", "august", "sep", "sept", "september", "oct", "october",
    "nov", "november", "dec", "december", "fy", "month", "quarter", "q1", "q2", "q3",
    "q4", "ytd", "mtd",
}


def _question_concepts(question: str) -> set[str]:
    """Business concepts in a question, excluding words used only to ask or identify a period."""
    concepts = {word for word in _norm(question)
                if word not in _QUESTION_NOISE and word not in _TIME_WORDS
                and not re.fullmatch(r"(?:19|20)?\d{2}|fy\d{2,4}", word)}
    # Business users say "average balance" while compact table headers normally say "AVG".
    # Once "average" is present, "balance" is descriptive rather than a second measure.
    if "average" in concepts:
        concepts.discard("balance")
    return concepts


def _column_map(columns) -> dict[str, str]:
    result = {}
    for column in columns:
        normalized = re.sub(r"\s+", " ", str(column).strip().lower())
        for target, aliases in HEADER_ALIASES.items():
            if normalized in aliases:
                result[target] = column
    return result


def _read_table(path: str) -> list[dict]:
    if path != "-" and Path(path).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        frame = pd.read_excel(path, dtype=str).fillna("")
        return frame.to_dict("records")
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8-sig")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    header_at = next((i for i, line in enumerate(lines)
                      if "metric family" in line.lower() and "pipeline" in line.lower()), None)
    if header_at is None:
        raise ValueError("No calculation-table header was found")
    lines = lines[header_at:]
    if "\t" in lines[0]:
        return list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"))
    if "|" in lines[0]:
        clean = []
        for line in lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                continue
            clean.append("\t".join(cells))
        return list(csv.DictReader(io.StringIO("\n".join(clean)), delimiter="\t"))
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def _entity(token: str) -> str:
    compact = re.sub(r"\s+", "", token.strip().lower())
    if compact in ("entity", "*"):
        return "*"
    return token.strip().upper() if token.strip().isupper() else token.strip()


def _reference(symbol: str, args: str) -> dict:
    parts = [p.strip().replace("−", "-") for p in args.split(",") if p.strip()]
    offset = 0
    period_label = ""
    period_month = 0
    period_quarter = 0
    period_half = 0
    selectors = []
    for part in parts:
        # In the supplied business sheets ``Q-4`` means quarter 4.  Preserve lowercase
        # ``q-1``/``q-2`` as the existing relative-quarter notation.
        literal_quarter = re.fullmatch(r"Q\s*-\s*([1-4])", part)
        if literal_quarter:
            period_quarter = int(literal_quarter.group(1))
            continue
        compact = re.sub(r"\s+", "", part.lower())
        relative = re.fullmatch(r"(m|t|v|q|d|qe)-(\d+)", compact)
        if relative:
            offset = -int(relative.group(2))
            continue
        if compact in PERIOD_VARIABLES or compact in DIMENSION_VARIABLES:
            continue
        if compact == "base":
            raise ValueError("unsupported period selector 'base'")
        # Business workbooks commonly append scenario shorthand to the period (Jan A, Q4 E,
        # FY26 AOP). Keep that scenario as a selector and parse the remaining period normally.
        period_text = part
        status_patterns = (
            (r"(?i)(?:^|\s)(actuals?|act|a)\s*$", "Actual"),
            (r"(?i)(?:^|\s)(budget|bud|b|aop)\s*$", "Budget"),
            (r"(?i)(?:^|\s)(estimate|estimated|est|e|projections?|projected|proj)\s*$", "Estimate"),
        )
        for pattern, status in status_patterns:
            match = re.search(pattern, period_text)
            if match:
                selectors.append(status)
                if match.group(1).lower() == "aop":
                    selectors.append("AOP")
                period_text = period_text[:match.start()].strip()
                break
        # The same scenario markers are also printed before a period in review decks
        # ("Actuals Jan'26", "AOP Mar-26", "Projections Dec-25").
        prefix = re.match(
            r"(?i)^(actuals?|act|budget|bud|aop|estimate|estimated|est|projections?|projected|proj)\b[\s:./-]*",
            period_text,
        )
        if prefix:
            token = prefix.group(1).lower()
            if token.startswith(("actual", "act")):
                selectors.append("Actual")
            elif token in {"budget", "bud", "aop"}:
                selectors.append("Budget")
                if token == "aop":
                    selectors.append("AOP")
            else:
                selectors.append("Estimate")
            period_text = period_text[prefix.end():].strip()
        if re.search(r"(?i)\bytd\b", period_text):
            selectors.append("YTD")
            period_text = re.sub(r"(?i)\bytd\b", "", period_text).strip()
        # In Agri tables "% Vs Mar'25" is the printed column name, not a request for the
        # March-2025 period.  The value itself is stored in the projection/output column.
        if re.match(r"(?i)^%?\s*vs\s+mar(?:ch)?[' -]?\d{2,4}$", period_text):
            selectors.append(period_text)
            continue
        parsed = period_parser.parse(period_text)
        if parsed and re.search(r"\d", period_text):
            period_label = parsed.label
            continue
        # A column heading can combine a measure qualifier and one period, for example
        # "YoY Growth % Feb'26".  Preserve the qualifier as a selector while extracting the
        # unique period.
        period_hits = period_parser.find_all(period_text)
        if len(period_hits) == 1:
            parsed, (start, end) = period_hits[0]
            remainder = re.sub(r"[\s:./-]+", " ", period_text[:start] + " " + period_text[end:]).strip()
            if remainder:
                selectors.append(remainder)
            period_label = parsed.label
            continue
        month = re.fullmatch(
            r"(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
            period_text,
        )
        if month:
            period_month = ("jan feb mar apr may jun jul aug sep oct nov dec".split()
                            .index(month.group(1)[:3].lower()) + 1)
            continue
        quarter = re.fullmatch(r"(?i)q\s*-?\s*([1-4])", period_text)
        if quarter:
            period_quarter = int(quarter.group(1))
            continue
        half = re.fullmatch(r"(?i)h\s*-?\s*([12])", period_text)
        if half:
            period_half = int(half.group(1))
            continue
        selectors.append(part)
    entity = " | ".join(selectors) if selectors else "*"
    encoded_periods = {"smanov25": "Nov-25", "smaoct25": "Oct-25"}
    if not period_label and symbol.lower() in encoded_periods:
        period_label = encoded_periods[symbol.lower()]
    compact_symbol = symbol.lower()
    percentage_measure = any(re.search(r"(?i)\b(?:current|sma[012])\s*rr\b", value)
                             for value in selectors)
    return {"type": "ref", "symbol": compact_symbol, "entity": _entity(entity),
            "selectors": selectors, "period_label": period_label, "period_month": period_month,
            "period_quarter": period_quarter, "period_half": period_half, "period_offset": offset,
            # These generated MFI symbols use percentage facts as spreadsheet fractions and
            # explicitly multiply by 100. Other workbook families use display percentages.
            "percent_fraction": (compact_symbol in {"resolutionrate", "disbursementshare"}
                                 or (compact_symbol in {"smanov25", "smaoct25"}
                                     and percentage_measure))}


def _ast_node(node, refs):
    if isinstance(node, ast.Expression):
        return _ast_node(node.body, refs)
    if isinstance(node, ast.Name) and node.id in refs:
        return refs[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return {"type": "collection", "values": [_ast_node(value, refs) for value in node.elts]}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {"type": "number", "value": float(node.value)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return {"type": "unary", "op": "-" if isinstance(node.op, ast.USub) else "+",
                "value": _ast_node(node.operand, refs)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        return {"type": "binary", "op": {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                                               ast.Pow: "**"}[type(node.op)],
                "left": _ast_node(node.left, refs), "right": _ast_node(node.right, refs)}
    raise ValueError("formula contains an unsupported expression")


def _output_unit(name: str, formula: str, example: str) -> str:
    sample = f"{name} {formula} {example}".lower()
    if " bps" in sample: return "bps"
    if " pp" in sample or "percentage point" in sample: return "pp"
    if "mix" in sample and re.search(r"\b(?:change|difference|delta)\b", sample): return "pp"
    if re.search(r"\d(?:\.\d+)?x\b", sample) or " ratio" in sample: return "x"
    if "%" in sample or re.search(r"\b(?:growth|rate|mix)\b", sample): return "%"
    if " cr" in sample or "₹" in sample: return "Cr"
    if "month" in sample and "streak" in sample: return "months"
    return ""


def compile_formula(name: str, formula: str, example: str = "") -> tuple[str, dict, str]:
    raw = (formula or "").strip()
    # Flatten generated nested notation where the inner call merely names the measure. The
    # outer identifier is the printed source row and the remaining arguments are its selectors.
    raw = re.sub(
        r"(?i)\b(LCSBLCBGforBuyerscredit|BankGuarantee|TotalNFBOS)\s*"
        r"\(\s*NFB\s*OS(?:\s*\(\s*\*\s*\))?\s*,",
        r"\1(", raw,
    )
    raw = re.sub(
        r"(?i)\bResolutionRate\s*\(\s*SMA\s*\(\s*\*\s*\)\s*-\s*([012])\s*,",
        r"ResolutionRate(Bucket\1,", raw,
    )
    raw = re.sub(
        r"(?i)\bResolutionRate\s*\(\s*SMA\s*-\s*([012])\s*,",
        r"ResolutionRate(Bucket\1,", raw,
    )
    # Business notation commonly writes a percentage measure as RES%(t). The percent sign is a
    # unit marker, not an arithmetic operator; the source binding still determines the unit.
    raw = re.sub(r"\b([A-Za-z][A-Za-z0-9_]*)%\s*(?=\()", r"\1", raw)
    for symbol in sorted(BARE_FACT_SYMBOLS, key=len, reverse=True):
        # Single-letter symbols are case-sensitive: lowercase ``m`` is the standard
        # month selector and must never be rewritten as the bare metric ``M``.
        flags = re.I if len(symbol) > 1 else 0
        raw = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_(])",
                     f"{symbol}(*)", raw, flags=flags)
    refs = {}
    def replace(match):
        key = f"R{len(refs)}"
        refs[key] = _reference(match.group(1), match.group(2))
        return key
    expression = REF_RE.sub(replace, raw)
    if not refs:
        raise ValueError("no fact reference could be identified")
    aggregate = re.match(r"(?i)^\s*(sum|total|average|mean|maximum|max|minimum|min|median|standard deviation)\s+(R\d+)\s*$", expression)
    if aggregate:
        op = {"sum":"sum", "total":"sum", "average":"average", "mean":"average", "maximum":"maximum", "max":"maximum",
              "minimum":"minimum", "min":"minimum", "median":"median",
              "standard deviation":"stddev"}[aggregate.group(1).lower()]
        spec = {"type":"aggregate", "op":op, "value":refs[aggregate.group(2)]}
        return op, spec, _output_unit(name, raw, example)
    expression = expression.replace("−", "-").replace("–", "-").replace("÷", "/").replace("×", "*")
    expression = expression.replace("%", "")
    expression = re.sub(r"^\s*[A-Za-zΔ][A-Za-z0-9_Δ̂]*\s*=\s*", "", expression)
    expression = re.sub(r"(?i),?\s*as reported.*$", "", expression).strip()
    expression = re.sub(r"(?i),?\s*(?:in\s+)?(?:pp|bps?|percentage points?)\s*$", "", expression).strip()
    expression = expression.replace("^", "**")
    try:
        parsed = ast.parse(expression, mode="eval")
        spec = _ast_node(parsed, refs)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"unsupported formula syntax: {raw}") from exc
    operation = "fact" if spec["type"] == "ref" else "collection" if spec["type"] == "collection" else "arithmetic"
    return operation, spec, _output_unit(name, raw, example)


def import_rows(path: str, *, default_page: int | None = None, source_name: str = "",
                apply: bool = False, rule_namespace: str = "") -> list[dict]:
    raw_rows = _read_table(path)
    if not raw_rows:
        raise ValueError("The calculation table is empty")
    columns = _column_map(raw_rows[0].keys())
    missing = {"number", "family", "name", "formula", "applies", "example"} - set(columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    compiled = []
    for position, row in enumerate(raw_rows, 1):
        name = str(row.get(columns["name"], "")).strip()
        if not name:
            continue
        number_text = str(row.get(columns["number"], position)).strip()
        number_match = re.search(r"\d+", number_text)
        number = int(number_match.group()) if number_match else position
        page_text = str(row.get(columns.get("page", ""), "")).strip() if "page" in columns else ""
        page = int(float(page_text)) if page_text else default_page
        if page is None:
            raise ValueError(f"Row {number}: page is required (column or --page)")
        formula = str(row.get(columns["formula"], "")).strip()
        executable_formula = (str(row.get(columns["executable_formula"], "")).strip()
                              if "executable_formula" in columns else "")
        example = str(row.get(columns["example"], "")).strip()
        if rule_namespace:
            namespace = hashlib.sha1(rule_namespace.encode("utf-8")).hexdigest()[:10]
            rule_id = f"g_{namespace}_{number:03d}"
        else:
            rule_id = f"g_p{page}_{number:03d}"
        aliases = [name]
        if "aliases" in columns:
            aliases += [x.strip() for x in re.split(r"[;|]", str(row.get(columns["aliases"], ""))) if x.strip()]
        def field(key):
            if key not in columns: return ""
            value=row.get(columns.get(key,""),"")
            return "" if value is None else str(value).strip()
        phrases=lambda key: [x.strip() for x in re.split(r"[;|]",field(key)) if x.strip()]
        routing={key:value for key,value in {
            "required_all":phrases("required_all"), "required_any":phrases("required_any"),
            "excluded":phrases("excluded"), "period_scope":field("period_scope").upper(),
            "result_period":field("result_period"),
            "result_entity":field("result_entity"),
            "priority":float(field("priority")) if field("priority") else 0,
            "intent_group":field("intent_group"),
        }.items() if value not in ([],"",0)}
        interpretation={key:value for key,value in {
            "answer_template":field("answer_template"), "success_condition":field("success_condition"),
            "true_text":field("true_text"), "false_text":field("false_text"),
        }.items() if value}
        try:
            operation, spec, unit = compile_formula(name, executable_formula or formula, example)
            # Units describe the business metric shown to users, while the optional executable
            # formula is only a safe, parser-friendly equivalent of that definition.
            unit = field("output_type") or _output_unit(name, formula, example) or unit
            status, error = "approved", ""
        except ValueError as exc:
            # The compiler's message is exception text: the operator gets it in the log, while the
            # stored review reason (which can reach users in an answer) is a constant (CWE-209).
            logger.warning("[DOC-GM] rule %s needs review - formula did not compile: %s",
                           rule_id, sanitize_log_input(str(exc)))
            operation, spec, unit, status, error = ("unsupported", {}, "", "needs_review",
                                                    "the business formula could not be compiled into a validated calculation")
        compiled.append({"rule_id":rule_id, "source_name":source_name,
                         "document_pattern":document_pattern(source_name), "source_page":page,
                         "metric_family":str(row.get(columns["family"], "")).strip(), "metric_name":name,
                         "definition":formula, "applies_to":str(row.get(columns["applies"], "")).strip(),
                         "example":example, "operation":operation, "compiled_spec":spec,
                         "aliases":aliases, "routing_spec":routing,"interpretation_spec":interpretation,
                         "output_type":unit, "status":status, "error":error})
    if apply:
        save_rules(compiled)
    return compiled


async def save_rules(compiled: list[dict], conn=None, *, commit: bool = True) -> None:
    """Upsert already-compiled rules, optionally as part of a caller-owned transaction."""
    async def _run(c):
        sql = """INSERT INTO metric_rules
                     (rule_id,source_name,document_pattern,source_page,metric_family,metric_name,definition,applies_to,
                      example,operation,compiled_spec,aliases,routing_spec,interpretation_spec,
                      output_type,enabled,status,review_reason,version,updated_at)
                     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,1,$16,$17,1,now())
                     ON CONFLICT (rule_id) DO UPDATE SET source_name=EXCLUDED.source_name,
                      document_pattern=EXCLUDED.document_pattern,
                      source_page=EXCLUDED.source_page,metric_family=EXCLUDED.metric_family,
                      metric_name=EXCLUDED.metric_name,definition=EXCLUDED.definition,applies_to=EXCLUDED.applies_to,
                      example=EXCLUDED.example,operation=EXCLUDED.operation,compiled_spec=EXCLUDED.compiled_spec,
                      aliases=EXCLUDED.aliases,output_type=EXCLUDED.output_type,status=EXCLUDED.status,
                      review_reason=EXCLUDED.review_reason,
                      routing_spec=EXCLUDED.routing_spec,interpretation_spec=EXCLUDED.interpretation_spec,
                      version=metric_rules.version+1,updated_at=now()"""
        await c.executemany(sql, [(r["rule_id"],r["source_name"],
                                   r.get("document_pattern",document_pattern(r["source_name"])),
                                   r["source_page"],r["metric_family"],r["metric_name"],
                                   r["definition"],r["applies_to"],r["example"],r["operation"],
                                   json.dumps(r["compiled_spec"]),json.dumps(r["aliases"]),
                                   json.dumps(r["routing_spec"]),json.dumps(r["interpretation_spec"]),
                                   r["output_type"],r["status"],r.get("error", ""))
                                  for r in compiled])
    if conn is None:
        async with store.acquire() as c:
            await _run(c)
    else:
        await _run(conn)


async def import_fact_rows(path: str, *, page: int, source_name: str, apply: bool = False) -> dict:
    """Import a small, business-verified wide fact table when PDF table extraction missed it.

    Required columns are Section, Metric and Unit; every other header must be a parseable period.
    Rows are idempotently replaced for the named document/page and kept separate from native facts.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    if not rows or not {"Section", "Metric", "Unit"} <= set(rows[0]):
        raise ValueError("Fact table requires Section, Metric and Unit columns")
    period_columns = []
    for column in rows[0]:
        if column in ("Section", "Metric", "Unit"):
            continue
        period = period_parser.parse(column)
        if not period:
            raise ValueError(f"Unrecognized fact-table period: {column}")
        period_columns.append((column, period))
    report = {"series": len(rows), "facts": sum(bool(row.get(c, "").strip()) for row in rows for c, _ in period_columns)}
    if not apply:
        return report
    async with store.acquire() as conn:
        docs = await conn.fetch("SELECT doc_id FROM documents WHERE file_name = $1", source_name)
        if len(docs) != 1:
            raise ValueError(f"Expected one document named {source_name!r}, found {len(docs)}")
        doc_id = docs[0]["doc_id"]
        old = [r[0] for r in await conn.fetch(
            "SELECT series_id FROM series WHERE doc_id = $1 AND page = $2 AND source = 'business-fact-table'",
            doc_id, page)]
        if old:
            await conn.execute("DELETE FROM facts WHERE series_id = ANY($1::bigint[])", old)
            await conn.execute("DELETE FROM series WHERE series_id = ANY($1::bigint[])", old)
            # Weaviate vectors for these series are cleared opportunistically on the next ingest.
        heading_row = await conn.fetchrow(
            "SELECT heading FROM pages WHERE doc_id = $1 AND page = $2", doc_id, page)
        heading = heading_row["heading"] if heading_row else f"Page {page}"
        for row in rows:
            section, name, unit = row["Section"].strip(), row["Metric"].strip(), row["Unit"].strip()
            label = " › ".join(x for x in (heading, section, name) if x)
            kind = "percent" if unit == "%" else "amount" if unit == "Cr" else "count" if unit == "count" else ""
            sid = await store.add_series(conn, doc_id, page, None, label, name, "", kind, unit, "",
                                         "business-fact-table", 0.95, [heading, section, name])
            for column, period in period_columns:
                printed = row.get(column, "").strip()
                if not printed:
                    continue
                value = float(printed.replace(",", ""))
                await store.add_fact(conn, sid, doc_id, page, period, value, printed, False, 0.95,
                                     "transcribed from the page image after native table extraction missed the grid")
    return report


# NOTE: import_fact_rows above is called from CLI and is async; wrap its def line separately.


@dataclass
class Fact:
    fact_id: int
    period_key: str
    period_label: str
    period_kind: str
    period_start: str
    value: float
    unit: str


@dataclass
class Series:
    series_id: int
    label: str
    name: str
    entity: str
    unit: str
    confidence: float
    path: list[str]
    facts: list[Fact]


def _json_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


async def _series_for_page(conn, doc_id, page) -> list[Series]:
    rows = await conn.fetch(
        """SELECT s.series_id,s.label,s.name,s.entity,s.unit,s.confidence,s.path,
                  f.fact_id,f.period_key,f.period_label,f.period_kind,f.period_start,f.value
             FROM series s JOIN facts f USING(series_id)
             WHERE s.doc_id = $1 AND s.page = $2
             ORDER BY s.series_id, f.period_start, f.fact_id""",
        doc_id, page)
    grouped = {}
    for r in rows:
        grouped.setdefault(r["series_id"], Series(r["series_id"],r["label"],r["name"],r["entity"],
                                                   r["unit"],r["confidence"],
                                                   _json_list(r["path"]),[])).facts.append(
            Fact(r["fact_id"],r["period_key"],r["period_label"],r["period_kind"],r["period_start"],
                 float(r["value"]),r["unit"]))
    return list(grouped.values())


def _series_entity(series: Series, ref: dict) -> str:
    if ref["entity"] != "*":
        return ref["entity"]
    parts = series.label.split(" › ")
    terminal = parts[-1].lower().strip()
    source = ref.get("source_name", "")
    if source == "Overall Advances.pdf":
        return series.entity or series.name or parts[-1]
    if source == "Liablity Summary.pdf" and len(parts) >= 2:
        return series.entity or f"{parts[-1]} › {parts[-2]}"
    if source == "Collection Daily Refresh..pdf" and re.match(r"(?i)^B[123]\s*›", series.name):
        return series.entity or series.name
    # Repeated measure names under product sections need the product as their entity.  Without
    # this, all three Cross Sell "Disbursal" rows collapse into one wildcard entity.
    if len(parts) >= 2 and ("disbursal" in terminal or "no.of cases" in terminal
                            or "no. of cases" in terminal):
        return parts[-2]
    if ref.get("source_name", "").startswith("01_"):
        cleaned = " › ".join(parts[1:])
        cleaned = re.sub(r"(?i)\b(?:CM|LM|LQ|LY|LYCM)\b", "", cleaned)
        cleaned = re.sub(r"(?i)selected segment\s*/\s*(?:MEB|MAB)?\s*/?", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" /›-")
        return series.entity or cleaned or parts[-1]
    if ref.get("source_name", "").startswith(("03_", "07_")):
        measure = re.compile(r"(?i)\b(?:amount|no\.?\s*of\s*accounts|accounts?|advances?|yield)\b")
        cleaned = [re.sub(r"\s+", " ", measure.sub("", part)).strip(" /›-") for part in parts[1:]]
        cleaned = [part for part in cleaned if part]
        return series.entity or " › ".join(cleaned) or parts[-1]
    return series.entity or parts[-1]


def _row_value(row, key: str, default=""):
    return row[key] if key in row.keys() else default


def _period_label_matches(actual: str, wanted: str, ref: dict) -> bool:
    actual_norm = (actual or "").lower().strip()
    wanted_norm = (wanted or "").lower().strip()
    if actual_norm == wanted_norm:
        return True
    # YTD values extracted from a table are deliberately labelled as a partial fiscal year.
    # A business reference such as "YTD FY26" compiles to period FY26 plus selector YTD.
    return ("ytd" in {str(value).lower() for value in ref.get("selectors", [])}
            and actual_norm.removesuffix(" (partial)") == wanted_norm)


def _match_rank(series: Series, ref: dict) -> tuple[int, float]:
    """Prefer the printed row itself over an incidental mention in a ratio/path label."""
    compact = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    symbol = compact(ref.get("symbol", ""))
    name = compact(series.name)
    terminal = compact(series.label.split(" › ")[-1])
    exact = 2 if symbol and symbol in (name, terminal) else 1 if symbol and (
        name.startswith(symbol) or terminal.startswith(symbol)
    ) else 0
    return exact, series.confidence


def _rule_applies_to_document(rule, doc) -> bool:
    source = str(rule["source_name"] or "")
    if source == str(doc["file_name"] or ""):
        return True
    pattern = str(rule["document_pattern"] or "") if "document_pattern" in rule.keys() else ""
    if pattern.startswith("family:"):
        return document_family(doc["file_name"], doc["title"]) == pattern.removeprefix("family:")
    return False


def _source_symbol_match(series: Series, ref: dict) -> bool | None:
    """Interpret compact workbook symbols using the intended PDF and metric sheet."""
    source = ref.get("source_name", "")
    sheet = ref.get("metric_sheet", "")
    symbol = ref["symbol"]
    label = series.label.lower()
    terminal = series.label.split(" › ")[-1].lower()
    measure = (series.path[-2] if len(series.path) >= 2 else
               series.label.split(" › ")[-2] if " › " in series.label else "").lower().strip()
    family = document_family(source)
    parts = [part.lower().strip() for part in series.label.split(" › ")]
    if family == "wholesale_banking_review":
        # The workbook uses identifier-safe compound names while the deck stores a hierarchy,
        # e.g. AdvancesEOP -> Advances / A / EOP.  Match the semantic path, leaving scenario,
        # entity and period discrimination to the normal checks below.
        exact_paths = {
            "advanceseop": ("advances", "eop"),
            "advancesavg": ("advances", "avg"),
            "depositseop": ("deposits", "deposits eop"),
            "depositsavg": ("deposits", "deposits avg"),
            "ca": ("deposits", "ca"), "caavg": ("deposits", "ca avg"),
            "sa": ("deposits", "sa"), "saavg": ("deposits", "sa avg"),
            "td": ("deposits", "td"), "tdavg": ("deposits", "td avg"),
            "assetnii": ("asset nii",), "liabnii": ("liab nii",),
            "totalnii": ("total nii",), "otherincome": ("other income",),
            "totalrevenue": ("total revenue",), "pbt": ("pbt",),
            "ppop": ("ppop",), "totalcosts": ("total costs",),
            "pslcharge": ("psl charge",), "centralfunccost": ("central func",),
            "tradefee": ("trade fee",), "nfbos": ("total nfb os",),
            "sma0": ("sma 0",), "sma1": ("sma 1",), "sma2": ("sma 2",),
            "sma012total": ("sma 0/1/2",), "ewsalwltotal": ("ews/al/wl",),
            "ews": ("ews",), "al": ("al",), "wl": ("wl",),
            "pctoftotalportfolio": ("% contribution to total portfolio",),
            "pctofadvances": ("% contribution to advances",),
        }
        wanted = exact_paths.get(symbol)
        if wanted:
            return all(any(token in part for part in parts) for token in wanted)
        if symbol in {"wbinryield", "wbinryieldpct"}:
            return "inr yield and gross interest income trend" in label and series.unit == "%"
        terminal_map = {
            # PDF table extraction keeps the printed row as "Provisions" even when the
            # continuation line "(Net off Recovery)" is not part of that series label.
            "provisionsnetofrecovery": ("provisions",),
            "totalwbinterestincome": ("wholesale banking",),
            "wbinryieldpct": ("wholesale banking",),
        }
        wanted = terminal_map.get(symbol)
        if wanted:
            return all(token in label for token in wanted)
    if family == "treasury_review":
        if symbol == "revenue":
            return "in text" not in parts and terminal in {"trading", "treasury sales", "dcm", "total"}
        if symbol == "tradingrevenue":
            return terminal == "trading"
        if symbol == "treasurysalesrevenue":
            return terminal == "treasury sales" or ("in text" in parts and "treasury sales" in label)
        if symbol == "dcmdeal":
            return "debt capital markets" in label and "in text" in parts
    if family == "bbg_review":
        cluster_locations = {
            "panipatnorthdelhincr": "panipat", "karnalnorthdelhincr": "karnal",
            "moradabadnorthdelhincr": "moradabad", "ludhiananorthpunjabchandigarh": "ludhiana",
            "jaipurnorthrajasthan": "jaipur", "jodhpurnorthrajasthan": "jodhpur",
            "kanpurnorthup": "kanpur", "bangaloresouthkarnataka": "bangalore",
            "hyderabadsouthaptelangana": "hyderabad", "vijaywadasouthaptelangana": "vijaywada",
            "vizagsouthaptelangana": "vizag", "chennaisouthchennai": "chennai",
            "coimbatoresouthrotnkerala": "coimbatore", "salemsouthrotnkerala": "salem",
            "tirupursouthrotnkerala": "tirupur", "suratmumbaigujgujarat": "surat",
            "ichhalkaranjiromrom": "ichhalkaranji", "kolhapurromrom": "kolhapur",
            "puneromrom": "pune", "nasikromrom": "nasik", "aurangabadromrom": "aurangabad",
        }
        if symbol in cluster_locations:
            return cluster_locations[symbol] in parts
        pool_locations = {
            "punjabsachinpool": "punjab", "delhi1sachinpool": "delhi 1",
            "delhi2sachinpool": "delhi 2", "delhi3sachinpool": "delhi 3",
            "delhi4sachinpool": "delhi 4", "rajasthansachinpool": "rajasthan",
            "mpsachinpool": "mp", "eastsachinpool": "east",
            "pune1shaileshpool": "pune 1", "pune2shaileshpool": "pune 2",
            "gujaratshaileshpool": "gujarat", "mumbai1riteshpool": "mumbai 1",
            "mumbai2riteshpool": "mumbai 2", "mumbai3riteshpool": "mumbai 3",
            "chennairiteshpool": "chennai", "aptriteshpool": "apt",
            "rotnriteshpool": "rotn", "bangaloreriteshpool": "bangalore",
        }
        if symbol in pool_locations:
            return pool_locations[symbol] in parts
    if family in {"mortgages_hl_lap_review", "mortgages_lago_review", "rbg_review"}:
        compact_terminal = re.sub(r"[^a-z0-9]+", "", terminal)
        productivity = {
            "disbursalproductivitycount": ("disbursalproductivity", "#"),
            "disbursalproductivityamt": ("disbursalproductivity", "$"),
            "disbursalcount": ("disbursal", "#"),
            "disbursalamt": ("disbursal", "$"),
            "loginproductivity": ("loginproductivity", "#"),
            "login": ("login", "#"),
        }
        if symbol in productivity:
            name, marker = productivity[symbol]
            return name in compact_terminal and marker in terminal
        aliases = {
            "centralfunctionscost": "centralfuntionscost",
            "directlyidentifiablecost": "directlyidentifiablecost",
            "directlyidentifiablecosts": "directlyidentifiablecost",
            "pbtwocentercost": "pbtwocentercost",
            "pbtwocentralfunctioncost": "pbtwocentralfunctioncost",
        }
        if symbol in aliases:
            # Mortgage decks contain the misspelling ``Central funtions cost`` while
            # RBG prints the correctly-spelled form. Accept both the exact symbol and
            # the known source alias.
            return any(candidate in compact_terminal
                       for candidate in (symbol, aliases[symbol]))
    if family == "mfi_review" and symbol in {"smanov25", "smaoct25"}:
        selectors = [str(value).lower().strip() for value in ref.get("selectors", [])]
        state = next((value for value in selectors if value not in {
            "pos", "current rr", "sma0 rr", "sma1 rr", "sma2 rr"
        }), "")
        measure_sel = next((value for value in selectors if value in {
            "pos", "current rr", "sma0 rr", "sma1 rr", "sma2 rr"
        }), "")
        if not state or terminal != state:
            return False
        if measure_sel == "pos":
            return len(parts) == 2 and parts[0] == "sma pool with resolution"
        if measure_sel == "current rr":
            return "resolution current" in parts
        part = next((value for value in parts if re.fullmatch(r"part \d+", value)), "")
        if symbol == "smanov25":
            return ((measure_sel == "sma0 rr" and not part and "resolution" in parts)
                    or (measure_sel == "sma1 rr" and part == "part 2")
                    or (measure_sel == "sma2 rr" and part == "part 3"))
        return ((measure_sel == "sma0 rr" and part == "part 3")
                or (measure_sel == "sma1 rr" and part == "part 4")
                or (measure_sel == "sma2 rr" and part == "part 5"))
    if family == "mfi_review" and symbol == "resolutionrate":
        bucket = next((str(value).lower().strip() for value in ref.get("selectors", [])), "")
        wanted = {"bucket0": "sma - 0", "bucket1": "sma - 1", "bucket2": "sma - 2",
                  "current": "current"}.get(bucket)
        return "resolution rate" in parts and wanted is not None and terminal == wanted
    if family == "mfi_review" and symbol == "disbursementshare":
        state = next((str(value).lower().strip() for value in ref.get("selectors", [])), "")
        period = str(ref.get("period_label") or "")
        month = period[:3].lower()
        part_number = {name: index for index, name in enumerate(
            ("apr", "may", "jun", "jul", "aug", "sep", "oct", "nov"), 2
        )}.get(month)
        return (state and terminal == state and part_number is not None
                and f"part {part_number}" in parts)
    if family == "mfi_review":
        partner_metric = re.fullmatch(r"(rfl|pahal|nocpl)(actual|budget|variance)", symbol)
        if partner_metric:
            partner, scenario = partner_metric.groups()
            return partner in parts and scenario in parts
        if symbol == "disbursementfy25":
            entity = next((str(value).lower().strip() for value in ref.get("selectors", [])), "")
            return entity and terminal == entity and any(f.period_label == "FY25" for f in series.facts)
        trend_metric = re.fullmatch(r"(rfl|pahal|nocpl|total)(pos|slippage)", symbol)
        if trend_metric:
            entity, metric = trend_metric.groups()
            return terminal == f"{entity} / {metric}"
    if source == "Overall Advances.pdf":
        if symbol == "g": return "gross advances" in measure
        if symbol == "n": return "# accounts" in measure
        if symbol == "c": return "# customers" in measure
    if source == "Liablity Summary.pdf":
        if symbol == "b": return measure.endswith("/ current balance")
        if symbol == "n": return measure.endswith("/ no of accounts")
    if source == "Collection Daily Refresh..pdf":
        measures = {
            "cb": "currbal in cr closing cycles",
            "res": "resolution % closing cycles",
            "norm": "normalization % closing cycles",
            "rb": "roll back % closing cycles",
            "stab": "stab % closing cycles",
        }
        if symbol in measures:
            return measures[symbol] in series.label.lower()
    if source == "Retail Assets - Collections Efficiency.pdf":
        exact = {
            "pos": "pos", "respos": "res pos", "resn": "res #",
            "stab": "stab%", "rb": "rb%", "norm": "norm%", "res": "% res",
        }
        if symbol in exact:
            return measure == exact[symbol]
    if source.startswith("03_"):
        if symbol in {"a", "n"}:
            matches_measure = terminal.endswith("amount") if symbol == "a" else terminal.endswith("no of accounts")
            if not matches_measure: return False
            metric = ref.get("metric_name", "").lower()
            segments = [part.lower().strip() for part in series.label.split(" › ")]
            maturity = segments[-3] if len(segments) >= 3 else ""
            rate = terminal.removesuffix("amount").removesuffix("no of accounts").strip(" ›-")
            if metric == "row totals": return maturity == "grand total" and rate != "grand total"
            if metric == "column totals": return rate == "grand total" and maturity != "grand total"
            if "by rate bucket" in metric: return maturity == "grand total" and rate != "grand total"
            if "by maturity bucket" in metric: return rate == "grand total" and maturity != "grand total"
            if metric.startswith("cell "): return rate != "grand total" and maturity not in {"grand total", "null"}
            return True
    if source.startswith("07_"):
        if symbol == "a": return "advances" in label
        if symbol == "n": return "accounts" in label
        if symbol == "g": return "total" in terminal
    if source.startswith("09_"):
        if sheet == "P2_Metrics":
            if symbol == "gt": return terminal == "grand total"
            if symbol == "b": return series.unit == "Cr" and terminal != "grand total"
        return False
    if source.startswith("10_"):
        if symbol == "os": return "total outstanding / amt" in label
        if symbol == "n_os": return "total outstanding / no of accounts" in label
        if symbol == "sma": return "sma portfolio / amt" in label
        if symbol == "n_sma": return "sma portfolio / no of accounts" in label
        if symbol == "npa": return "npa portfolio / amt" in label
        if symbol == "n_npa": return "npa portfolio / no of accounts" in label
    if source.startswith("04_"):
        terms = {"n": ("customer", "count"), "s": ("sanction",), "ats": ("ats", "ticket")}
        if symbol in terms: return any(term in label for term in terms[symbol])
    if source.startswith("05_"):
        terms = {"d": ("disbursement",), "n": ("count", "number"), "y": ("yield",), "ats": ("ats",)}
        if symbol in terms: return any(term in label for term in terms[symbol])
    if source.startswith("01_"):
        if symbol == "gt": return "grand total" in label
        if symbol == "b": return "meb" in label and series.unit != "%"
        if symbol == "m": return "mab" in label and series.unit != "%"
    if source.startswith("02_"):
        normalized = re.sub(r"[^a-z0-9]+", " ", label)
        if symbol == "m": return "cm mab" in normalized and "diff" not in normalized and "3 cr" not in normalized
        if symbol == "m3": return ("cm mab" in normalized and "3 cr" in normalized
                                    and "indi" not in normalized and "diff" not in normalized)
        if symbol == "mi": return "cm mab indi" in normalized and "diff" not in normalized
        if symbol == "d_lm": return "cm vs lm mab diff" in normalized
        if symbol == "d_ly": return "cm vs ly mab diff" in normalized
    if source.startswith("08_"):
        if symbol == "b": return "end balance" in label
        if symbol == "n": return "end accounts" in label
    return None


def _matches(series: Series, ref: dict) -> bool:
    label = f"{series.label} {series.name}".lower().replace("sma - ", "sma-")
    symbol = ref["symbol"]
    terminal = series.label.split(" › ")[-1].lower().strip()
    # Page headings can contain both metric names ("CIF Net Addition"), so these symbols must
    # match the row itself rather than an incidental word in the full section path.
    funnel_rates={"pdd bounce%", "pdd4 bounce%", "res pdd4to10%", "res pex (pdd)%", "b1 flow"}
    bounce_amounts={"pdd bounce", "pdd4 bounce"}
    source_match = _source_symbol_match(series, ref)
    if source_match is False:
        return False
    if source_match is True:
        pass
    elif symbol == "pb":
        if terminal != "pdd bounce%": return False
    elif symbol == "p4":
        if terminal != "pdd4 bounce%": return False
    elif symbol == "r10":
        if terminal != "res pdd4to10%": return False
    elif symbol == "rx":
        if terminal != "res pex (pdd)%": return False
    elif symbol == "b1":
        if terminal != "b1 flow": return False
    elif symbol == "pba":
        if terminal != "pdd bounce": return False
    elif symbol == "p4a":
        if terminal != "pdd4 bounce": return False
    elif symbol == "funnelrate":
        if terminal not in funnel_rates: return False
    elif symbol == "bounceamount":
        if terminal not in bounce_amounts: return False
    elif symbol == "performanceitem":
        if terminal not in funnel_rates | bounce_amounts: return False
    elif symbol in ("volume", "volumeitem"):
        if " › volume › " not in f" {series.label.lower()} ": return False
    elif symbol in ("cleanvolume", "cleanvolumeitem"):
        if " › clean volume › " not in f" {series.label.lower()} ": return False
    elif symbol in ("pnl", "pnlitem", "pnlflow"):
        if " › p&l › " not in f" {series.label.lower()} ": return False
        if symbol == "pnlflow" and terminal == "c/i": return False
    elif symbol in ("cleanpnl", "cleanpnlitem", "cleanpnlflow"):
        if " › clean p&l › " not in f" {series.label.lower()} ": return False
        if symbol == "cleanpnlflow" and terminal == "c/i": return False
    elif symbol in ("dupont", "dupontitem"):
        if " › dupont › " not in f" {series.label.lower()} ": return False
    elif symbol in ("cleandupont", "cleandupontitem"):
        if " › clean dupont › " not in f" {series.label.lower()} ": return False
    elif symbol == "annualfactor":
        if terminal != "annualisation factor": return False
    elif symbol == "costitem":
        if " › p&l › " not in f" {series.label.lower()} " or terminal not in (
                "direct cost", "attributable cost", "central allocation"):
            return False
    elif symbol == "sourcingitem":
        path = {part.lower().strip() for part in series.label.split(" › ")[:-1]}
        if not ({"channel mix", "product mix"} & path): return False
    elif symbol == "cleancostitem":
        if " › clean p&l › " not in f" {series.label.lower()} " or terminal not in (
                "direct cost", "attributable cost", "central allocation"):
            return False
    elif symbol == "book":
        if terminal not in ("opening book", "closing book"):
            return False
    elif symbol == "inflowitem":
        if terminal not in ("spend", "stpl booking", "xps booking", "income/gst posted"):
            return False
    elif symbol == "outflowitem":
        if terminal not in ("payment", "write off"):
            return False
    elif symbol == "term":
        if terminal != "term":
            return False
    elif symbol == "nonterm":
        if terminal != "non-term":
            return False
    elif symbol == "spendcr":
        if series.name.lower().strip() != "spends (cr.)":
            return False
    elif symbol == "spc":
        parts=[part.lower().strip() for part in series.label.split(" › ")]
        if not (len(parts) >= 3 and parts[-2] == "spend/card"
                and parts[-1] == "spends/card/day"):
            return False
    elif symbol == "spcd":
        parts=[part.lower().strip() for part in series.label.split(" › ")]
        if not (len(parts) == 2 and parts[-1] == "spends/card/day"):
            return False
    elif symbol == "cif":
        if series.name.lower().strip() != "cif" and terminal != "cif": return False
    elif symbol == "na":
        if "net addition" not in series.name.lower() and "net addition" not in terminal: return False
    elif symbol.startswith("rows_"):
        group = symbol.removeprefix("rows_").replace("_", " ")
        path = " ".join(series.label.split(" › ")[:-1]).lower()
        normalize_group=lambda value:" ".join(re.findall(
            r"[a-z0-9]+", value.lower().replace("non-term", "nonterm")
        ))
        if normalize_group(group) not in normalize_group(path) or terminal in (
            "total", "grand total", "subtotal"
        ): return False
    else:
        terms = SYMBOL_TERMS.get(symbol, (symbol,))
        compact_label = re.sub(r"[^a-z0-9]+", "", label).replace("currentdemad", "currentdemand")
        # Workbook identifiers cannot contain spaces, so business templates use names such as
        # InterestIncome and CurrentDemand.  Match their compact form to the printed row label
        # while retaining all selector and period checks below.
        if not any(term in label or re.sub(r"[^a-z0-9]+", "", term.lower()) in compact_label
                   for term in terms):
            return False
    selectors = ref.get("selectors") or ([] if ref["entity"] == "*" else [ref["entity"]])
    if selectors:
        haystack = re.sub(r"[^a-z0-9]+", " ", f"{series.label} {series.name}".lower()).strip()
        source_words = set(re.findall(r"[a-z0-9]+", Path(ref.get("source_name", "")).stem.lower()))
        for wanted in selectors:
            needle = re.sub(r"[^a-z0-9]+", " ", wanted.lower()).strip()
            needle_words = needle.split()
            # The source document already scopes portfolio names such as Agri/RBG/RVF.  Remove
            # those words from a combined selector (``Agri projection``); the remaining scenario
            # still has to match the printed table header.
            remaining = [word for word in needle_words if word not in source_words]
            if needle_words and not remaining:
                continue
            needle = " ".join(remaining)
            needle = re.sub(r"\b(?:printed|reported|stated)\b", "", needle).strip()
            family = document_family(ref.get("source_name", ""))
            path_parts = {part.lower().strip() for part in series.label.split(" › ")}
            if family == "wholesale_banking_review":
                scenario = {
                    "actual": "a", "budget": "b", "aop": "b",
                    "projection": "p", "projected": "p", "estimate": "p",
                    "estimated": "p",
                }.get(needle)
                if scenario:
                    if scenario not in path_parts:
                        # Several WB P&L tables lose the "A" header only for the FY25 and
                        # Jan-26 columns during PDF extraction.  Their series is otherwise the
                        # untagged main row; tagged B/P and "part N" series remain ineligible.
                        period = str(ref.get("period_label") or "").lower()
                        untagged = not ({"a", "b", "p", "b finance"} & path_parts) \
                            and not any(part.startswith("part ") for part in path_parts)
                        if scenario == "a" and untagged and (
                                period.startswith("fy25") or period.startswith("jan-26")):
                            continue
                        if scenario == "b" and untagged and (
                                period.startswith("fy26") or period.startswith("q4fy26")):
                            continue
                        return False
                    continue
                if needle == "b finance":
                    if "b finance" in path_parts or any(
                            part.startswith("part ") for part in path_parts):
                        continue
                    return False
            if family == "treasury_review" and needle in {"amount", "revenue"}:
                continue
            if family == "mfi_review" and symbol in {
                    "smanov25", "smaoct25", "disbursementshare", "disbursementfy25",
                    "resolutionrate"}:
                # State and measure were both checked structurally above.
                continue
            if family == "mfi_review" and (re.fullmatch(
                    r"(rfl|pahal|nocpl)(actual|budget|variance)", symbol
                ) or re.fullmatch(r"(rfl|pahal|nocpl|total)(pos|slippage)", symbol)):
                continue
            if family == "bbg_review":
                if needle == "e2e tat" and "tat reduction" in haystack:
                    continue
                if needle.startswith("potential monthly limit") and "potential monthly limit" in haystack:
                    continue
            if family == "rvf_review" and needle == "total":
                compact_terminal = re.sub(r"[^a-z0-9]+", "", terminal)
                if compact_terminal.startswith(re.sub(r"[^a-z0-9]+", "", symbol)):
                    continue
            if (family == "mortgages_lago_review" and needle == "mortgage total lap hl"
                    and "mortgage" in haystack and "total lap hl" in haystack):
                continue
            variants = {
                "actual": ("actual", "actuals"),
                "budget": ("budget", "aop"),
                "estimate": ("estimate", "estimated", "projection", "projections", "projected", "proj"),
                "projection": ("projection", "projections", "projected", "proj", "estimate", "estimated"),
            }.get(needle, (needle,))
            if needle == "total" and re.sub(r"[^a-z0-9]+", "", series.name.lower()) == re.sub(
                    r"[^a-z0-9]+", "", symbol.lower()):
                continue
            if needle.startswith("gap") and "gap" in haystack:
                continue
            if needle and not any(variant in haystack for variant in variants):
                return False
    if ref.get("period_index") is not None:
        if len(series.facts) != ref.get("period_count") or ref["period_index"] >= len(series.facts):
            return False
    elif ref.get("period_label"):
        wanted = ref["period_label"].lower()
        if not any(_period_label_matches(f.period_label, wanted, ref) for f in series.facts):
            return False
    if ref.get("period_month") and not any(
            f.period_kind == "M" and f.period_start and int(f.period_start[5:7]) == ref["period_month"]
            for f in series.facts):
        return False
    if ref.get("period_quarter") and not any(
            re.match(rf"(?i)^Q{ref['period_quarter']}(?:FY)?", f.period_label)
            for f in series.facts):
        return False
    if ref.get("period_half") and not any(
            re.match(rf"(?i)^H{ref['period_half']}(?:FY)?", f.period_label)
            for f in series.facts):
        return False
    return True


def _refs(spec: dict) -> list[dict]:
    if spec.get("type") == "ref": return [spec]
    if spec.get("type") == "binary": return _refs(spec["left"]) + _refs(spec["right"])
    if spec.get("type") == "unary": return _refs(spec["value"])
    if spec.get("type") == "aggregate": return _refs(spec["value"])
    if spec.get("type") == "collection": return [ref for value in spec["values"] for ref in _refs(value)]
    return []


@dataclass
class EvalValue:
    value: float
    facts: list[Fact]


def _evaluate(spec, bindings, entity, target_fact, *, normalize_percent: bool = False):
    kind = spec["type"]
    if kind == "number": return EvalValue(spec["value"], [])
    if kind == "ref":
        candidates = bindings[id(spec)]
        series = candidates.get(entity) or candidates.get("*")
        if not series: raise KeyError("series not found")
        eligible = [f for f in series.facts if f.period_kind == target_fact.period_kind]
        if spec.get("period_index") is not None:
            eligible = series.facts
            index = spec["period_index"]
        elif spec.get("period_label"):
            eligible = [f for f in series.facts if _period_label_matches(
                f.period_label, spec["period_label"], spec
            )]
            if not eligible: raise KeyError("period not available")
            index = 0
        elif spec.get("period_month"):
            eligible = [f for f in series.facts if f.period_kind == "M" and f.period_start
                        and int(f.period_start[5:7]) == spec["period_month"]]
            if not eligible: raise KeyError("period not available")
            index = 0
        elif spec.get("period_quarter"):
            eligible = [f for f in series.facts if re.match(
                rf"(?i)^Q{spec['period_quarter']}(?:FY)?", f.period_label
            )]
            if not eligible: raise KeyError("period not available")
            index = 0
        elif spec.get("period_half"):
            eligible = [f for f in series.facts if re.match(
                rf"(?i)^H{spec['period_half']}(?:FY)?", f.period_label
            )]
            if not eligible: raise KeyError("period not available")
            index = 0
        else:
            index = next((i for i,f in enumerate(eligible) if f.period_key==target_fact.period_key), -1)
        index += spec.get("period_offset",0)
        if index < 0 or index >= len(eligible): raise KeyError("period not available")
        value = eligible[index].value
        # Extracted percentage facts are stored in display units (53.72 means 53.72%).
        # Workbook arithmetic uses percentages as fractions and explicitly multiplies by 100.
        # Preserve display units for a direct fact lookup, but normalize inputs participating in
        # a larger arithmetic expression.
        if normalize_percent and spec.get("percent_fraction"):
            value /= 100
        return EvalValue(value,[eligible[index]])
    if kind == "unary":
        value=_evaluate(spec["value"],bindings,entity,target_fact,
                        normalize_percent=normalize_percent)
        return EvalValue(-value.value if spec["op"]=="-" else value.value,value.facts)
    if kind == "binary":
        left=_evaluate(spec["left"],bindings,entity,target_fact,
                       normalize_percent=normalize_percent)
        right=_evaluate(spec["right"],bindings,entity,target_fact,
                        normalize_percent=normalize_percent)
        if spec["op"]=="+": value=left.value+right.value
        elif spec["op"]=="-": value=left.value-right.value
        elif spec["op"]=="*": value=left.value*right.value
        elif spec["op"]=="/":
            if right.value==0: raise ZeroDivisionError
            value=left.value/right.value
        else:
            value=left.value**right.value
        return EvalValue(value,left.facts+right.facts)
    raise KeyError("aggregate expressions require a period-level implementation")


def _fmt(value, unit):
    if unit=="%": return f"{value:+.2f}%" if value < 0 else f"{value:.2f}%"
    if unit=="pp": return f"{value:+.2f} pp"
    if unit=="bps": return f"{value:+.0f} bps"
    if unit=="x": return f"{value:.2f}x"
    if unit=="Cr": return f"{value:,.2f} Cr"
    if unit=="lakh": return f"{value:.2f} lakh"
    if unit=="count": return f"{value:,.0f}"
    if unit in ("'000", "000", "K"): return f"{value:,.2f}K"
    return f"{value:,.2f}"


def _reference_unit(ref: dict, series: Series) -> str:
    """Unit implied by an unambiguous source-specific fact symbol."""
    if series.unit:
        return series.unit
    source, symbol = ref.get("source_name", ""), ref.get("symbol", "")
    if source == "Overall Advances.pdf":
        return {"g": "Cr", "n": "count", "c": "count"}.get(symbol, "")
    if source == "Liablity Summary.pdf":
        return {"b": "Cr", "n": "count"}.get(symbol, "")
    if source == "Retail Assets - Collections Efficiency.pdf":
        return {"pos": "Cr", "respos": "Cr", "resn": "count"}.get(symbol, "")
    return ""


def _fact_position(series: Series, ref: dict) -> int | None:
    """Position of an explicit template period within one source series."""
    matches = []
    for index, fact in enumerate(series.facts):
        if ref.get("period_label") and _period_label_matches(
                fact.period_label, ref["period_label"], ref):
            matches.append(index)
        elif ref.get("period_month") and fact.period_kind == "M" and fact.period_start \
                and int(fact.period_start[5:7]) == ref["period_month"]:
            matches.append(index)
        elif ref.get("period_quarter") and re.match(
                rf"(?i)^Q{ref['period_quarter']}(?:FY)?", fact.period_label):
            matches.append(index)
        elif ref.get("period_half") and re.match(
                rf"(?i)^H{ref['period_half']}(?:FY)?", fact.period_label):
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


async def _relative_period_spec(conn, rule, doc, spec: dict) -> dict:
    """Map explicit periods to stable column slots for another document in the same family.

    The mapping is enabled only for a non-identical filename and only when the original source
    series resolves unambiguously.  The target must expose the same number of facts in that
    metric/scenario series, otherwise matching fails closed.
    """
    source_name = str(_row_value(rule, "source_name") or "")
    if not source_name or source_name == str(doc["file_name"] or ""):
        return spec
    source = await conn.fetchrow("SELECT * FROM documents WHERE file_name = $1", source_name)
    if not source or document_family(source["file_name"], source["title"]) != document_family(
            doc["file_name"], doc["title"]):
        return spec
    mapped = copy.deepcopy(spec)
    source_series = await _series_for_page(conn, source["doc_id"], int(rule["source_page"]))
    routing_raw = rule["routing_spec"] if "routing_spec" in rule.keys() else "{}"
    routing = routing_raw if isinstance(routing_raw, dict) else _json_object(routing_raw)
    for ref in _refs(mapped):
        if not any(ref.get(key) for key in ("period_label", "period_month", "period_quarter", "period_half")):
            continue
        ref["source_name"] = source_name
        ref["metric_name"] = rule["metric_name"]
        ref["metric_sheet"] = routing.get("metric_sheet", "")
        candidates = [series for series in source_series if _matches(series, ref)]
        if not candidates:
            continue
        base = max(candidates, key=lambda series: _match_rank(series, ref))
        position = _fact_position(base, ref)
        if position is None:
            continue
        ref["period_index"] = position
        ref["period_count"] = len(base.facts)
        ref["template_period_label"] = base.facts[position].period_label
        ref["period_label"] = ""
        ref["period_month"] = 0
        ref["period_quarter"] = 0
        ref["period_half"] = 0
    return mapped


async def _relative_result_periods(conn, rule, doc, ref: dict, target: Series,
                                   template_periods: set[str]) -> set[str]:
    """Translate a fixed output period to the same series slot in a recurring document."""
    source_name = str(_row_value(rule, "source_name") or "")
    if not template_periods or not source_name or source_name == doc["file_name"]:
        return template_periods
    source = await conn.fetchrow("SELECT * FROM documents WHERE file_name = $1", source_name)
    if not source or len(target.facts) == 0:
        return template_periods
    source_ref = copy.deepcopy(ref)
    source_ref.pop("period_index", None)
    source_ref.pop("period_count", None)
    source_ref["period_label"] = ""
    source_ref["period_month"] = 0
    source_ref["period_quarter"] = 0
    source_ref["period_half"] = 0
    candidates = [series for series in await _series_for_page(
        conn, source["doc_id"], int(rule["source_page"])
    ) if _matches(series, source_ref)]
    if not candidates:
        return template_periods
    base = max(candidates, key=lambda series: _match_rank(series, source_ref))
    if len(base.facts) != len(target.facts):
        return set()  # same-format contract failed; do not emit a result for the wrong window
    positions = [i for i, fact in enumerate(base.facts) if fact.period_label in template_periods]
    return {target.facts[i].period_label for i in positions}


def _has_division(spec):
    return (spec.get("type")=="binary" and (spec.get("op")=="/" or _has_division(spec["left"])
            or _has_division(spec["right"]))) or (spec.get("type") in ("unary","aggregate")
            and _has_division(spec["value"]))


def _has_explicit_percent_scale(spec):
    """Whether the executable expression already converts a ratio to percentage points."""
    if spec.get("type") == "binary":
        left, right = spec["left"], spec["right"]
        if spec.get("op") == "*" and any(
                child.get("type") == "number" and child.get("value") == 100
                for child in (left, right)):
            return True
        return _has_explicit_percent_scale(left) or _has_explicit_percent_scale(right)
    if spec.get("type") in ("unary", "aggregate"):
        return _has_explicit_percent_scale(spec["value"])
    if spec.get("type") == "collection":
        return any(_has_explicit_percent_scale(value) for value in spec.get("values", []))
    return False


async def _materialize_rule(conn, rule, doc) -> list[dict]:
    page=int(rule["source_page"])
    spec=json.loads(rule["compiled_spec"])
    spec=await _relative_period_spec(conn,rule,doc,spec)
    operation=rule["operation"] if "operation" in rule.keys() else ""
    if operation == "business_example":
        text=str(spec.get("text") or rule["example"] or "").strip()
        if not text:
            return []
        routing_raw=rule["routing_spec"] if "routing_spec" in rule.keys() else "{}"
        routing=routing_raw if isinstance(routing_raw,dict) else _json_object(routing_raw)
        entity=str(routing.get("result_entity") or "Business-approved result")
        result_period=str(routing.get("result_period") or "").strip()
        period_hits=[period for period,_ in period_parser.find_all(result_period)]
        primary_period=period_hits[0] if period_hits else None
        comparison_period=period_hits[1] if len(period_hits) > 1 else None
        if primary_period:
            period_key=primary_period.key
            period_label=primary_period.label
        elif re.search(r"(?i)\bytd\b|year.to.date",result_period):
            period_key="YTD"
            period_label="YTD"
        else:
            period_key="BUSINESS_APPROVED"
            period_label=result_period or "business-approved workbook output"
        provenance=str(spec.get("provenance") or "business metric catalogue")
        answer=f"{rule['metric_name']}: {text}"
        return [dict(
            doc_id=doc["doc_id"],source_page=page,metric_id=rule["rule_id"],
            metric_name=rule["metric_name"],metric_family=rule["metric_family"],entity=entity,
            period_key=period_key,period_label=period_label,
            comparison_period_key=comparison_period.key if comparison_period else "",
            comparison_period_label=comparison_period.label if comparison_period else "",
            value=None,formatted_value=text,
            unit=rule["output_type"],formula=rule["definition"],
            inputs_json=json.dumps({"approved_output":text,"provenance":provenance}),
            source_fact_ids="[]",answer=answer,
            search_text=" ".join((rule["metric_name"],rule["metric_family"],entity,result_period,text)),
            calculation_status="validated",calculation_version=rule["version"],
        )]
    series=await _series_for_page(conn,doc["doc_id"],page)
    refs=_refs(spec)
    if not refs: return []
    if spec.get("type") == "collection":
        results = []
        for value in spec["values"]:
            child = dict(rule)
            child["compiled_spec"] = json.dumps(value)
            child["output_type"] = "source"  # collections may combine counts, amounts and rates
            child_results=await _materialize_rule(conn, child, doc)
            child_refs=_refs(value)
            symbol=(child_refs[0].get("symbol", "value") if child_refs else "value")
            labels={
                "respos":"RES POS", "resn":"RES #", "res":"RES%",
                "stab":"STAB%", "rb":"RB%", "norm":"NORM%",
                "b":"Current Balance", "n":"No of Accounts", "c":"Customers",
            }
            for row in child_results:
                row["_collection_label"]=labels.get(symbol,symbol.replace("_"," ").upper())
            results.extend(child_results)
        # Different measures can share the same source entity and period (for example B1 POS,
        # RES POS and RES%). Give those rows distinct, readable entities before persistence so
        # they cannot overwrite one another in calculated_metrics.
        collisions={}
        for row in results:
            key=(row["entity"],row["period_key"],row["comparison_period_key"])
            collisions[key]=collisions.get(key,0)+1
        for row in results:
            key=(row["entity"],row["period_key"],row["comparison_period_key"])
            if collisions[key] > 1:
                row["entity"]=f"{row['entity']} › {row['_collection_label']}"
                row["answer"]=(f"{row['metric_name']} for {row['entity']} in "
                               f"{row['period_label']} is {row['formatted_value']}.")
                row["search_text"]=" ".join((row["search_text"],row["_collection_label"]))
            row.pop("_collection_label",None)
        unique = {}
        for row in results:
            key = (row["entity"], row["period_key"], row["comparison_period_key"], tuple(json.loads(row["source_fact_ids"])))
            unique.setdefault(key, row)
        return list(unique.values())
    bindings={}; wildcard_entities=None
    for ref in refs:
        # Symbol conventions belong to the template source, even when the template is being
        # reused for a newer file in the same document family.
        ref["source_name"] = _row_value(rule, "source_name") or doc["file_name"]
        ref["target_source_name"] = doc["file_name"]
        ref["metric_name"] = rule["metric_name"]
        routing_raw = rule["routing_spec"] if "routing_spec" in rule.keys() else "{}"
        routing_context = routing_raw if isinstance(routing_raw, dict) else _json_object(routing_raw)
        ref["metric_sheet"] = routing_context.get("metric_sheet", "")
        matched=[s for s in series if _matches(s,ref)]
        # Narrative callouts can yield two facts with the same label (for example, a run-rate
        # number and a percentage). Prefer the unit declared by the business rule.
        if operation == "fact" and rule["output_type"] and rule["output_type"] != "source" \
                and any(s.unit == rule["output_type"] for s in matched):
            matched=[s for s in matched if s.unit == rule["output_type"]]
        by_entity={}
        for s in sorted(matched,key=lambda x:_match_rank(x, ref),reverse=True):
            # A wildcard reference to one uniquely named row (for example proprietary(m)) is a
            # scalar input. Multiple matches (for example RR(*)) remain entity-wise calculations.
            key=("*" if ref["entity"]!="*" or len(matched)==1 else _series_entity(s,ref))
            by_entity.setdefault(key,s)
        bindings[id(ref)]=by_entity
        if ref["entity"]=="*":
            keys=set(by_entity)-{"*"}
            if keys:
                wildcard_entities=keys if wildcard_entities is None else wildcard_entities & keys
    entities=sorted(wildcard_entities or {"*"})
    primary=refs[0]; primary_map=bindings[id(primary)]
    routing_raw = rule["routing_spec"] if "routing_spec" in rule.keys() else "{}"
    routing = routing_raw if isinstance(routing_raw, dict) else _json_object(routing_raw)
    result_periods = ({period.label for period, _ in period_parser.find_all(routing.get("result_period", ""))}
                      if primary.get("period_index") is None else set())
    results=[]
    if spec.get("type")=="aggregate":
        for entity in entities:
            base=primary_map.get(entity) or primary_map.get("*")
            if not base: continue
            facts=[f for f in base.facts if "month" not in rule["applies_to"].lower() or f.period_kind=="M"]
            if not facts: continue
            values=[f.value for f in facts]; op=spec["op"]
            value={"sum":sum,"average":statistics.mean,"maximum":max,"minimum":min,"median":statistics.median,
                   "stddev":statistics.pstdev}[op](values)
            unit=base.unit if rule["output_type"] == "source" else rule["output_type"] or base.unit
            formatted=_fmt(value,unit)
            shown_entity=routing.get("result_entity") or (entity if entity!="*" else _series_entity(base,primary))
            period_label=f"{facts[0].period_label} to {facts[-1].period_label}"
            answer=f"{rule['metric_name']} for {shown_entity} over {period_label} is {formatted}."
            results.append(dict(doc_id=doc["doc_id"],source_page=page,metric_id=rule["rule_id"],
                metric_name=rule["metric_name"],metric_family=rule["metric_family"],entity=shown_entity,
                period_key=f"R:{facts[0].period_key}:{facts[-1].period_key}",period_label=period_label,
                comparison_period_key="",comparison_period_label="",value=value,formatted_value=formatted,
                unit=unit,formula=rule["definition"],inputs_json=json.dumps({f.period_label:f.value for f in facts}),
                source_fact_ids=json.dumps([f.fact_id for f in facts]),answer=answer,
                search_text=" ".join((rule["metric_name"],rule["metric_family"],shown_entity,period_label,answer)),
                calculation_status="validated",calculation_version=rule["version"]))
        return results
    for entity in entities:
        base=primary_map.get(entity) or primary_map.get("*")
        if not base: continue
        entity_result_periods = await _relative_result_periods(
            conn, rule, doc, primary, base, result_periods
        )
        for index,fact in enumerate(base.facts):
            if primary.get("period_index") is not None and index != primary["period_index"]:
                continue
            if primary.get("period_label") and fact.period_label != primary["period_label"]:
                continue
            if primary.get("period_month") and not (
                    fact.period_kind == "M" and fact.period_start
                    and int(fact.period_start[5:7]) == primary["period_month"]):
                continue
            if primary.get("period_quarter") and not re.match(
                    rf"(?i)^Q{primary['period_quarter']}(?:FY)?", fact.period_label):
                continue
            if primary.get("period_half") and not re.match(
                    rf"(?i)^H{primary['period_half']}(?:FY)?", fact.period_label):
                continue
            if "month" in rule["applies_to"].lower() and fact.period_kind!="M": continue
            if entity_result_periods and not any(
                    _period_label_matches(fact.period_label, period, primary)
                    for period in entity_result_periods):
                continue
            if result_periods and not entity_result_periods: continue
            try:
                out=_evaluate(spec,bindings,entity,fact,
                              normalize_percent=spec.get("type") != "ref")
            except (KeyError,ZeroDivisionError): continue
            used=list({f.fact_id:f for f in out.facts}.values())
            comparison=next((f for f in used if f.period_key!=fact.period_key),None)
            source_unit = _reference_unit(primary, base)
            unit = source_unit if rule["output_type"] == "source" else rule["output_type"] or source_unit
            value=out.value
            if (unit == "%" and spec.get("type") != "ref" and _has_division(spec)
                    and not _has_explicit_percent_scale(spec)):
                value*=100
            if unit=="bps" and spec.get("type")!="ref":
                value*=100
            formatted=_fmt(value,unit)
            shown_entity=routing.get("result_entity") or (entity if entity!="*" else _series_entity(base,primary))
            # A fixed-window formula can be evaluated relative to its first input (for example
            # Q1) while its business output is explicitly YTD or a later required month.  Showing
            # the anchor input as the result period is misleading.
            shown_period=fact.period_label
            result_period=str(routing.get("result_period") or "").strip()
            if result_period:
                if re.search(r"(?i)\bytd\b|year.to.date",result_period):
                    shown_period="YTD"
                else:
                    parsed_periods=period_parser.find_all(result_period)
                    if parsed_periods:
                        shown_period=parsed_periods[0][0].label
            answer=f"{rule['metric_name']} for {shown_entity} in {shown_period} is {formatted}."
            results.append(dict(doc_id=doc["doc_id"],source_page=page,metric_id=rule["rule_id"],
                metric_name=rule["metric_name"],metric_family=rule["metric_family"],entity=shown_entity,
                period_key=fact.period_key,period_label=shown_period,
                comparison_period_key=comparison.period_key if comparison else "",
                comparison_period_label=comparison.period_label if comparison else "",value=value,
                formatted_value=formatted,unit=unit,formula=rule["definition"],
                inputs_json=json.dumps({f.period_label:f.value for f in used}),
                source_fact_ids=json.dumps(sorted(f.fact_id for f in used)),answer=answer,
                search_text=" ".join((rule["metric_name"],rule["metric_family"],shown_entity,shown_period,
                                      comparison.period_label if comparison else "",answer)),
                calculation_status="validated",calculation_version=rule["version"]))
    return results


async def approve_business_examples(conn=None, *, apply: bool = False,
                                    rule_ids: list[str] | None = None) -> dict:
    """Promote reviewed workbook outputs without guessing unsupported formula notation."""
    async def _run(c):
        base_sql = (
            """SELECT * FROM metric_rules r
               WHERE r.status='needs_review'
                  OR (r.status='approved' AND r.operation!='business_example' AND NOT EXISTS (
                        SELECT 1 FROM calculated_metrics cm
                        WHERE cm.metric_id=r.rule_id AND cm.calculation_version=r.version
                     ))"""
        )
        if rule_ids:
            rules = await c.fetch(
                f"SELECT * FROM ({base_sql}) sub WHERE rule_id = ANY($1::text[]) ORDER BY rule_id",
                list(rule_ids),
            )
        else:
            rules = await c.fetch(base_sql + " ORDER BY rule_id")
        prepared=[]; missing=[]; review_promotions=0
        for rule in rules:
            text=(rule["example"] or "").strip()
            routing=_json_object(rule["routing_spec"])
            workbook=Path(str(routing.get("source_workbook") or "")).name
            workbook_path=Path("business_metrics_source")/workbook if workbook else None
            provenance=(str(workbook_path) if workbook_path and workbook_path.is_file()
                        else BUSINESS_WORKBOOKS.get(rule["source_name"]))
            if not text or not provenance:
                missing.append(rule["rule_id"])
                continue
            audit={
                "type":"business_example", "text":text, "provenance":provenance,
                "metric_sheet":routing.get("metric_sheet", ""),
                "previous_status":rule["status"],
                "previous_operation":rule["operation"],
                "previous_compiled_spec":_json_object(rule["compiled_spec"]),
                "previous_review_reason":rule["review_reason"],
            }
            prepared.append((json.dumps(audit,ensure_ascii=False),rule["rule_id"]))
            review_promotions += rule["status"] == "needs_review"
        if apply and prepared:
            await c.executemany(
                """UPDATE metric_rules SET operation='business_example',compiled_spec=$1,status='approved',
                   review_reason='Formally business-approved workbook output',enabled=1,
                   version=version+1,updated_at=now()
                   WHERE rule_id=$2 AND status IN ('needs_review','approved')""",
                prepared,
            )
            report=await materialize(c,rule_ids=[rule_id for _,rule_id in prepared])
        else:
            report={"rules":len(prepared),"results":0,"failures":[],"warnings":[]}
        remaining = await c.fetchval("SELECT COUNT(*) FROM metric_rules WHERE status='needs_review'")
        return {"applied":apply,"eligible":len(prepared),"missing":missing,
                "remaining_needs_review":remaining if apply else remaining-review_promotions,
                "materialized_results":report["results"],"failures":report["failures"]}

    if conn is None:
        async with store.acquire() as c:
            return await _run(c)
    return await _run(conn)


def _validate_example(rule, rows):
    """True/False for numeric examples; None when the example is descriptive only."""
    if not rows: return None
    operation=rule["operation"] if "operation" in rule.keys() else ""
    if operation == "business_example":
        return True  # exact formally approved workbook output; intentionally not reparsed as one number
    example=(rule["example"] or "").strip().replace("−", "-").replace("–", "-").replace("—", "-")
    if not example: return None
    if re.search(r"(?i)\b(?:all\s+)?tie\b.*\bwithin\b", example):
        return None  # a tolerance/rounding assertion, not one expected numeric result
    tail=example.split(":",1)[-1]
    matches=list(re.finditer(r"(?<![A-Za-z0-9'])([+-]?\d[\d,]*(?:\.\d+)?)",tail))
    if not matches: return None
    period_hits=period_parser.find_all(example)
    candidates=rows
    try:
        operation=rule["operation"]
    except (KeyError,IndexError):
        operation=""
    # Aggregate results are stored against the full input range.  An example may name the
    # month where the maximum/minimum occurred, which should not be mistaken for the result's
    # own period label during numeric validation.
    if len(period_hits)==1 and operation not in ("sum", "average", "maximum", "minimum", "median", "stddev"):
        label=period_hits[0][0].label
        period_candidates=[r for r in candidates if r["period_label"]==label]
        # Strings such as "FY 20.6%" can be read as FY20 by the period parser. Only scope the
        # example when that period actually exists among the materialized results.
        if period_candidates:
            candidates=period_candidates
    for match in matches:
        expected=float(match.group(1).replace(",",""))
        decimals=len(match.group(1).split(".",1)[1]) if "." in match.group(1) else 0
        tolerance=max(1e-9,0.5*(10**-decimals))
        if any(r["value"] is not None and abs(r["value"]-expected)<=tolerance for r in candidates): return True
    return False


async def materialize(conn=None, *, doc_ids=None, pages=None, rule_ids=None, dry_run=False) -> dict:
    async def _run(c):
        sql = "SELECT * FROM metric_rules WHERE enabled=1 AND status IN ('approved','example_failed')"
        args: list = []
        if pages:
            args.append(list(pages))
            sql += f" AND source_page = ANY(${len(args)}::int[])"
        if rule_ids:
            args.append(list(rule_ids))
            sql += f" AND rule_id = ANY(${len(args)}::text[])"
        rules = await c.fetch(sql, *args)
        dsql = "SELECT * FROM documents WHERE status IN ('ready','indexing')"
        dargs: list = []
        if doc_ids:
            dargs.append(list(doc_ids))
            dsql += f" AND doc_id = ANY(${len(dargs)}::text[])"
        docs = await c.fetch(dsql, *dargs)
        produced = 0
        failures: list[str] = []
        warnings: list[str] = []
        insert_sql = """INSERT INTO calculated_metrics
          (doc_id,source_page,metric_id,metric_name,metric_family,entity,period_key,period_label,
           comparison_period_key,comparison_period_label,value,formatted_value,unit,formula,inputs_json,
           source_fact_ids,answer,search_text,calculation_status,calculation_version)
          VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
          ON CONFLICT (doc_id, source_page, metric_id, entity, period_key,
                       comparison_period_key, calculation_version)
          DO UPDATE SET metric_name=EXCLUDED.metric_name, metric_family=EXCLUDED.metric_family,
                        period_label=EXCLUDED.period_label,
                        comparison_period_label=EXCLUDED.comparison_period_label,
                        value=EXCLUDED.value, formatted_value=EXCLUDED.formatted_value,
                        unit=EXCLUDED.unit, formula=EXCLUDED.formula,
                        inputs_json=EXCLUDED.inputs_json, source_fact_ids=EXCLUDED.source_fact_ids,
                        answer=EXCLUDED.answer, search_text=EXCLUDED.search_text,
                        calculation_status=EXCLUDED.calculation_status,
                        calculated_at=now()"""
        for rule in rules:
            rule_rows: list[dict] = []
            applicable_docs: list[str] = []
            for doc in docs:
                if rule["source_name"] and not _rule_applies_to_document(rule, doc):
                    continue
                if rule["source_page"] > (doc["pages"] or 0):
                    continue
                applicable_docs.append(doc["doc_id"])
                try:
                    rows_m = await _materialize_rule(c, rule, doc)
                except Exception as exc:
                    failures.append(f"{rule['rule_id']} / {doc['file_name']}: {exc}")
                    continue
                rule_rows.extend(rows_m)
            example_ok = _validate_example(rule, rule_rows)
            if not dry_run and applicable_docs:
                await c.execute(
                    "DELETE FROM issues WHERE doc_id = ANY($1::text[]) AND page=$2 "
                    "AND kind='calculation_example_mismatch' AND ref=$3",
                    applicable_docs, rule["source_page"], rule["rule_id"],
                )
            if example_ok is False:
                warning = f"{rule['rule_id']}: supplied example did not match the formula result; formula result retained"
                warnings.append(warning)
                if not dry_run:
                    for doc_id_ in applicable_docs:
                        await store.add_issue(
                            c, doc_id_, rule["source_page"], "calculation_example_mismatch",
                            rule["rule_id"], warning,
                            "Review the business example or formula definition",
                        )
            if not dry_run and rule["status"] == "example_failed":
                await c.execute(
                    "UPDATE metric_rules SET status='approved' WHERE rule_id=$1",
                    rule["rule_id"],
                )
            produced += len(rule_rows)
            if not dry_run and rule_rows:
                if applicable_docs:
                    await c.execute(
                        "DELETE FROM calculated_metrics "
                        "WHERE doc_id = ANY($1::text[]) AND metric_id=$2",
                        applicable_docs, rule["rule_id"],
                    )
                await c.executemany(insert_sql, [
                    tuple(r[k] for k in (
                        "doc_id","source_page","metric_id","metric_name",
                        "metric_family","entity","period_key","period_label",
                        "comparison_period_key","comparison_period_label","value",
                        "formatted_value","unit","formula","inputs_json",
                        "source_fact_ids","answer","search_text",
                        "calculation_status","calculation_version"))
                    for r in rule_rows
                ])
        return {"rules": len(rules), "documents": len(docs), "results": produced,
                "failures": failures, "warnings": warnings}

    if conn is None:
        async with store.acquire() as c:
            return await _run(c)
    return await _run(conn)


def _json_object(raw) -> dict:
    try:
        value=json.loads(raw or "{}")
        return value if isinstance(value,dict) else {}
    except (TypeError,json.JSONDecodeError):
        return {}


def _question_scopes(question: str) -> set[str]:
    scopes=set(); found=period_parser.find_all(question)
    kinds={period.kind for period,_ in found}
    capped=question[:MAX_REGEX_INPUT_LEN]
    if "Q" in kinds or re.search(r"(?i)\bq[1-4]\b",capped): scopes.add("Q")
    if "FY" in kinds and "Q" not in scopes: scopes.add("FY")
    if (("M" in kinds and len({period.label for period,_ in found})>=2)
            or _YTD_RANGE_RE.search(capped)):
        scopes.update(("RANGE","YTD"))
    elif "M" in kinds: scopes.add("M")
    return scopes


def _routing_matches(spec: dict, question_words: set[str], scopes: set[str]) -> bool:
    phrase=lambda value:_norm(str(value))
    if any(not phrase(value)<=question_words for value in spec.get("required_all",[])): return False
    required_any=spec.get("required_any",[])
    if required_any and not any(phrase(value)<=question_words for value in required_any): return False
    if any(phrase(value)<=question_words for value in spec.get("excluded",[])): return False
    wanted=str(spec.get("period_scope","")).upper()
    compatible={"M":{"M","RANGE"},"YTD":{"YTD","RANGE"},"RANGE":{"RANGE","YTD"},
                "Q":{"Q"},"FY":{"FY"}}
    return not (wanted and scopes and not (compatible.get(wanted,{wanted})&scopes))


def _inferred_period_scope(metric_name: str, applies_to: str) -> str:
    text=f"{metric_name} {applies_to}".lower()
    if re.search(r"\bquarter(?:ly)?\b",text): return "Q"
    if re.search(r"\bytd\b|year.to.date",text): return "YTD"
    if re.search(r"\bfull.year\b|\bfy\b|annual",text): return "FY"
    if re.search(r"\bmonth(?:ly)?\b",text): return "M"
    return ""


async def intended_metrics(conn, question: str, page: int | None = None, doc_ids=None) -> set[str]:
    """Return calculation IDs explicitly named by the question.

    Rules are the authoritative catalog. Distinct stored result names are also read so a database
    created by an older release remains queryable while its rules are imported.
    """
    logger.info("[DOC-GM] intended_metrics() question='%s' page=%s doc_ids=%s", sanitize_log_input(question[:120]), page, doc_ids)
    t0 = time.perf_counter()
    sql=("SELECT rule_id,source_name,document_pattern,metric_name,metric_family,definition,aliases,"
         "routing_spec,applies_to "
         "FROM metric_rules WHERE enabled=1 "
         "AND status IN ('approved','routing_only')"); args: list = []
    if page is not None: args.append(page); sql+=f" AND source_page=${len(args)}"
    scoped_docs: list = []
    if doc_ids:
        scoped_docs = [dict(row) for row in await conn.fetch(
            "SELECT * FROM documents WHERE doc_id = ANY($1::text[])", list(doc_ids)
        )]
    q=_norm(question); scopes=_question_scopes(question); candidates=[]
    for row in await conn.fetch(sql,*args):
        if scoped_docs and row["source_name"] and not any(
                _rule_applies_to_document(row, doc) for doc in scoped_docs):
            continue
        aliases=[row["metric_name"]]
        try: aliases+=json.loads(row["aliases"] or "[]")
        except json.JSONDecodeError: pass
        matches=[]
        for alias in aliases:
            alias_words=_norm(alias)
            compact_words=alias_words-{"segment", "product"}
            if alias_words and (alias_words<=q or (len(compact_words)>=2 and compact_words<=q)):
                matches.append(alias)
        if not matches: continue
        best_alias=max(matches,key=lambda alias:len(_norm(alias)))
        alias_words=_norm(best_alias)
        # A one-word alias is often an entity (RFL) or generic aggregation word (Total), not the
        # requested metric.  Do not let it hijack a richer question unless at least one of the
        # remaining business concepts is also present in this rule's own semantics.
        extra_concepts=_question_concepts(question)-alias_words
        if len(alias_words)==1 and extra_concepts:
            rule_concepts=_norm(" ".join(str(row[key] or "") for key in
                                         ("metric_name","metric_family","definition","applies_to")))
            if not (extra_concepts & rule_concepts):
                continue
        spec=_json_object(row["routing_spec"])
        if not spec.get("period_scope"):
            inferred=_inferred_period_scope(row["metric_name"],row["applies_to"])
            if inferred: spec={**spec,"period_scope":inferred}
        if not _routing_matches(spec,q,scopes): continue
        specificity=max(len(_norm(alias)) for alias in matches)
        specificity+=sum(len(_norm(value)) for value in spec.get("required_all",[]))
        score=float(spec.get("priority",0))+specificity
        if spec.get("period_scope") in scopes: score+=5
        candidates.append((row["rule_id"],str(spec.get("intent_group","")).strip(),score))
    found={rule_id for rule_id,group,_ in candidates if not group}
    groups={group for _,group,_ in candidates if group}
    for group in groups:
        members=[item for item in candidates if item[1]==group]
        best=max(item[2] for item in members)
        found.update(rule_id for rule_id,_,score in members if score==best)
    legacy_sql="SELECT DISTINCT metric_id,metric_name FROM calculated_metrics WHERE calculation_status='validated'"
    legacy_args: list = []
    if doc_ids:
        legacy_args.append(list(doc_ids))
        legacy_sql += f" AND doc_id = ANY(${len(legacy_args)}::text[])"
    if page is not None: legacy_args.append(page); legacy_sql+=f" AND source_page=${len(legacy_args)}"
    for row in await conn.fetch(legacy_sql,*legacy_args):
        words=_norm(row["metric_name"])
        if words and words<=q: found.add(row["metric_id"])
    # Natural questions commonly combine a calculation name with its row entity, for example
    # "Affordable Housing QoQ growth" for the stored metric "Product QoQ growth".  Match that
    # combined vocabulary only inside an already unambiguous document scope.  Requiring at least
    # three concepts keeps generic questions such as "total growth" out of this path.
    concepts=_question_concepts(question)
    if doc_ids and len(concepts)>=3:
        semantic_sql=("SELECT DISTINCT metric_id,metric_name,entity,period_label,"
                      "comparison_period_label FROM calculated_metrics "
                      "WHERE calculation_status='validated' AND doc_id = ANY($1::text[])")
        semantic_args: list = [list(doc_ids)]
        if page is not None:
            semantic_args.append(page)
            semantic_sql += f" AND source_page=${len(semantic_args)}"
        requested_periods=_intended_periods(question)
        semantic=[]
        for row in await conn.fetch(semantic_sql,*semantic_args):
            if requested_periods and not requested_periods & {
                    row["period_label"],row["comparison_period_label"]}:
                continue
            words=_norm(f"{row['metric_name']} {row['entity']}")
            if concepts<=words:
                semantic.append((len(concepts),-len(words-concepts),row["metric_id"]))
        if semantic:
            best=max(item[:2] for item in semantic)
            found.update(metric_id for *score,metric_id in semantic if tuple(score)==best)
    # Business questions often name the row/entity rather than the catalogue rule that imported
    # it ("CIF" rather than "Month-end CIF", "Xpress Cash" rather than "Monthly disbursal").
    # When no rule name matched, map an exact stored entity phrase back to its approved fact rule.
    # Restricting this fallback to fact rules prevents an entity such as CIF from also selecting
    # every derived calculation whose result entity happens to be CIF.
    if not found:
        entity_sql=("SELECT DISTINCT c.metric_id,c.entity,r.metric_name FROM calculated_metrics c "
                    "JOIN metric_rules r ON r.rule_id=c.metric_id "
                    "WHERE c.calculation_status='validated' AND r.enabled=1 "
                    "AND r.status='approved' AND r.operation='fact'")
        entity_args: list = []
        if doc_ids:
            entity_args.append(list(doc_ids))
            entity_sql += f" AND c.doc_id = ANY(${len(entity_args)}::text[])"
        if page is not None:
            entity_args.append(page)
            entity_sql += f" AND c.source_page=${len(entity_args)}"
        entity_matches=[]
        question_stems={word[:5] for word in q}
        for row in await conn.fetch(entity_sql,*entity_args):
            words=_norm(row["entity"] or "")
            if words and words<=q:
                metric_stems={word[:5] for word in _norm(row["metric_name"] or "")}
                entity_matches.append((len(metric_stems & question_stems),row["metric_id"]))
        if entity_matches:
            best=max(score for score,_ in entity_matches)
            found.update(metric_id for score,metric_id in entity_matches if score==best)
    logger.info("[DOC-GM] intended_metrics() -> %d ids: %s  (%.0fms)",
                len(found), sorted(found)[:10], (time.perf_counter() - t0) * 1000)
    return found


async def reviewed_metric_matches(conn, question: str, doc_ids=None, page: int | None = None,
                                  limit: int = 3) -> list[dict]:
    """Find a business metric the user named but which is not safe to execute.

    Review rules participate in intent discovery only. They never enter calculation search or
    evaluation through this function; callers use the returned reason to prevent an LLM fallback
    from inventing a derived answer from nearby numbers.
    """
    logger.info("[DOC-GM] reviewed_metric_matches() question='%s' doc_ids=%s page=%s limit=%d",
                sanitize_log_input(question[:120]), doc_ids, page, limit)
    t0 = time.perf_counter()
    sql = ("SELECT rule_id,source_name,document_pattern,source_page,metric_name,metric_family,aliases,"
           "routing_spec,applies_to,review_reason FROM metric_rules "
           "WHERE enabled=1 AND status='needs_review'")
    args: list = []
    if page is not None:
        args.append(page)
        sql += f" AND source_page=${len(args)}"
    scoped_docs: list = []
    if doc_ids:
        scoped_docs = await conn.fetch(
            "SELECT * FROM documents WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )
    question_words = _norm(question)
    scopes = _question_scopes(question)
    matches = []
    for row in await conn.fetch(sql, *args):
        if scoped_docs and row["source_name"] and not any(
                _rule_applies_to_document(row, doc) for doc in scoped_docs):
            continue
        if re.search(r"(?i)\bcallout\b", row["metric_name"] or ""):
            continue  # narrative claims belong to passage retrieval, not calculation blocking
        aliases = [row["metric_name"]]
        try:
            aliases += json.loads(row["aliases"] or "[]")
        except json.JSONDecodeError:
            pass
        matching = [alias for alias in aliases if _norm(alias) and _norm(alias) <= question_words]
        if not matching:
            continue
        routing = _json_object(row["routing_spec"])
        if not routing.get("period_scope"):
            inferred = _inferred_period_scope(row["metric_name"], row["applies_to"])
            if inferred:
                routing = {**routing, "period_scope": inferred}
        if not _routing_matches(routing, question_words, scopes):
            continue
        score = max(len(_norm(alias)) for alias in matching) + float(routing.get("priority", 0))
        reason = (row["review_reason"] or "").strip()
        if not reason:
            reason = "the business formula has not yet passed PDF-input and example validation"
        matches.append((score, {
            "metric_id": row["rule_id"], "metric_name": row["metric_name"],
            "metric_family": row["metric_family"], "source_name": row["source_name"],
            "source_page": row["source_page"], "reason": reason,
        }))
    matches.sort(key=lambda item: (-item[0], item[1]["metric_name"], item[1]["metric_id"]))
    result = [item for _, item in matches[:limit]]
    logger.info("[DOC-GM] reviewed_metric_matches() -> %d matches  (%.0fms)%s",
                len(result), (time.perf_counter() - t0) * 1000,
                "  names=" + str([m["metric_name"] for m in result]) if result else "")
    return result


def reviewed_metric_answer(matches: list[dict]) -> str:
    if not matches:
        return ""
    if len(matches) == 1:
        item = matches[0]
        source = f" in {item['source_name']} p.{item['source_page']}" if item["source_name"] else ""
        return (f"I found the business metric “{item['metric_name']}”{source}, but cannot safely "
                f"calculate it yet: {item['reason']}. No unvalidated number was returned.")
    lines = ["I found matching business metrics, but none is currently safe to calculate:"]
    for item in matches:
        source = f" — {item['source_name']} p.{item['source_page']}" if item["source_name"] else ""
        lines.append(f"- {item['metric_name']}{source}: {item['reason']}")
    lines.append("No unvalidated number was returned.")
    return "\n".join(lines)


async def _runtime_template(conn, question: str, doc_ids=None, page: int | None = None) -> tuple[dict, set[str]] | None:
    q = _norm(question)
    source_names = None
    if doc_ids:
        source_names = {row[0] for row in await conn.fetch(
            "SELECT file_name FROM documents WHERE doc_id = ANY($1::text[])",
            list(doc_ids),
        )}
    candidates = []
    for template in metric_bindings.RUNTIME_TEMPLATES:
        words = _norm(template["metric"])
        if not words or not words <= q:
            continue
        if page is not None and int(template["page"]) != int(page):
            continue
        if source_names is not None and not any(
                document_family(name) == document_family(template["source"]) for name in source_names):
            continue
        candidates.append((len(words), template))
    if not candidates:
        return None
    template = max(candidates, key=lambda item: item[0])[1]
    rule_ids = {row["rule_id"] for row in await conn.fetch(
        """SELECT rule_id, source_name FROM metric_rules
           WHERE enabled=1 AND status='approved' AND source_page=$1 AND metric_name=$2""",
        template["page"], template["base_metric"],
    ) if document_family(row["source_name"]) == document_family(template["source"])}
    return (template, rule_ids) if rule_ids else None


def _extremum_directions(question: str) -> list[tuple[str, str]]:
    """Return requested numeric extrema in wording order; values are selected by code."""
    patterns={
        "minimum": r"\b(?:lowest|smallest|minimum|min|trough|least)\b",
        "maximum": r"\b(?:highest|largest|maximum|max|peak|greatest)\b",
    }
    hits=[]
    for direction,pattern in patterns.items():
        for match in re.finditer(pattern,question,re.I):
            hits.append((match.start(),direction))
    return [(direction,"lowest" if direction=="minimum" else "highest")
            for _,direction in sorted(hits)]


def _comparison_windows(question: str) -> list[str]:
    """Two period-window phrases from wording such as 'between H1 and Oct-Feb'."""
    match=_COMPARISON_WINDOWS_RE.search(question[:MAX_REGEX_INPUT_LEN])
    return [match.group(1).strip(),match.group(2).strip()] if match else []


def _window_months(text: str) -> set[int] | None:
    if len(text)>MAX_REGEX_INPUT_LEN: return None
    half_match=_HALF_YEAR_RE.fullmatch(text.strip())
    if half_match:
        return set(range(4,10)) if half_match.group(1)=="1" else {10,11,12,1,2,3}
    named=period_parser.months_without_year(text)
    if len(named)==2 and re.search(r"[-–—]|\b(?:to|through|thru|till|until)\b",text,re.I):
        start,end=named
        return (set(range(start,end+1)) if start<=end
                else set(range(start,13))|set(range(1,end+1)))
    return None


def _rows_for_month_window(candidates: list[dict], months: set[int]) -> list[dict]:
    """Choose the best-covered, latest fiscal year when a window omits its year."""
    by_fy={}
    for row in candidates:
        period=period_parser.parse(row["period_label"] or "")
        if not period or period.kind!="M" or period.start.month not in months:
            continue
        by_fy.setdefault(period_parser.fiscal_year(period.start),[]).append((period,row))
    if not by_fy:
        return []
    fiscal_year=max(by_fy,key=lambda fy:(len({p.start.month for p,_ in by_fy[fy]}),fy))
    return [row for _,row in sorted(by_fy[fiscal_year],key=lambda item:item[0].start)]


def _matching_entity_rows(candidates: list[dict], question_tokens: set[str]) -> list[dict]:
    """Return the most specifically named entities, including useful partial paths.

    Business collections often store an entity as a hierarchy such as
    ``INSIGNIA › CA / Current Balance``.  Requiring every stored token to occur in
    the question makes a natural request for "INSIGNIA CA" miss both the balance
    and account rows.  A two-token overlap is strong enough to identify such a
    hierarchy. The leading path component is also a valid selector, so asking for
    ``B1`` returns all measures stored below ``B1``.
    """
    scored=[]
    for row in candidates:
        entity=row.get("entity") or ""
        tokens=_norm(entity)
        base_tokens=_norm(entity.split("›",1)[0])
        overlap=len(tokens & question_tokens)
        if not tokens or not overlap:
            continue
        exact=tokens <= question_tokens
        base_exact=bool(base_tokens and base_tokens <= question_tokens)
        if not exact and not base_exact and overlap < 2:
            continue
        match_class=3 if exact else 2 if base_exact else 1
        scored.append(((match_class,overlap),row))
    if not scored:
        return []
    best=max(score for score,_ in scored)
    return [row for score,row in scored if score==best]


def _grouped_period_comparison(question: str, rows: list[dict], intended: set[str], q: set[str]) -> list[dict]:
    """Aggregate two user-named multi-month windows from approved monthly fact rows."""
    windows=_comparison_windows(question)
    if len(windows)!=2:
        return []
    selected=[]
    for metric_id in sorted(intended):
        candidates=[r for r in rows if r["metric_id"]==metric_id and r["value"] is not None]
        entities=[r for r in candidates if r["entity"] and _norm(r["entity"])<=q]
        if entities:
            specificity=max(len(_norm(r["entity"])) for r in entities)
            candidates=[r for r in entities if len(_norm(r["entity"]))==specificity]
        if not candidates:
            continue
        sample=candidates[0]
        semantic_text=" ".join((question,sample.get("metric_name", ""),sample.get("metric_family", "")))
        average=bool(re.search(r"(?i)\b(?:average|avg|mean)\b",semantic_text))
        additive=bool(re.search(
            r"(?i)\b(?:sum|total|cumulative|addition|flow|volume|disbursal|spend|cases?|booking|"
            r"sourcing|cost|income|write[ -]?off)\b",semantic_text))
        if not (average or additive):
            continue
        groups=[]
        for label in windows:
            months=_window_months(label)
            group=_rows_for_month_window(candidates,months) if months else []
            if len(group)<2:
                groups=[]
                break
            groups.append((label,group))
        if len(groups)!=2:
            continue
        for label,group in groups:
            values=[float(row["value"]) for row in group]
            value=statistics.mean(values) if average else sum(values)
            unit=sample.get("unit","")
            shown_label=label.replace("-","–")
            operation="averaged" if average else "totaled"
            entity_suffix=("" if _norm(sample["entity"] or "")<=_norm(sample["metric_name"] or "")
                           else f" for {sample['entity']}")
            result=dict(sample)
            result.update(
                period_key="G:"+re.sub(r"\s+","_",label),period_label=shown_label,
                comparison_period_key="",comparison_period_label="",value=value,
                formatted_value=_fmt(value,unit),
                formula=f"{'average' if average else 'sum'} over {label}",
                inputs_json=json.dumps({row["period_label"]:row["value"] for row in group}),
                source_fact_ids=json.dumps(sorted({fact_id for row in group
                    for fact_id in json.loads(row.get("source_fact_ids") or "[]")})),
                answer=(f"{sample['metric_name']}{entity_suffix} {operation} "
                        f"{_fmt(value,unit)} over {shown_label}."),
            )
            selected.append(result)
    return selected


def _two_period_calculation(question: str, rows: list[dict], intended: set[str], q: set[str]) -> list[dict]:
    """Calculate a user-requested two-period change from validated fact-series rows only."""
    ordered_periods = []
    for period, _ in period_parser.find_all(question):
        if period.label not in ordered_periods:
            ordered_periods.append(period.label)
    if len(ordered_periods) != 2:
        return []
    percent = bool(re.search(
        r"(?i)\b(?:growth(?:\s+rate)?|percentage\s+change|percent\s+change|grew|declined\s+by)\b",
        question,
    ))
    difference = bool(re.search(r"(?i)\b(?:absolute\s+change|difference|change)\b", question))
    ratio = bool(re.search(r"(?i)\bratio\b", question))
    if not (percent or difference or ratio):
        return []
    start_label, end_label = ordered_periods
    results = []
    for metric_id in sorted(intended):
        candidates = [row for row in rows if row["metric_id"] == metric_id and row["value"] is not None]
        entities = [row for row in candidates if row["entity"] and _norm(row["entity"]) <= q]
        if entities:
            specificity = max(len(_norm(row["entity"])) for row in entities)
            candidates = [row for row in entities if len(_norm(row["entity"])) == specificity]
        by_period = {}
        for row in candidates:
            by_period.setdefault(row["period_label"], []).append(row)
        if len(by_period.get(start_label, [])) != 1 or len(by_period.get(end_label, [])) != 1:
            continue
        start, end = by_period[start_label][0], by_period[end_label][0]
        start_value, end_value = float(start["value"]), float(end["value"])
        if percent or ratio:
            if start_value == 0:
                continue
            value = end_value / start_value
            if percent:
                value = (value - 1) * 100
                unit = "%"
                formula = f"({end_label} / {start_label} - 1) × 100"
                name = f"{start['metric_name']} growth"
            else:
                unit = "x"
                formula = f"{end_label} / {start_label}"
                name = f"{start['metric_name']} ratio"
        else:
            value = end_value - start_value
            unit = "pp" if start.get("unit") == "%" else start.get("unit", "")
            formula = f"{end_label} - {start_label}"
            name = f"{start['metric_name']} change"
        result = dict(end)
        entity = end.get("entity") or start.get("entity") or ""
        formatted = _fmt(value, unit)
        result.update(
            metric_name=name, period_key=f"C:{start['period_key']}:{end['period_key']}",
            period_label=end_label, comparison_period_key=start["period_key"],
            comparison_period_label=start_label, value=value, formatted_value=formatted,
            unit=unit, formula=formula,
            inputs_json=json.dumps({start_label:start_value, end_label:end_value}),
            source_fact_ids=json.dumps(sorted(set(json.loads(start.get("source_fact_ids") or "[]")) |
                                              set(json.loads(end.get("source_fact_ids") or "[]")))),
            answer=(f"{name} for {entity} from {start_label} to {end_label} is {formatted}."),
            calculation_status="runtime_validated",
        )
        results.append(result)
    return results


def _quarterly_rollup(question: str, rows: list[dict], intended: set[str], q: set[str],
                      metric_name: str) -> list[dict]:
    """Sum approved monthly fact rows into Indian fiscal quarters with full provenance."""
    results = []
    requested = {period.label for period, _ in period_parser.find_all(question) if period.kind == "Q"}
    for metric_id in sorted(intended):
        candidates = [row for row in rows if row["metric_id"] == metric_id and row["value"] is not None]
        entities = [row for row in candidates if row["entity"] and _norm(row["entity"]) <= q]
        if entities:
            specificity = max(len(_norm(row["entity"])) for row in entities)
            candidates = [row for row in entities if len(_norm(row["entity"])) == specificity]
        groups = {}
        for row in candidates:
            period = period_parser.parse(row["period_label"] or "")
            if not period or period.kind != "M":
                continue
            quarter_number = ((period.start.month - 4) % 12) // 3 + 1
            quarter = period_parser.quarter(period_parser.fiscal_year(period.start), quarter_number)
            groups.setdefault((row["entity"], quarter.key, quarter.label), []).append((period, row))
        for (entity, quarter_key, quarter_label), group in sorted(groups.items()):
            if requested and quarter_label not in requested:
                continue
            group.sort(key=lambda item: item[0].start)
            values = [float(row["value"]) for _, row in group]
            sample = group[-1][1]
            value = sum(values)
            partial = len({period.start.month for period, _ in group}) < 3
            shown_period = quarter_label + (" (partial)" if partial else "")
            fact_ids = sorted({fact_id for _, row in group
                               for fact_id in json.loads(row.get("source_fact_ids") or "[]")})
            unit = sample.get("unit", "")
            result = dict(sample)
            result.update(
                metric_name=metric_name, period_key=f"R:{quarter_key}", period_label=shown_period,
                comparison_period_key="", comparison_period_label="", value=value,
                formatted_value=_fmt(value, unit), unit=unit, formula="sum of validated monthly facts",
                inputs_json=json.dumps({row["period_label"]:row["value"] for _, row in group}),
                source_fact_ids=json.dumps(fact_ids),
                answer=(f"{metric_name} for {entity} in {shown_period} is {_fmt(value, unit)}."),
                calculation_status="runtime_validated",
            )
            results.append(result)
    return results


def _display_amount_magnitudes(question: str, rows: list[dict]) -> list[dict]:
    """Render passive outflow amounts positively while retaining the source sign in the answer."""
    if not re.search(
        r"(?i)\bhow\s+much\b.*\b(?:written|wrote|charged|paid|repaid)\b(?:\s+off)?",
        question,
    ):
        return rows
    displayed=[]
    for original in rows:
        row=dict(original)
        if row.get("value") is not None and float(row["value"])<0:
            source_value=row["formatted_value"]
            row["value"]=abs(float(row["value"]))
            row["formatted_value"]=_fmt(row["value"],row.get("unit",""))
            measure=row.get("entity") or row.get("metric_name") or "Amount"
            row["answer"]=(f"{measure} was {row['formatted_value']} in {row['period_label']} "
                           f"(reported as {source_value} in the source walk).")
        displayed.append(row)
    return displayed


def _intended_periods(question: str) -> set[str]:
    # A fiscal YTD range is commonly written with the year only on its end month
    # (for example, "Apr–Nov'25").  The generic period parser correctly finds
    # Nov-25, but treating that endpoint as the requested period selects a monthly
    # result instead of the stored Apr-to-Nov YTD result.
    explicit_ytd=bool(re.search(r"(?i)\bytd\b|year.to.date",question))
    fiscal_ytd_range=bool(re.search(
        r"(?i)\bapr(?:il)?\s*(?:-|\u2013|\u2014|to|through|until|till)\s*"
        r"(?:may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?|jan(?:uary)?|feb(?:ruary)?)\b",
        question,
    ))
    if explicit_ytd or fiscal_ytd_range:
        return {"YTD"}
    periods={period.label for period,_ in period_parser.find_all(question)}
    return periods


async def search(conn, question: str, doc_ids=None, page: int | None = None, limit: int = 12) -> list[dict]:
    """Rank stored calculations without page-specific aliases or branches."""
    logger.info("[DOC-GM] search() question='%s' doc_ids=%s page=%s limit=%d", sanitize_log_input(question[:120]), doc_ids, page, limit)
    t0 = time.perf_counter()
    sql="SELECT * FROM calculated_metrics WHERE calculation_status='validated'"; args: list = []
    if doc_ids:
        args.append(list(doc_ids))
        sql += f" AND doc_id = ANY(${len(args)}::text[])"
    if page is not None:
        args.append(page)
        sql += f" AND source_page=${len(args)}"
    q=_norm(question); periods=_intended_periods(question)
    intended=await intended_metrics(conn,question,page,doc_ids)
    scored=[]
    for raw in await conn.fetch(sql,*args):
        row=dict(raw); score=len(q&_norm(row["search_text"] or ""))
        if row["metric_id"] in intended: score+=24
        if row["metric_name"] and _norm(row["metric_name"])<=q: score+=10
        if row["entity"] and _norm(row["entity"])<=q: score+=6
        if periods:
            if row["period_label"] in periods or row["comparison_period_label"] in periods: score+=12
            else: score-=2
        if score>1: scored.append((score,row))
    scored.sort(key=lambda item:(-item[0],item[1]["metric_id"],item[1]["entity"],item[1]["period_key"]))
    selected=[]; seen=set()
    for metric_id in sorted(intended):
        hit=next((row for _,row in scored if row["metric_id"]==metric_id),None)
        if hit: selected.append(hit); seen.add(hit["result_id"])
    for _,row in scored:
        if len(selected)>=limit: break
        if row["result_id"] not in seen: selected.append(row); seen.add(row["result_id"])
    logger.info("[DOC-GM] search() -> %d results  (%.0fms)", len(selected[:limit]), (time.perf_counter() - t0) * 1000)
    return selected[:limit]


async def _result_period_intended(conn, question: str, doc_ids, page: int | None) -> set[str]:
    """Discover a reused template by its target document's materialized period and base alias."""
    if not doc_ids:
        return set()
    periods = _intended_periods(question)
    if not periods:
        return set()
    sql = ("SELECT c.metric_id, c.entity, c.period_label, c.comparison_period_label, "
           "r.metric_name, r.aliases FROM calculated_metrics c JOIN metric_rules r ON r.rule_id=c.metric_id "
           "WHERE c.calculation_status='validated' AND c.doc_id = ANY($1::text[])")
    args: list = [list(doc_ids)]
    if page is not None:
        args.append(page)
        sql += f" AND c.source_page=${len(args)}"
    q = _norm(question)
    matched = []
    for row in await conn.fetch(sql, *args):
        if row["period_label"] not in periods and row["comparison_period_label"] not in periods:
            continue
        aliases = [row["metric_name"]]
        try:
            aliases += json.loads(row["aliases"] or "[]")
        except json.JSONDecodeError:
            pass
        specificity = max((len(_norm(alias)) for alias in aliases if _norm(alias) and _norm(alias) <= q), default=0)
        if not specificity:
            continue
        entity_words = _norm(row["entity"] or "")
        entity_score = len(entity_words) if entity_words and entity_words <= q else 0
        matched.append((specificity, entity_score, row["metric_id"]))
    if not matched:
        return set()
    best = max((specificity, entity_score) for specificity, entity_score, _ in matched)
    return {metric_id for specificity, entity_score, metric_id in matched
            if (specificity, entity_score) == best}


async def _two_period_dynamic_intended(conn, question: str, doc_ids,
                                       page: int | None) -> set[str]:
    """Prefer one fact-backed series for an ad-hoc two-period calculation.

    Exhaustive business catalogues often register one scalar rule per period.  Those rows can
    otherwise hide an older, validated multi-period series and make a growth question return its
    two endpoints instead of the requested calculation.
    """
    periods = []
    for period, _ in period_parser.find_all(question):
        if period.label not in periods:
            periods.append(period.label)
    if len(periods) != 2 or not re.search(
        r"(?i)\b(?:growth(?:\s+rate)?|percentage\s+change|percent\s+change|grew|"
        r"declined\s+by|absolute\s+change|difference|change|ratio)\b",
        question,
    ):
        return set()
    sql = ("SELECT c.metric_id, c.metric_name, c.entity, c.period_label "
           "FROM calculated_metrics c JOIN metric_rules r ON r.rule_id=c.metric_id "
           "WHERE c.calculation_status='validated' AND c.value IS NOT NULL "
           "AND r.enabled=1 AND r.status='approved' AND r.operation='fact'")
    args: list = []
    if doc_ids:
        args.append(list(doc_ids))
        sql += f" AND c.doc_id = ANY(${len(args)}::text[])"
    if page is not None:
        args.append(page)
        sql += f" AND c.source_page=${len(args)}"
    grouped = {}
    for row in await conn.fetch(sql, *args):
        if row["period_label"] not in periods:
            continue
        grouped.setdefault(row["metric_id"], []).append(row)
    q = _norm(question)
    candidates = []
    for metric_id, rows in grouped.items():
        entity_rows = [row for row in rows if row["entity"] and _norm(row["entity"]) <= q]
        selected = entity_rows or rows
        if {row["period_label"] for row in selected} != set(periods):
            continue
        entity_score = max((len(_norm(row["entity"])) for row in entity_rows), default=0)
        metric_score = len(_norm(selected[0]["metric_name"]) & q)
        candidates.append(((entity_score, metric_score), metric_id))
    if not candidates:
        return set()
    best = max(score for score, _ in candidates)
    return {metric_id for score, metric_id in candidates if score == best}


def _direct_fact_answer(row: dict) -> dict:
    """Present one exact, printed source fact in the same shape as a calculated result."""
    try:
        path = json.loads(row.get("path") or "[]")
    except json.JSONDecodeError:
        path = []
    entity = str(row.get("name") or "").strip()
    measure = str(path[-2]).strip() if len(path) > 1 else str(row.get("label") or entity).strip()
    measure = re.sub(r"[\ue000-\uf8ff](?=\s*3\b)", "<", measure)
    measure = re.sub(r"(?i)\bindi\b", "Individual", measure)
    scenario_names = {"A": "Actual", "B": "Budget", "P": "Projection"}
    if measure in scenario_names and entity:
        measure = f"{scenario_names[measure]} {entity}"
    if _norm(measure) <= {"balance", "date", "month", "period"}:
        measure = entity or str(row.get("label") or "Value").strip()
    shown = str(row.get("as_printed") or "").strip()
    if not shown:
        shown = _fmt(float(row["value"]), str(row.get("unit") or ""))
    period = str(row.get("period_label") or "").strip()
    when = "" if not period or period == "period not stated" else f" in {period}"
    subject = measure if not entity or _norm(entity) <= _norm(measure) else f"{measure} for {entity}"
    return {
        "result_id": -int(row["fact_id"]), "doc_id": row["doc_id"],
        "source_page": int(row["page"]), "metric_id": f"source_fact:{row['fact_id']}",
        "metric_name": measure, "metric_family": "Direct source fact", "entity": entity,
        "period_key": row.get("period_key") or "", "period_label": period,
        "comparison_period_key": "", "comparison_period_label": "", "value": row["value"],
        "formatted_value": shown, "unit": row.get("unit") or "", "formula": "direct source value",
        "inputs_json": json.dumps({period or "source": row["value"]}),
        "source_fact_ids": json.dumps([row["fact_id"]]),
        "answer": f"{subject}{when} is {shown}.",
        "search_text": row.get("label") or "", "calculation_status": "source_validated",
        "calculation_version": 1, "interpretation_spec": {},
        "source_name": row.get("file_name") or "",
    }


async def resolve_direct_fact(conn, question: str, doc_ids=None, page: int | None = None) -> list[dict]:
    """Resolve an exact entity/measure/period question straight to a printed fact.

    This deliberately requires every business concept in the question to occur in the structured
    series path.  It is a high-precision guard in front of broad workbook aliases, not a fuzzy
    replacement for normal retrieval.  When a dashboard repeats the same table for current and
    prior snapshots without encoding those captions in the extracted path, reading order is used:
    dashboards conventionally put the current table first (left/top).
    """
    concepts = _question_concepts(question)
    if len(concepts) < 2:
        return []
    sql = ("SELECT f.fact_id, f.doc_id, f.page, f.period_key, f.period_label, f.value, f.as_printed, "
           "s.name, s.label, s.path, s.unit, s.series_id, s.block_id, d.file_name, d.title "
           "FROM facts f JOIN series s ON s.series_id = f.series_id "
           "JOIN documents d ON d.doc_id = f.doc_id WHERE f.value IS NOT NULL")
    args: list = []
    if doc_ids:
        args.append(list(doc_ids))
        sql += f" AND f.doc_id = ANY(${len(args)}::text[])"
    if page is not None:
        args.append(page)
        sql += f" AND f.page = ${len(args)}"
    periods = _intended_periods(question)
    candidates = []
    for raw in await conn.fetch(sql, *args):
        row = dict(raw)
        if periods and row["period_label"] not in periods:
            continue
        descriptor = " ".join(str(row.get(key) or "") for key in ("label", "name", "path"))
        descriptor_words = _norm(descriptor)
        try:
            path_parts = json.loads(row.get("path") or "[]")
        except json.JSONDecodeError:
            path_parts = []
        # Several business tables encode scenario columns as A/B/P.  Interpret these only as
        # complete path components; treating every letter "B" as budget would create false hits.
        scenario_words = {"A": "actual", "B": "budget", "P": "projection"}
        descriptor_words.update(scenario_words[part] for part in path_parts
                                if part in scenario_words)
        if not concepts <= descriptor_words:
            continue
        # Prefer the shortest exact structured path, then question/report overlap.
        doc_overlap = len(_norm(f"{row['file_name']} {row['title']}") & concepts)
        unit_bonus = int("rate" in concepts and row.get("unit") == "%")
        score = (len(concepts) * 20 + doc_overlap * 3 + unit_bonus * 4
                 - len(descriptor_words - concepts))
        candidates.append((score, int(row.get("block_id") or 0), int(row["series_id"]), row))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    signature = lambda item: (item[3]["doc_id"], item[3]["page"],
                              frozenset(_norm(item[3]["label"] or "")))
    # Equally good but semantically different sources are genuinely ambiguous; let normal
    # retrieval show context instead of silently choosing one.
    if len({signature(item) for item in best}) > 1:
        return []
    return [_direct_fact_answer(best[0][3])]


async def resolve(conn, question: str, doc_ids=None, page: int | None = None, limit: int = 12) -> list[dict]:
    """Resolve only registered/stored metric names; unknown questions use normal document Q&A."""
    logger.info("[DOC-GM] resolve() question='%s' doc_ids=%s page=%s limit=%d", sanitize_log_input(question[:120]), doc_ids, page, limit)
    t0 = time.perf_counter()
    if doc_ids is None:
        doc_ids = await infer_document_scope(conn, question)
    catalog_intended=await intended_metrics(conn,question,page,doc_ids)
    logger.info("[DOC-GM]   catalog_intended=%s", sorted(catalog_intended)[:10])
    # An exact catalogue-name lookup is authoritative.  Period discovery and runtime templates
    # are useful for natural-language questions, but they must not replace a rule when callers
    # ask for that rule by its registered name (the ingestion verifier does exactly this).
    normalized_question=re.sub(r"\s+"," ",question.strip().lower()[:MAX_REGEX_INPUT_LEN])
    # Whitespace is already collapsed to single spaces, so the lead-in is followed by
    # exactly one space and the metric phrase is simply the rest of the text.
    asked_match=_ASKED_PREFIX_RE.search(normalized_question)
    asked_phrase=normalized_question[asked_match.end():] if asked_match else ""
    metric_phrase=(asked_phrase.strip(" ?.!") if asked_phrase else normalized_question)

    def explicitly_asked(metric_name: str) -> bool:
        name=re.sub(r"\s+"," ",(metric_name or "").strip().lower())
        return bool(name) and (name == metric_phrase or (
            len(_norm(name)) >= 2 and metric_phrase.startswith(name + " ")
        ))

    exact_catalog_ids = set()
    if catalog_intended:
        exact_catalog_ids = {row["rule_id"] for row in await conn.fetch(
            "SELECT rule_id, metric_name, status FROM metric_rules "
            "WHERE rule_id = ANY($1::text[])",
            sorted(catalog_intended),
        ) if row["status"] != "routing_only" and explicitly_asked(row["metric_name"])}
    if exact_catalog_ids:
        longest=max((len(_norm(row["metric_name"] or "")),
                     len(re.sub(r"\s+"," ",(row["metric_name"] or "").strip()))) for row in await conn.fetch(
            "SELECT metric_name FROM metric_rules WHERE rule_id = ANY($1::text[])",
            sorted(exact_catalog_ids)
        ))
        exact_catalog_ids={row["rule_id"] for row in await conn.fetch(
            "SELECT rule_id, metric_name FROM metric_rules WHERE rule_id = ANY($1::text[])",
            sorted(exact_catalog_ids)
        ) if (len(_norm(row["metric_name"] or "")),
              len(re.sub(r"\s+"," ",(row["metric_name"] or "").strip())))==longest}
        catalog_intended=exact_catalog_ids
    else:
        direct = await resolve_direct_fact(conn, question, doc_ids=doc_ids, page=page)
        alias_specificity = 0
        has_catalog_fact = False
        if catalog_intended:
            for row in await conn.fetch(
                "SELECT metric_name, aliases, operation FROM metric_rules WHERE rule_id = ANY($1::text[])",
                sorted(catalog_intended)
            ):
                has_catalog_fact = has_catalog_fact or row["operation"] == "fact"
                aliases=[row["metric_name"]]
                try:
                    aliases+=json.loads(row["aliases"] or "[]")
                except json.JSONDecodeError:
                    pass
                alias_specificity=max(alias_specificity,max(
                    (len(_norm(alias)) for alias in aliases
                     if _norm(alias) and _norm(alias)<=_norm(question)),default=0))
        if direct and alias_specificity < 2 and not has_catalog_fact:
            return direct[:limit]

    if not exact_catalog_ids and (dynamic_two_period := await _two_period_dynamic_intended(
            conn, question, doc_ids, page)):
        catalog_intended=dynamic_two_period
    result_intended = (set() if exact_catalog_ids else
                       await _result_period_intended(conn, question, doc_ids, page))
    if result_intended and (not catalog_intended or result_intended <= catalog_intended):
        catalog_intended = result_intended
    business_intended = set()
    if catalog_intended:
        business_intended = {row[0] for row in await conn.fetch(
            "SELECT rule_id FROM metric_rules WHERE operation='business_example' "
            "AND rule_id = ANY($1::text[])",
            sorted(catalog_intended),
        )}
    logger.info("[DOC-GM]   business_intended=%s", sorted(business_intended)[:10])
    runtime = None if business_intended or exact_catalog_ids else await _runtime_template(
        conn, question, doc_ids, page
    )
    if runtime:
        logger.info("[DOC-GM]   runtime_template matched: %s", runtime[0].get("metric", "?") if runtime else None)
    if not runtime and not catalog_intended and await reviewed_metric_matches(
            conn, question, doc_ids, page, limit=1):
        logger.info("[DOC-GM] resolve() -> [] (reviewed metric blocks fallback)  (%.0fms)",
                    (time.perf_counter() - t0) * 1000)
        return []  # caller will surface the persisted review reason; do not fall back to a raw entity
    intended = runtime[1] if runtime else catalog_intended
    if not intended:
        logger.info("[DOC-GM] resolve() -> [] (no intended metrics found)  (%.0fms)",
                    (time.perf_counter() - t0) * 1000)
        return []
    logger.info("[DOC-GM]   intended=%s", sorted(intended)[:10])
    sql="SELECT * FROM calculated_metrics WHERE calculation_status='validated'"
    args: list = []
    if doc_ids:
        args.append(list(doc_ids))
        sql += f" AND doc_id = ANY(${len(args)}::text[])"
    if page is not None:
        args.append(page)
        sql += f" AND source_page=${len(args)}"
    args.append(sorted(intended))
    sql += f" AND metric_id = ANY(${len(args)}::text[])"
    rows=[dict(row) for row in await conn.fetch(sql,*args)]
    if not rows:
        logger.info("[DOC-GM] resolve() -> [] (no validated rows for intended ids)  (%.0fms)",
                    (time.perf_counter() - t0) * 1000)
        return []
    # A workbook example that contains a multi-period sentence cannot answer a request for one
    # specific period.  Previously the absence of a matching period left the fallback row in
    # place, so a rule called merely "Total" overrode an exact Jun-26 fact.
    asked_periods=set() if exact_catalog_ids else _intended_periods(question)
    if asked_periods:
        row_ids=sorted({row["metric_id"] for row in rows})
        rule_periods={row["rule_id"]:(row["operation"],_intended_periods(
            f"{row['metric_name']} {row['applies_to']}"
        )) for row in await conn.fetch(
            "SELECT rule_id, operation, metric_name, applies_to FROM metric_rules WHERE rule_id = ANY($1::text[])",
            row_ids)}
        rows=[row for row in rows if rule_periods.get(row["metric_id"],("",set()))[0] != "business_example"
              or row["period_label"] in asked_periods
              or row["comparison_period_label"] in asked_periods
              or bool(rule_periods.get(row["metric_id"],("",set()))[1] & asked_periods)]
        if not rows:
            return []
        intended &= {row["metric_id"] for row in rows}
    # Token normalization intentionally treats punctuation and conjunctions as noise, but that
    # can make two catalogue labels such as "Accounts and customers" and "Accounts / customers"
    # look identical. When the user's wording contains an exact catalogue label, prefer it before
    # deciding whether the answer comes from a dynamic calculation or an approved workbook value.
    question_text=metric_phrase
    rule_names={row["rule_id"]:row["metric_name"] for row in await conn.fetch(
        "SELECT rule_id,metric_name FROM metric_rules WHERE rule_id = ANY($1::text[])",
        sorted(intended))}
    if not re.search(r"(?i)\b(compare|all|each|across|breakdown|rank)\b", question):
        q_words = _norm(question)
        asked_periods_inner = set() if exact_catalog_ids else _intended_periods(question)
        in_scope_ids = {row["metric_id"] for row in rows}
        relevance = {}
        for rule_id, name in rule_names.items():
            if rule_id not in in_scope_ids:
                continue
            words = _norm(name)
            has_asked_period = (not asked_periods_inner or any(
                row["metric_id"] == rule_id and (
                    row["period_label"] in asked_periods_inner
                    or row["comparison_period_label"] in asked_periods_inner
                ) for row in rows
            ))
            relevance[rule_id] = (int(has_asked_period), len(words & q_words), -len(words - q_words))
        if relevance:
            best = max(relevance.values())
            preferred_ids = {rule_id for rule_id, score in relevance.items() if score == best}
            if preferred_ids:
                rows = [row for row in rows if row["metric_id"] in preferred_ids]
                intended &= preferred_ids
    exact_ids={rule_id for rule_id,name in rule_names.items() if explicitly_asked(name)}
    # An exact short name from another report (for example "YoY growth") must not erase the
    # portfolio-scoped candidates selected above. Only exact IDs that have rows in scope compete.
    exact_ids &= {row["metric_id"] for row in rows}
    if exact_ids:
        longest=max(len(re.sub(r"\s+"," ",rule_names[rule_id].strip())) for rule_id in exact_ids)
        exact_ids={rule_id for rule_id in exact_ids
                   if len(re.sub(r"\s+"," ",rule_names[rule_id].strip()))==longest}
        rows=[row for row in rows if row["metric_id"] in exact_ids]
        intended=set(exact_ids)
    # Intent discovery can match a shorter business-example alias from another document
    # ("Advances" inside "Row gross advances").  Only treat this result set as a business-example
    # answer when such a rule actually survived the document/page filters above.
    result_metric_ids=sorted({row["metric_id"] for row in rows})
    business_ids=set()
    if result_metric_ids:
        business_ids = {row[0] for row in await conn.fetch(
            "SELECT rule_id FROM metric_rules WHERE operation='business_example' "
            "AND rule_id = ANY($1::text[])", result_metric_ids)}
    if business_ids:
        # These rows are complete, formally approved workbook outputs. Do not reinterpret words
        # such as "top", "growth", or periods in their metric names as a new runtime operation.
        selected=[row for row in rows if row["metric_id"] in business_ids]
        q=_norm(question)
        periods=set() if exact_catalog_ids else _intended_periods(question)
        if periods:
            period_rows=[row for row in selected if row["period_label"] in periods
                         or row["comparison_period_label"] in periods]
            if period_rows:
                selected=period_rows
        entity_rows=_matching_entity_rows(selected,q)
        if entity_rows:
            selected=entity_rows
        selected=selected[:limit]
        metadata={row["rule_id"]:_json_object(row["interpretation_spec"]) for row in await conn.fetch(
            "SELECT rule_id,interpretation_spec FROM metric_rules "
            "WHERE rule_id = ANY($1::text[])", sorted(business_ids))}
        for row in selected:
            row["interpretation_spec"]=metadata.get(row["metric_id"],{})
        logger.info("[DOC-GM] resolve() -> %d rows (business_example path)  (%.0fms)",
                    len(selected), (time.perf_counter() - t0) * 1000)
        return selected
    periods=(set() if exact_catalog_ids else _intended_periods(question)); q=_norm(question)
    ranked=await search(conn,question,doc_ids,page,max(limit*4,48))
    order={r["result_id"]:i for i,r in enumerate(ranked)}
    if runtime and runtime[0]["operation"] == "quarterly_sum":
        quarterly = _quarterly_rollup(question, rows, intended, q, runtime[0]["metric"])
        if quarterly:
            logger.info("[DOC-GM] resolve() -> %d rows (quarterly_sum)  (%.0fms)",
                        len(quarterly[:limit]), (time.perf_counter() - t0) * 1000)
            return quarterly[:limit]
    two_period = _two_period_calculation(question, rows, intended, q)
    if two_period:
        logger.info("[DOC-GM] resolve() -> %d rows (two_period)  (%.0fms)",
                    len(two_period[:limit]), (time.perf_counter() - t0) * 1000)
        return two_period[:limit]
    grouped=_grouped_period_comparison(question,rows,intended,q)
    if grouped:
        logger.info("[DOC-GM] resolve() -> %d rows (grouped)  (%.0fms)",
                    len(grouped[:limit]), (time.perf_counter() - t0) * 1000)
        return grouped[:limit]
    extrema=_extremum_directions(question)
    if extrema:
        selected=[]
        for metric_id in sorted(intended):
            candidates=[r for r in rows if r["metric_id"]==metric_id and r["value"] is not None]
            if periods:
                matched=[r for r in candidates if r["period_label"] in periods
                         or r["comparison_period_label"] in periods]
                if matched: candidates=matched
            entities=_matching_entity_rows(candidates,q)
            if entities:
                specificity=max(len(_norm(r["entity"])) for r in entities)
                candidates=[r for r in entities if len(_norm(r["entity"]))==specificity]
            if not candidates:
                continue
            for direction,label in extrema:
                target=(min if direction=="minimum" else max)(float(r["value"]) for r in candidates)
                tied=sorted((r for r in candidates if float(r["value"])==target),
                            key=lambda r:(r["period_key"],r["entity"]))
                for row in tied:
                    row=dict(row)
                    entity_suffix=("" if _norm(row["entity"] or "")<=_norm(row["metric_name"] or "")
                                   else f" for {row['entity']}")
                    row["answer"]=(f"The {label} {row['metric_name']}{entity_suffix} was "
                                   f"{row['formatted_value']} in {row['period_label']}.")
                    selected.append(row)
        logger.info("[DOC-GM] resolve() -> %d rows (extrema)  (%.0fms)",
                    len(selected[:limit]), (time.perf_counter() - t0) * 1000)
        return selected[:limit]
    compare_many=bool(re.search(
        r"(?i)\b(compare|all|each|across|which|rank|channels?|products?|segments?|categories|buckets?|partners?)\b",
        question,
    ))
    collection_ids = set()
    if intended:
        collection_ids = {row[0] for row in await conn.fetch(
            "SELECT rule_id FROM metric_rules WHERE operation='collection' "
            "AND rule_id = ANY($1::text[])", sorted(intended))}
    selected=[]; extras=[]
    for metric_id in sorted(intended):
        candidates=[r for r in rows if r["metric_id"]==metric_id]
        if periods:
            matched=[r for r in candidates if r["period_label"] in periods or r["comparison_period_label"] in periods]
            if matched: candidates=matched
        entity_question=(q-_norm(candidates[0]["metric_name"]) if candidates
                         and metric_id in collection_ids else q)
        entities=_matching_entity_rows(candidates,entity_question)
        if entities:
            if metric_id in collection_ids:
                candidates=entities
            else:
                specificity=max(len(_norm(r["entity"])) for r in entities)
                candidates=[r for r in entities if len(_norm(r["entity"]))==specificity]
        multiple_entities=len({r["entity"] for r in candidates if r["entity"]})>1
        candidates.sort(key=lambda r:(order.get(r["result_id"],10_000),r["entity"],r["period_key"]))
        if candidates:
            selected.append(candidates[0])
            if compare_many or metric_id in collection_ids or (multiple_entities and not entities):
                extras.extend(candidates[1:])
    selected.extend(extras)
    selected.sort(key=lambda r:order.get(r["result_id"],10_000))
    selected=_display_amount_magnitudes(question,selected[:limit])
    if selected:
        ids=sorted({r["metric_id"] for r in selected})
        metadata={r["rule_id"]:_json_object(r["interpretation_spec"]) for r in await conn.fetch(
            "SELECT rule_id,interpretation_spec FROM metric_rules WHERE rule_id = ANY($1::text[])", ids)}
        for row in selected:
            row["interpretation_spec"]=metadata.get(row["metric_id"],{})
    logger.info("[DOC-GM] resolve() -> %d rows (general path)  (%.0fms)  metric_ids=%s",
                len(selected), (time.perf_counter() - t0) * 1000,
                sorted({r["metric_id"] for r in selected})[:5])
    return selected


def context(rows: list[dict]) -> str:
    return "\n".join(
        f"- {r['metric_name']} | entity={r['entity']} | period={r['period_label']}"
        f"{(' vs '+r['comparison_period_label']) if r['comparison_period_label'] else ''} | "
        f"result={r['formatted_value']} | answer={r['answer']} | source=p.{r['source_page']}"
        for r in rows)


def deterministic_answer(rows: list[dict]) -> str:
    if not rows:
        logger.info("[DOC-GM] deterministic_answer() -> '' (no rows)")
        return ""
    logger.info("[DOC-GM] deterministic_answer() building answer from %d rows", len(rows))
    statements=[]
    for row in rows:
        spec=row.get("interpretation_spec") or {}
        values={"formatted_value":row["formatted_value"],"value":row["value"],
                "entity":row["entity"],"period_label":row["period_label"],
                "metric_name":row["metric_name"]}
        answer=""; condition=str(spec.get("success_condition","")).strip()
        match=re.fullmatch(r"(?:value\s*)?(>=|<=|>|<|==|!=)\s*(-?\d+(?:\.\d+)?)",condition)
        if match and row["value"] is not None:
            threshold=float(match.group(2)); value=float(row["value"])
            passed={">=":value>=threshold,"<=":value<=threshold,">":value>threshold,
                    "<":value<threshold,"==":value==threshold,"!=":value!=threshold}[match.group(1)]
            answer=str(spec.get("true_text" if passed else "false_text","")).format(**values)
        if not answer and spec.get("answer_template"):
            answer=str(spec["answer_template"]).format(**values)
        if not answer: answer=row["answer"] or ""
        answer=answer.rstrip()
        if answer and answer[-1] not in ".!?": answer+="."
        statements.append(answer)
    grouped=[row for row in rows if str(row.get("period_key","")).startswith("G:")]
    if len(grouped)==2 and all(row.get("value") is not None for row in grouped):
        first,second=(float(row["value"]) for row in grouped)
        if first<0<=second:
            statements.append("This represents a shift from negative to positive.")
        elif first>=0>second:
            statements.append("This represents a shift from positive to negative.")
    pages=", ".join(f"p.{p}" for p in sorted({r["source_page"] for r in rows}))
    result = (" ".join(statements) if len(statements)<=3 else "\n".join(f"- {s}" for s in statements))+f" ({pages})"
    logger.info("[DOC-GM] deterministic_answer() -> %d chars: '%s'", len(result), result[:200])
    return result


async def rebuild(conn, doc_id: str) -> int:
    """Rebuild every approved rule applicable to one document through this engine only."""
    await conn.execute("DELETE FROM calculated_metrics WHERE doc_id=$1", doc_id)
    report = await materialize(conn, doc_ids=[doc_id])
    return report["results"]


def main():
    import asyncio
    from core.db import init_db, close_db

    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="cmd",required=True)
    imp=sub.add_parser("import"); imp.add_argument("file"); imp.add_argument("--page",type=int)
    imp.add_argument("--source",default=""); imp.add_argument("--page-column",default="Page")
    imp.add_argument("--apply",action="store_true"); imp.add_argument("--dry-run",action="store_true")
    fimp=sub.add_parser("import-facts"); fimp.add_argument("file"); fimp.add_argument("--page",type=int,required=True)
    fimp.add_argument("--source",required=True); fimp.add_argument("--apply",action="store_true")
    mat=sub.add_parser("materialize"); mat.add_argument("--document",action="append"); mat.add_argument("--page",type=int,action="append")
    mat.add_argument("--rule",action="append"); mat.add_argument("--dry-run",action="store_true"); mat.add_argument("--all",action="store_true")
    approve=sub.add_parser("approve-business-examples")
    approve.add_argument("--apply",action="store_true")
    approve.add_argument("--rule",action="append")
    sub.add_parser("list")
    args=parser.parse_args()

    async def run():
        await init_db()
        try:
            if args.cmd == "import":
                rows = import_rows(args.file, default_page=args.page, source_name=args.source,
                                   apply=False)
                for r in rows:
                    print(f"{r['rule_id']:12} {r['status']:12} {r['operation']:12} {r['metric_name']}"
                          + (f" — {r['error']}" if r['error'] else ""))
                print(f"{sum(r['status']=='approved' for r in rows)}/{len(rows)} compiled")
                if args.apply and not args.dry_run:
                    await save_rules(rows)
                    report = await materialize()
                    print(json.dumps(report, indent=2))
            elif args.cmd == "import-facts":
                print(json.dumps(await import_fact_rows(
                    args.file, page=args.page, source_name=args.source, apply=args.apply), indent=2))
            elif args.cmd == "materialize":
                print(json.dumps(await materialize(
                    doc_ids=args.document, pages=args.page, rule_ids=args.rule,
                    dry_run=args.dry_run), indent=2))
            elif args.cmd == "approve-business-examples":
                print(json.dumps(await approve_business_examples(
                    apply=args.apply, rule_ids=args.rule), indent=2))
            else:
                async with store.acquire() as conn:
                    for r in await conn.fetch(
                        "SELECT rule_id,source_name,source_page,metric_name,operation,status,version "
                        "FROM metric_rules ORDER BY source_page,rule_id"):
                        print(f"{r['rule_id']:12} p.{r['source_page']:<3} v{r['version']:<3} "
                              f"{r['status']:12} {r['operation']:12} {r['metric_name']}")
        finally:
            await close_db()

    asyncio.run(run())


if __name__ == "__main__":
    main()
