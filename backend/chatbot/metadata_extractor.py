"""
chatbot/metadata_extractor.py
================================
Rule-based metadata extractor for email chunks.

Extracts key signals from email subject + body and returns a compact
metadata summary string that gets prepended to the embedding text.
This improves retrieval for emails with generic subject lines like
'Actions from today's WB Review' by surfacing the actual content
(figures, topics, people, dates) at embedding time.

The metadata string is ONLY used for embedding - the LLM still
receives the original clean email body for answer generation.

Ported from the standalone chatbot_final project, unchanged apart from
using this project's centralised logger.
"""

import re
from typing import Optional

from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")


# ── Department / topic keywords ───────────────────────────────
_TOPIC_MAP = [
    (["wholesale banking", "wb review", "wholesale bank", "wb mbr",
      "wholesale liabilities", "corporate credit", "corporate underwriting"],
     "Wholesale Banking"),
    (["retail banking", "retail branch", "branch banking"],
     "Retail Banking"),
    (["credit card", "cards", "credit cards"],
     "Credit Cards"),
    (["liabilities", "deposits", "figu", "casa", "fd ", "fixed deposit",
      "deposit mix", "balance sheet"],
     "Liabilities & Deposits"),
    (["orion", "project orion", "underwriting digitisation", "tat", "turnaround time",
      "corporate credit tat", "underwriting delay"],
     "Underwriting / ORION"),
    (["trade finance", "trade", "lc ", "letter of credit", "gtl"],
     "Trade Finance"),
    (["treasury", "fx", "forex", "interest rate"],
     "Treasury"),
    (["risk", "npa", "credit quality", "portfolio quality", "slippage"],
     "Credit Risk"),
    (["human resources", "hr ", "talent", "learning", "training", "nexus"],
     "HR / Learning"),
    (["compliance", "rbi", "regulatory", "audit"],
     "Compliance / Regulatory"),
]


# ── Month name → number map ───────────────────────────────────
_MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _extract_topic(text: str) -> Optional[str]:
    """Return the best matching department/topic label."""
    lower = text.lower()
    for keywords, label in _TOPIC_MAP:
        if any(kw in lower for kw in keywords):
            return label
    return None


def _extract_dates(text: str) -> list[str]:
    """Extract month-year mentions like 'September 2026', 'Aug 2026', '09-2026'."""
    lower = text.lower()
    found = []

    # Pattern: "Month YYYY" or "Month'YY"
    for month_name, month_num in _MONTH_MAP.items():
        # e.g. "september 2026" or "september'26"
        pattern = rf"\b{month_name}\s*['\-]?\s*(20\d{{2}}|\d{{2}})\b"
        matches = re.findall(pattern, lower)
        for yr in matches:
            year = f"20{yr}" if len(yr) == 2 else yr
            found.append(f"{month_name.capitalize()} {year}")

    # Pattern: numeric "MM-YYYY" or "MM/YYYY"
    numeric = re.findall(r"\b(0[1-9]|1[0-2])[-/](20\d{2})\b", text)
    for m, y in numeric:
        month_name = [k for k, v in _MONTH_MAP.items() if v == m and len(k) > 3]
        if month_name:
            found.append(f"{month_name[0].capitalize()} {y}")

    # Deduplicate preserving order
    seen = set()
    unique = []
    for d in found:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _extract_figures(text: str) -> list[str]:
    """Extract currency figures like 'Rs 3,680 Cr', '₹860 Cr', percentages, and day counts."""
    figures = []

    # Rs / INR figures: "Rs 3,680 Cr", "₹860 Cr", "Rs. 1,200 Lakh"
    currency = re.findall(
        r"(?:Rs\.?\s*|₹\s*)([\d,]+(?:\.\d+)?)\s*(?:Cr|cr|Lakh|lakh|L\b)",
        text
    )
    units = re.findall(
        r"(?:Rs\.?\s*|₹\s*)[\d,]+(?:\.\d+)?\s*(Cr|cr|Lakh|lakh|L\b)",
        text
    )
    for amt, unit in zip(currency, units):
        figures.append(f"Rs {amt} {unit}")

    # Percentages: "19%", "4.5%"
    pcts = re.findall(r"\b(\d+(?:\.\d+)?)\s*%", text)
    for p in pcts[:5]:   # cap at 5
        figures.append(f"{p}%")

    # Day counts: "9.5 days", "15.2 days"
    days = re.findall(r"\b(\d+(?:\.\d+)?)\s+days?\b", text, re.IGNORECASE)
    for d in days[:5]:
        figures.append(f"{d} days")

    # Account/unit counts: "45 accounts", "5 branches"
    counts = re.findall(
        r"\b(\d+)\s+(accounts?|branches?|proposals?|cases?)\b",
        text, re.IGNORECASE
    )
    for n, unit in counts[:5]:
        figures.append(f"{n} {unit}")

    # Deduplicate
    seen = set()
    unique = []
    for f in figures:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique[:12]   # cap total figures at 12


def _extract_people(text: str) -> list[str]:
    """Extract names that appear with a role indicator or ownership marker."""
    # Look for patterns like "Owner: Sameer Kapoor", "Kavya Iyer to", "Amit Sethi:"
    names = []

    # "Owner: Name" / "Action: Name" / "Due: Name"
    owner_pattern = re.findall(
        r"(?:Owner|Action|Assigned to|From|By)\s*[:\-]\s*([A-Z][a-z]+ [A-Z][a-z]+)",
        text
    )
    names.extend(owner_pattern)

    # Capitalised Name + common title/role
    role_pattern = re.findall(
        r"([A-Z][a-z]+ [A-Z][a-z]+)\s*(?:,\s*(?:MD|CEO|GM|Head|DGM|AGM|VP|SVP|EVP|Director))",
        text
    )
    names.extend(role_pattern)

    # Deduplicate
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique[:6]   # cap at 6 names


def _extract_review_type(subject: str, body: str) -> Optional[str]:
    """Detect the type of review/meeting from subject + body."""
    combined = (subject + " " + body[:500]).lower()
    if "mom" in combined or "minutes of meeting" in combined or "minutes" in combined:
        return "Meeting Minutes"
    if "action" in combined and ("review" in combined or "mbr" in combined):
        return "Review Action Items"
    if "follow" in combined and "up" in combined:
        return "Follow-up"
    if "balance sheet" in combined:
        return "Balance Sheet Update"
    if "pilot" in combined or "status update" in combined:
        return "Pilot Status Update"
    if "review" in combined or "mbr" in combined:
        return "Review Summary"
    return None


def extract_metadata(subject: str, raw_content: str) -> str:
    """Extract metadata from an email and return a compact summary string.

    This string is prepended to the embedding text to improve retrieval.
    It is NOT shown to the LLM - only used at embedding time.

    Example output:
        [METADATA]
        Topic: Wholesale Banking | Underwriting / ORION
        Type: Review Action Items
        Dates mentioned: September 2026
        Key figures: Rs 3,680 Cr, 45 accounts, Rs 860 Cr, 19%, 9.5 days
        People: Sameer Kapoor, Kavya Iyer
        [/METADATA]
    """
    combined = subject + "\n" + raw_content

    topic       = _extract_topic(combined)
    dates       = _extract_dates(combined)
    figures     = _extract_figures(combined)
    people      = _extract_people(combined)
    review_type = _extract_review_type(subject, raw_content)

    lines = ["[METADATA]"]

    if topic:
        lines.append(f"Topic: {topic}")
    if review_type:
        lines.append(f"Type: {review_type}")
    if dates:
        lines.append(f"Dates mentioned: {', '.join(dates[:4])}")
    if figures:
        lines.append(f"Key figures: {', '.join(figures)}")
    if people:
        lines.append(f"People: {', '.join(people)}")

    lines.append("[/METADATA]")

    metadata_str = "\n".join(lines)

    logger.debug(
        "Metadata for '%s': topic=%s, dates=%s, figures=%d, people=%d",
        subject[:50], topic, dates[:2], len(figures), len(people),
    )

    return metadata_str
