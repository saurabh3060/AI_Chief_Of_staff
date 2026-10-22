"""
clean_bu_data.py

One-off cleanup script for the bu_emails.csv / department_views.csv pairs
under backend/data and backend/data3.

What it does (per folder passed in):
  1. Removes duplicate email rows (same subject + content, first one kept).
  2. Strips stray "16 Sept" / "16th September" / "16-Sep" style date stamps
     out of email_subject (these were mistakenly baked into subjects).
  3. Re-routes department/topic for rows that were obviously mis-classified
     (e.g. a credit-card email that ended up filed under BB&RL) using a
     keyword classifier over the subject + content.
  4. Cleans up vague "what is the next step for <person>" / "what are the
     next steps for <person>" style questions from top_questions and the
     matching vague entries from action_items, replacing them with a
     concrete fallback tied to the row's topic/department when the list
     would otherwise go empty.
  5. Regenerates department_views.csv from the cleaned bu_emails.csv so the
     rollup view matches the corrected per-email data.

Usage:
    python backend/scripts/clean_bu_data.py backend/data
    python backend/scripts/clean_bu_data.py backend/data3
    python backend/scripts/clean_bu_data.py backend/data backend/data3

The script edits files in place. It writes backups next to the originals
first (bu_emails.csv.bak / department_views.csv.bak) so you can diff or
revert if something looks wrong.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone

FIELDNAMES = [
    "unique_id", "bu_id", "message_id", "email_subject", "email_content",
    "attachment_type", "attachment_content", "summary", "action_items",
    "top_questions", "received_at", "from_address", "from_name",
    "department", "topic",
]

DEPT_VIEW_FIELDNAMES = [
    "department", "topics", "master_summary", "master_questions",
    "master_action_items", "updated_at",
]

# --- Keyword classifier used to fix obviously wrong department routing ---
DEPARTMENT_KEYWORDS = OrderedDict([
    ("Credit Cards", [
        "credit card", "card acquisition", "card activation", "emi",
        "rewards", "interchange", "mbr review credit card",
    ]),
    ("Retail Advances - Excluding Credit Cards", [
        "home loan", "lap", "personal loan", "secured and unsecured",
        "retail assets", "category upgradation", "profit & loss branches",
    ]),
    ("BB&RL", [
        "branch banking", "vrm", "casa", "current account", "ca growth",
        "liability", "digital products", "business banking", "cms",
    ]),
    ("Wholesale Banking", [
        "credit tat", "sanction tat", "committee level", "wholesale",
        "zonal credit", "large-ticket", "watchlist account",
    ]),
])

# Any of these words present in a subject/content is a strong signal that
# the email is fundamentally about credit cards, regardless of where it
# was originally filed.
CREDIT_CARD_STRONG_SIGNAL = re.compile(r"credit\s*card", re.IGNORECASE)

# --- Date-stamp junk to strip out of subjects ---
DATE_JUNK_PATTERNS = [
    r"\b16\s*(?:th)?\s*sept(?:ember)?\s*['’]?\s*(?:20)?26\b",
    r"\b16\s*(?:th)?\s*sept(?:ember)?\b",
    r"\bsept(?:ember)?\s*16(?:th)?\s*['’]?\s*(?:20)?26\b",
    r"\bsept(?:ember)?\s*16(?:th)?\b",
]
DATE_JUNK_RE = re.compile("|".join(DATE_JUNK_PATTERNS), re.IGNORECASE)

# --- Vague "next step for <name>" style question / action detection ---
VAGUE_QUESTION_RE = re.compile(
    r"what\s+(?:is|are)\s+the\s+next\s+step(?:s)?\s+for\s+[\w .'-]+\??",
    re.IGNORECASE,
)
VAGUE_ACTION_RE = re.compile(
    r"^[\w .'-]+:\s*(?:review|next\s+step(?:s)?)\s*(?:for)?\s*[\w .'-]*\(by\s*TBA\)\s*$",
    re.IGNORECASE,
)


def normalize_key(subject: str, content: str) -> str:
    """Build a dedup key from subject + content, ignoring whitespace noise."""
    s = re.sub(r"\s+", " ", (subject or "").strip().lower())
    c = re.sub(r"\s+", " ", (content or "").strip().lower())
    return f"{s}||{c[:500]}"


def clean_subject(subject: str) -> str:
    if not subject:
        return subject
    cleaned = DATE_JUNK_RE.sub("", subject)
    # collapse leftover punctuation/whitespace artifacts from the removal
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"[-–—:]\s*$", "", cleaned.strip())
    cleaned = re.sub(r"^\s*[-–—:]\s*", "", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return cleaned.strip()


def classify_department(subject: str, content: str, current_dept: str) -> str:
    text = f"{subject or ''} {content or ''}".lower()

    # Strong override: credit-card content should never sit under BB&RL
    # (or any other department) if it clearly mentions credit cards.
    if CREDIT_CARD_STRONG_SIGNAL.search(text) and current_dept != "Credit Cards":
        return "Credit Cards"

    if current_dept:
        return current_dept

    scores = {}
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        scores[dept] = sum(1 for kw in keywords if kw in text)
    best_dept, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return current_dept  # leave as-is if nothing matches
    return best_dept


def clean_top_questions(raw: str, department: str, topic: str) -> str:
    if not raw:
        return raw
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    kept = [p for p in parts if not VAGUE_QUESTION_RE.search(p)]
    if not kept:
        subject_hint = topic or department or "this item"
        kept = [f"What are the key open items and owners for {subject_hint}?"]
    return "|".join(kept)


def clean_action_items(raw: str, department: str, topic: str) -> str:
    if not raw:
        return raw
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    kept = [p for p in parts if not VAGUE_ACTION_RE.match(p)]
    if not kept:
        subject_hint = topic or department or "this item"
        kept = [f"Unassigned: Confirm concrete next steps for {subject_hint} - TBD"]
    return "|".join(kept)


def process_bu_emails(folder: str) -> list[dict]:
    path = os.path.join(folder, "bu_emails.csv")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return []

    backup_path = path + ".bak"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  backup written: {backup_path}")

    seen = set()
    cleaned_rows = []
    dup_count = 0

    for row in rows:
        key = normalize_key(row.get("email_subject", ""), row.get("email_content", ""))
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)

        row["email_subject"] = clean_subject(row.get("email_subject", ""))

        new_dept = classify_department(
            row.get("email_subject", ""),
            row.get("email_content", ""),
            row.get("department", ""),
        )
        row["department"] = new_dept

        row["top_questions"] = clean_top_questions(
            row.get("top_questions", ""), new_dept, row.get("topic", "")
        )
        row["action_items"] = clean_action_items(
            row.get("action_items", ""), new_dept, row.get("topic", "")
        )

        cleaned_rows.append(row)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"  bu_emails.csv: {len(rows)} rows -> {len(cleaned_rows)} rows "
          f"({dup_count} duplicates removed)")
    return cleaned_rows


def regenerate_department_views(folder: str, cleaned_rows: list[dict]) -> None:
    path = os.path.join(folder, "department_views.csv")
    backup_path = path + ".bak"

    if os.path.exists(path) and not os.path.exists(backup_path):
        with open(path, "r", encoding="utf-8-sig") as f:
            original = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(original)
        print(f"  backup written: {backup_path}")

    by_dept = defaultdict(lambda: {
        "topics": OrderedDict(),
        "summary": OrderedDict(),
        "questions": OrderedDict(),
        "actions": OrderedDict(),
    })

    for row in cleaned_rows:
        dept = (row.get("department") or "").strip()
        if not dept:
            continue
        bucket = by_dept[dept]

        topic = (row.get("topic") or "").strip()
        if topic:
            bucket["topics"][topic] = True

        for s in (row.get("summary") or "").split("\n"):
            s = s.strip().lstrip("•").strip()
            if s:
                bucket["summary"][s] = True

        for q in (row.get("top_questions") or "").split("|"):
            q = q.strip()
            if q:
                bucket["questions"][q] = True

        for a in (row.get("action_items") or "").split("|"):
            a = a.strip()
            if a:
                bucket["actions"][a] = True

    now = datetime.now(timezone.utc).isoformat()
    out_rows = []
    for dept, bucket in by_dept.items():
        out_rows.append({
            "department": dept,
            "topics": "|".join(bucket["topics"].keys()),
            "master_summary": "|".join(bucket["summary"].keys()),
            "master_questions": "|".join(bucket["questions"].keys()),
            "master_action_items": "|".join(bucket["actions"].keys()),
            "updated_at": now,
        })

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEPT_VIEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"  department_views.csv regenerated: {len(out_rows)} departments")


def main(folders: list[str]) -> None:
    if not folders:
        folders = ["backend/data", "backend/data3"]

    for folder in folders:
        print(f"Processing {folder} ...")
        cleaned_rows = process_bu_emails(folder)
        if cleaned_rows:
            regenerate_department_views(folder, cleaned_rows)
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
