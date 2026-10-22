"""
tests/agent_email/test_meeting_assignment_eval.py
=================================================
OFFLINE EVALUATION HARNESS (diagnosis/evaluation only — changes NO production
behavior). Compares the CURRENT adaptive-Otsu keep/drop policy against a
PROPOSED confirmed/probable/unassigned policy on a labelled dataset derived
from the verified pipeline run 20260930T101713Z (backend/logs/email_agent.log).

Why this file exists
--------------------
The request asks to evaluate a proposed matching change (evidence-confirmed
"confirmed" band above Otsu, a "probable" review band below it, and "no match"
when no candidate has support) against the current pure-Otsu keep/drop, using
precision/recall/FP/FN/total-correct — and to choose the probable-band margin
from measured results rather than assuming a fixed 8-point gap.

The repo has NO labelled ground-truth set and NO matching-eval harness, so this
file supplies both, offline and deterministic:
  * It calls the REAL `compute_adaptive_cutoff` (relevance.py) for the cutoff,
    so the Otsu math under test is production code, not a copy.
  * It reuses the recorded per-email Stage-2 scores + selected meetings + a
    lightweight, transparent "evidence" signal (does the selected meeting's
    title/topic share a concrete anchor with the email subject?) so the
    proposed policy can be simulated WITHOUT the LLM/Graph/network.

LABEL PROVENANCE / HONESTY
--------------------------
Each label carries a `confidence` and `basis`. Labels marked confidence="low"
are hypotheses (the email BODY is not in the log, so the true meeting cannot be
confirmed). Metrics are reported BOTH including and excluding low-confidence
labels so a reader can see how much the conclusion rests on unverified rows.

Nothing here is imported by production code. Running it does not re-ingest,
call any model, or touch the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent_email.relevance import compute_adaptive_cutoff


# ---------------------------------------------------------------------------
# Meeting map (verified from [MEETING-HISTORY] embedded occurrence lines)
# ---------------------------------------------------------------------------
# Only the human-readable subject matters for the harness; ID suffixes are the
# tail of the Graph ids in the log.
MEETINGS = {
    "M1": "Project Orion Update Meeting",
    "M2": "project orion update meeting",
    "M3": "Wholesale liabilities & deposit review",
    "M4": "Wholesale Banking Review Sept",
    "M5": "Business Review - Credit Card & Credit Card Collection - Aug 2026",
    "M6": "Project Orion Update",
    "M7": "current account growth updates - sept 2026",
    "M8": "Industry + Competition Assessment for us to improve",
    "M9": "Wholesale Banking Review Sep-2026",
    "M10": "Business Review - Credit Card & Credit Card Collection - Aug 2026",
}

# Duplicate/near-duplicate title groups (title collapsed to a comparable form).
# Used only to report the "duplicate-title" slice.
DUPLICATE_GROUPS = [{"M5", "M10"}, {"M4", "M9"}, {"M1", "M2", "M6"}]


@dataclass
class Case:
    subject: str
    score: float                 # verified Stage-2 specific_score from the log
    selected: list[str]          # verified relevant_meeting_keys from the log
    expected: list[str]          # label: correct meeting key(s), or [] for no-match
    confidence: str              # "high" | "low"  (low = body not in log; hypothesis)
    body_present: bool           # was there real email body/facts (vs bare invite/fwd)
    basis: str = ""              # short label rationale
    # Evidence anchors actually present in the email (subject-level, verifiable
    # from the log's reason text). Empty => only generic/title similarity.
    evidence_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Labelled dataset — the Lumière + WB/Cross-Border/Orion examples under scrutiny
# ---------------------------------------------------------------------------
# score/selected are VERIFIED from email_agent.log. expected is the LABEL.
CASES: list[Case] = [
    Case(
        subject="Wholesale Liabilities & Deposit Mix Review- Discussion",
        score=91.0, selected=["M3", "M4"], expected=["M3"],
        confidence="high", body_present=True,
        basis="Title+content align verbatim with M3 (liabilities & deposit); M4 is generic WB overlap.",
        evidence_terms=["wholesale liabilities", "deposit"],
    ),
    Case(
        subject="Credit Card: Lumière: Industrial Benchmarking",
        score=82.7, selected=["M5", "M10"], expected=["M5"],
        confidence="high", body_present=True,
        basis="Credit-card review is the real match; M10 is the byte-identical duplicate of M5.",
        evidence_terms=["credit card"],
    ),
    Case(
        subject="RE: Credit Card: Lumière — customer experience, wallet share & outreach to Mid-Corp/SME",
        score=78.0, selected=["M5", "M6"], expected=["M5"],
        confidence="high", body_present=True,
        basis="Credit-card review (M5) supported; M6 Project Orion is an unsupported association.",
        evidence_terms=["credit card"],
    ),
    Case(
        subject="Fwd: Lumière Credit Card adoption metrics",
        score=77.2, selected=["M5", "M10"], expected=["M5"],
        confidence="high", body_present=True,
        basis="Credit-card adoption metrics -> M5; M10 is the duplicate.",
        evidence_terms=["credit card"],
    ),
    Case(
        subject="FW: Meeting - International Cross Border Business: Road Map",
        score=69.4, selected=["M6", "M8"], expected=[],
        confidence="high", body_present=False,
        basis="No Cross-Border meeting exists among eligible candidates -> correct answer is no-match.",
        evidence_terms=[],  # only title adjacency to M6/M8
    ),
    Case(
        subject="FW: Cross Border Strategy.pptx",
        score=77.2, selected=["M6", "M8"], expected=[],
        confidence="high", body_present=False,
        basis="No eligible Cross-Border meeting -> no-match.",
        evidence_terms=[],
    ),
    Case(
        subject="Q3 funding cost assumption for the NIM pack",
        score=62.0, selected=["M4", "M9"], expected=["M3"],
        confidence="low", body_present=True,
        basis="HYPOTHESIS: funding-cost/NIM likely belongs to deposit review (M3), not WB review twins. Body not in log.",
        evidence_terms=["funding cost", "nim"],
    ),
    Case(
        subject="FW: 14-05-2025 - Target needed for Profit & Loss Branches & Category upgradation",
        score=72.8, selected=["M6", "M8"], expected=[],
        confidence="low", body_present=True,
        basis="HYPOTHESIS: no clearly relevant eligible meeting -> likely no-match. Body not in log.",
        evidence_terms=[],
    ),
]


# ---------------------------------------------------------------------------
# Policy simulations
# ---------------------------------------------------------------------------
def current_policy(cases: list[Case]) -> dict[str, list[str]]:
    """
    CURRENT production behavior: pure adaptive Otsu keep/drop, and the kept
    email is 'assigned' to ALL meetings the LLM selected (score is max over
    selected; the whole selected set rides along).
    Returns subject -> assigned meeting keys ([] means dropped/no assignment).
    """
    scores = [c.score for c in cases]
    cutoff, _reason = compute_adaptive_cutoff(scores, floor=0.0, strictness=0.0)
    out: dict[str, list[str]] = {}
    for c in cases:
        out[c.subject] = list(c.selected) if c.score >= cutoff else []
    return out


def _has_specific_evidence(case: Case, key: str) -> bool:
    """
    Transparent stand-in for 'specific evidence supports this email->meeting
    association': the email carries at least one concrete anchor term AND that
    anchor is topically consistent with the meeting's title.
    This is deliberately conservative and title-aware so the harness does NOT
    silently reproduce pure title similarity: a meeting selected with NO
    evidence_terms never counts as confirmed.
    """
    if not case.evidence_terms:
        return False
    title = MEETINGS[key].lower()
    for term in case.evidence_terms:
        head = term.split()[0]
        if head in title:
            return True
    return False


def proposed_policy(cases: list[Case], probable_margin: float) -> dict[str, dict]:
    """
    PROPOSED behavior (simulation):
      * Confirm a selected meeting only if it has specific supporting evidence
        (evidence-confirmed) AND its score is >= Otsu cutoff.
      * Keep a 'probable' band [cutoff - margin, cutoff): route for review only
        the selected meetings that ALSO have specific evidence.
      * Among confirmed candidates, validate against alternatives: keep only the
        best-supported (here, the single evidence-consistent meeting; duplicates
        collapse to one).
      * Return no-match when no selected meeting has specific evidence.
    Returns subject -> {status, assigned:list}.
    """
    scores = [c.score for c in cases]
    cutoff, _reason = compute_adaptive_cutoff(scores, floor=0.0, strictness=0.0)

    out: dict[str, dict] = {}
    for c in cases:
        evidence_keys = [k for k in c.selected if _has_specific_evidence(c, k)]
        # collapse duplicate-title occurrences to one representative
        deduped: list[str] = []
        for k in evidence_keys:
            if not any(({k} | {d}) <= grp for grp in DUPLICATE_GROUPS for d in deduped):
                deduped.append(k)

        if not evidence_keys:
            out[c.subject] = {"status": "unassigned", "assigned": []}
        elif c.score >= cutoff:
            out[c.subject] = {"status": "confirmed", "assigned": deduped}
        elif c.score >= cutoff - probable_margin:
            out[c.subject] = {"status": "probable", "assigned": deduped}
        else:
            out[c.subject] = {"status": "unassigned", "assigned": []}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _score_assignment(expected: list[str], assigned: list[str]) -> tuple[int, int, int]:
    """
    Per-email set comparison against expected keys, with duplicate-title
    equivalence (M5==M10, etc.). Returns (tp, fp, fn) at the meeting level.
    No-match label = expected == [].
    """
    def canon(keys: list[str]) -> set[str]:
        s = set()
        for k in keys:
            grp = next((frozenset(g) for g in DUPLICATE_GROUPS if k in g), frozenset([k]))
            s.add(min(grp))  # representative
        return s

    exp, got = canon(expected), canon(assigned)
    tp = len(exp & got)
    fp = len(got - exp)
    fn = len(exp - got)
    return tp, fp, fn


def evaluate(assignments: dict, cases: list[Case], include_low: bool) -> dict:
    tp = fp = fn = 0
    correct_emails = 0
    nomatch_total = nomatch_correct = 0
    considered = 0
    for c in cases:
        if not include_low and c.confidence == "low":
            continue
        considered += 1
        got = assignments[c.subject]
        assigned = got["assigned"] if isinstance(got, dict) else got
        t, f, n = _score_assignment(c.expected, assigned)
        tp += t; fp += f; fn += n
        if t == len(set(c.expected)) and f == 0 and n == 0:
            correct_emails += 1
        if not c.expected:  # no-match case
            nomatch_total += 1
            if not assigned:
                nomatch_correct += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "emails": considered,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "correct_emails": correct_emails,
        "nomatch_accuracy": round(nomatch_correct / nomatch_total, 3) if nomatch_total else None,
    }


# ---------------------------------------------------------------------------
# Tests / reports
# ---------------------------------------------------------------------------
def _report(title: str, cur: dict, prop: dict) -> str:
    return (
        f"\n=== {title} ===\n"
        f"CURRENT (pure Otsu keep/drop): {cur}\n"
        f"PROPOSED (confirm/probable/unassigned): {prop}\n"
    )


def test_baseline_vs_proposed_high_confidence_only(capsys):
    cur_assign = current_policy(CASES)
    prop_assign = proposed_policy(CASES, probable_margin=8.0)
    cur = evaluate(cur_assign, CASES, include_low=False)
    prop = evaluate(prop_assign, CASES, include_low=False)
    print(_report("HIGH-CONFIDENCE LABELS ONLY (margin=8)", cur, prop))
    # Assert the harness is internally consistent, not that a change is 'better'.
    assert cur["emails"] == prop["emails"]
    # Proposed should not INCREASE false positives vs current on this slice.
    assert prop["fp"] <= cur["fp"]


def test_baseline_vs_proposed_all_labels(capsys):
    cur_assign = current_policy(CASES)
    prop_assign = proposed_policy(CASES, probable_margin=8.0)
    cur = evaluate(cur_assign, CASES, include_low=True)
    prop = evaluate(prop_assign, CASES, include_low=True)
    print(_report("ALL LABELS incl. low-confidence hypotheses (margin=8)", cur, prop))
    assert cur["emails"] == prop["emails"]


def test_body_present_vs_bodyless_slices(capsys):
    for slice_name, pred in (("BODY-PRESENT", True), ("BODYLESS", False)):
        subset = [c for c in CASES if c.body_present == pred]
        if not subset:
            continue
        cur = evaluate(current_policy(subset), subset, include_low=True)
        prop = evaluate(proposed_policy(subset, 8.0), subset, include_low=True)
        print(_report(f"{slice_name} slice (margin=8)", cur, prop))


def test_probable_margin_sweep(capsys):
    """Choose the probable-band margin from MEASURED results, not assumption."""
    print("\n=== PROBABLE-MARGIN SWEEP (high-confidence labels) ===")
    rows = []
    for margin in (0, 2, 4, 6, 8, 10, 12, 15):
        prop = evaluate(proposed_policy(CASES, float(margin)), CASES, include_low=False)
        rows.append((margin, prop["precision"], prop["recall"],
                     prop["fp"], prop["fn"], prop["correct_emails"]))
        print(f"margin={margin:>2}  P={prop['precision']:.3f}  R={prop['recall']:.3f}  "
              f"FP={prop['fp']}  FN={prop['fn']}  correct={prop['correct_emails']}")
    # A margin only helps if it recovers a correct match without adding an FP.
    # This documents that the sweep ran; it does not hardcode an 'optimal' value.
    assert rows, "sweep produced no rows"
