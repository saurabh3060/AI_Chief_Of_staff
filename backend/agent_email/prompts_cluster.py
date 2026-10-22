"""
agent_email/prompts_cluster.py
=============================
Cluster-synthesis prompt + parser: the CEO pre-meet brief built over ALL
emails in one embedding cluster. This is what the PreMeetingBrief surfaces
as summary / top questions / action items.

Split out of ``agent_email/prompts.py`` so each prompt family stays under
the module size limit. Shared JSON/regex/quality helpers come from
``agent_email/prompts_common``. ``prompts.py`` re-exports the public names
here so existing imports keep working.
"""

import re

from core.logging_config import get_agent_logger

from agent_email.prompts_common import (
    _MEETING_LOGISTICS_LEAK_RE,
    _NON_ACTION_RE,
    _NON_EXECUTIVE_SUMMARY_RE,
    _PLACEHOLDER_FIGURE_RE,
    _PROCESS_META_SUMMARY_RE,
    _PURE_MEETING_INVITE_PLACEHOLDER_RE,
    dedupe_preserving_order,
    is_low_confidence_text,
    quality_filter_questions,
    repair_llm_json_text,
    sanitize_llm_json_text,
    strip_meeting_boilerplate,
)

import json

logger = get_agent_logger("agent_email")


CLUSTER_SYNTHESIS_SYSTEM_PROMPT = (
    "You write a CEO pre-meet brief from the email bodies and MoM in ONE cluster. "
    "A cluster may be a single mail that is still a long reply thread - unpack it "
    "into 3-5 pointers even when there is only one source email. "
    "Summary: 3-5 crisp pointers (one fact each). Each pointer must use a source figure "
    "or a source disagreement, then say why it matters in the meeting. Questions: aim for AT LEAST 4 "
    "short questions (preferably 5-6 when substantive facts exist) taken from the mail content itself. "
    "Action Items: Exhaustively extract every source-backed concrete deliverable from all email bodies and MoM, "
    "then deduplicate the complete cluster before returning JSON. Return each logical action exactly once. "
    "Treat exact repeats and paraphrases as the same action when they have the same accountable owner/team and intended deliverable, "
    "including repetitions in quoted threads, forwards, reminders, and later email versions. Do not impose a numeric cap. "
    "TREAT MEETING/CALENDAR INVITES AS CARRYING NO BRIEFING CONTENT. If, after removing all "
    "Teams/Outlook/Webex scheduling logistics (meeting ID, passcode, dial-in number, video ID, "
    "join link, organizer options), a cluster has no real content left in ANY of its source "
    "emails - i.e. every email in the cluster is purely a meeting invite/forward with no "
    "substantive discussion of its own - the entire output must be: "
    "summary: [\"No relevant email discussion available for this topic\"], "
    "top_questions: [], action_items: []. Do not write a bullet describing it as a meeting "
    "invite, and do not ask a question about the meeting's purpose, ID, passcode, or logistics. "
    "NEVER extract, quote, or paraphrase a meeting ID, passcode, dial-in number, or organizer "
    "contact detail into summary/top_questions/action_items under any circumstances, even when "
    "other real content exists in the same cluster. "
    "Banned questions: 'Can the CEO "
    "confirm...', 'Will the CEO commit...', 'What is the purpose of the meeting...', 'What is "
    "the meeting ID...', 'What is the passcode...'. Never ask the CEO to confirm a target already "
    "in the mail. Names: only people named as "
    "actors in the BODY or MoM (e.g. signed off, 'Amit's numbers'). NEVER use From/To/Cc "
    "header names. Copy digits and units exactly. NEVER invent metrics. Never write "
    "process filler ('the explanation provided is...', 'the operational blocker is...'). "
    "Never use 'I', 'we', 'you', or 'our'. Department is already assigned - copy it. "
    "Return ONLY valid JSON, no markdown fences."
)


_FIGURE_PHRASE_PATTERNS = (
    re.compile(
        r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d+)?(?:\s*(?:[-–]|to)\s*[\d,]+(?:\.\d+)?)?\s*(?:Cr|cr|crore|bps)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b[\d,]+(?:\.\d+)?\s*(?:days|bps|%|Cr|crore)\b", re.IGNORECASE),
)


def _harvest_figure_phrases(text: str, *, limit: int = 40) -> list[str]:
    """Collect numeric phrases from cluster source text so the model can copy them exactly."""
    phrases: list[str] = []
    seen: set[str] = set()
    for pat in _FIGURE_PHRASE_PATTERNS:
        for match in pat.finditer(text or ""):
            phrase = re.sub(r"\s+", " ", match.group(0)).strip()
            key = phrase.lower()
            if key and key not in seen:
                seen.add(key)
                phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def _numeric_tokens(text: str) -> set[str]:
    """Normalised digit strings (commas stripped) appearing in text."""
    return {token.replace(",", "") for token in re.findall(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+|\d+", text or "")}


def _is_material_number(token: str) -> bool:
    """True for figures worth grounding (ignore lone 1-digit counters)."""
    if "." in token:
        return True
    if len(token) >= 2:
        return True
    return False


def _drop_ungrounded_numbers(items: list[str], source_text: str) -> list[str]:
    """Drop bullets/questions that introduce digits not present in the source emails."""
    if not source_text:
        return items
    allowed = _numeric_tokens(source_text)
    kept: list[str] = []
    for item in items:
        extra = {n for n in _numeric_tokens(item) if n not in allowed and _is_material_number(n)}
        if extra:
            continue
        kept.append(item)
    return kept


def _is_subject_restatement(bullet: str, source_text: str) -> bool:
    """True when a summary bullet is just an email subject restated."""
    subjects = re.findall(r"^Subject:\s*(.+)$", source_text or "", re.MULTILINE)
    normalised_bullet = re.sub(r"[^a-z0-9]+", " ", bullet.lower()).strip()
    if not normalised_bullet or len(normalised_bullet.split()) > 14:
        return False
    for subject in subjects:
        normalised_subject = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
        if normalised_subject and (
            normalised_bullet == normalised_subject
            or normalised_bullet in normalised_subject
            or normalised_subject in normalised_bullet
        ):
            return True
    return False


def _format_department_scope_block(departments: list, phrases_by_dept: dict[str, list[str]] | None = None) -> str:
    phrases_by_dept = phrases_by_dept or {}
    lines: list[str] = []
    for i, dept in enumerate(departments, start=1):
        summary = getattr(dept, "summary", "") or ""
        lines.append(f"{i}. {dept.name}")
        if summary:
            lines.append(f"   Scope: {summary}")
        extra = [p for p in phrases_by_dept.get(dept.name, []) if str(p).strip()]
        if extra:
            lines.append("   Examples: " + "; ".join(extra))
    return "\n".join(lines) if lines else "(no departments configured)"


def build_cluster_synthesis_prompt(
    appended_emails: str,
    departments: list,
    assigned_department: str = "",
    email_count: int = 0,
    phrases_by_dept: dict[str, list[str]] | None = None,
) -> str:
    # Strip meeting-invite logistics before the cluster text reaches the LLM
    # (mirrors the single-email path), so a meeting ID/passcode can never be
    # quoted into a cluster-level summary bullet or question.
    appended_emails = strip_meeting_boilerplate(appended_emails)
    dept_block = _format_department_scope_block(departments, phrases_by_dept)
    n = email_count if email_count > 0 else appended_emails.count("--- Email ")
    assigned = (assigned_department or "").strip()
    assigned_rule = (
        f"The department for this cluster is already assigned: {assigned}. "
        "Copy that string EXACTLY into 'department'. Do not pick a different department, "
        "and do not default to the first-listed name.\n\n"
        if assigned
        else "Assign exactly one department from the list; copy the name verbatim. Do not default to the first-listed name.\n\n"
    )
    figures = _harvest_figure_phrases(appended_emails)
    figures_block = (
        "\n".join(f"- {phrase}" for phrase in figures)
        if figures
        else "(no numeric figures found in this cluster - do not invent any)"
    )

    # Detect if CEO notes are present in the appended text
    has_ceo_notes = "CEO Note" in appended_emails

    prompt_parts = [
        "=== DEPARTMENT SCOPE (for context only; do not retag if a department is already assigned) ===\n"
        f"{dept_block}\n\n"
        f"{assigned_rule}"
        f"This cluster contains {n} email(s). You must use ALL of them, including every reply "
        "in a thread. A single source email is NOT a reason to write only one summary bullet "
        "or one question - unpack the figures, disagreements, and asks inside it. "
        "Do not let the longest body or MoM attachment dominate. "
        "Ignore meeting-invite logistics (Teams/Webex IDs, passcodes, dial-in numbers). "
        "If an email adds no new facts, skip it rather than padding.\n\n"
        "=== FIGURES PRESENT IN THIS CLUSTER (copy these digits and units exactly; invent none) ===\n"
        f"{figures_block}\n"
        "If a figure is not in this list and not in the emails/MoM below, do not use it. "
        "Never write X, Y, or placeholder crore/day amounts.\n\n"
        "=== EMAILS IN THIS CLUSTER (full body + full MoM where present) ===\n"
        f"{appended_emails}\n\n"
    ]

    if has_ceo_notes:
        prompt_parts.extend([
            "=== CEO NOTES (for action items extraction only) ===\n"
            "The following notes were written by the CEO after this meeting. "
            "Extract action items from these notes with higher priority. "
            "If an action item from CEO notes appears in email thread too, deduplicate "
            "by keeping the version with the clearest deadline and most specific owner. "
            "Include notes as a source when citing action items.\n\n"
        ])

    prompt_parts.extend([
        "=== YOUR TASK ===\n"
        "Write one combined pre-brief as 3-5 short pointers plus 3-4 short questions. "
        "Each summary pointer is one sentence: source fact (with the real figure) then "
        "the implication. Do not write essays or process commentary. Synthesize the "
        "thread; do not list subjects. Use only facts written in the bodies/MoM below.\n\n"
        "=== RESPONSE FORMAT ===\n"
        "Return ONLY valid JSON with exactly this structure:\n"
        "{\n"
        '  "department": "<exact assigned department name>",\n'
        '  "topic": "<shared series/thread label, max 8 words>",\n'
        '  "summary": ["<one crisp pointer with a source figure>"],\n'
        '  "top_questions": ["<short question taken from a mail fact>"],\n'
        '  "action_items": [{"owner": "<name from body/MoM or Unassigned>", "task": "<specific deliverable>", '
        '"deadline": "<date as written, or null>"}]\n'
        "}\n\n"
        "RULES:\n"
        "SUMMARY: Extract EXACTLY 3-5 distinct business facts (one per pointer). Split a reply "
        "thread into separate facts. If EVERY email in this cluster is nothing but a "
        "Teams/Outlook/Webex meeting invite (pure scheduling logistics, no substantive body "
        "text in any of them), return summary: [\"No relevant email discussion available for "
        "this topic\"] instead - do not write a bullet describing it as an invite, and never "
        "put a meeting ID, passcode, dial-in number, or organizer detail in a bullet. In that "
        "same no-content case, top_questions and action_items must both be empty arrays. "
        "Each pointer: one sentence, source number or named disagreement, why it matters. "
        "Banned phrasing: 'The explanation given/provided is...', 'The operational blocker is...'. "
        "If the source has a number, the bullet must carry that number. Never invent % or Cr. Direct third-person. "
        "Never restate the subject line.\n\n"
        "QUESTIONS: Generate AT LEAST 4 short questions (aim for 5-6 when substantive facts support them). "
        "Cite a figure or unresolved ask that is actually written. Write them as they would be asked in the room. "
        "Banned: addressing the CEO ('Can the CEO confirm...'); confirming a target already in mail; industry-benchmark "
        "questions not in source; two questions that restate each other; asking about the meeting's own purpose, "
        "ID, passcode, or dial-in logistics. Return an empty array if this cluster is nothing but a meeting "
        "invite with no substantive content to question.\n\n"
        "ACTION ITEMS: First scan every email body, MoM attachment, and CEO note and collect every source-backed concrete deliverable; "
        "then perform a cluster-wide semantic deduplication pass before producing the final JSON. There is No maximum and no target count: "
        "return all unique actions supported by the cluster, but return each logical action EXACTLY ONCE. "
        "An action's identity is its accountable owner/team plus its intended deliverable, not the source in which it appears. "
        "Exact repeats and differently worded versions of the same owner-task pair are duplicates. Repeated quotations, forwarded copies, "
        "reminders, acknowledgements, and deadline restatements do not create new actions. Merge those into one item using the clearest, "
        "most specific source wording and the most explicit source-backed deadline. "
        "Do not expand one request into separate actions for every person named, copied, or asked to review it; create separate items only "
        "when the source explicitly assigns materially different deliverables to those owners. "
        "For example, repeated variants of 'Bharat: provide a view on the funding cost assumption' are ONE action, while Bharat preparing "
        "the assumption and Sandeep approving it are two actions only if both responsibilities are explicitly stated. "
        "Include deliverables, analyses, approvals, data requests, decisions, and follow-ups only when the source actually requests or commits "
        "to the work; do not turn a discussion point, factual statement, sender/recipient name, or inferred helpful step into an action. "
        "Owner: copy exactly as written in body/MoM/CEO note; use 'Unassigned' if no owner is named. Deadline: copy the most explicit deadline as written, "
        "or null if none is stated. Banned: meeting attendance, preparation, and dial-in logistics. Never fabricate an owner, task, or deadline.\n\n"
        "NAMES: Use a name only if it appears as an actor in BODY, MoM, or CEO note (signed-off, 'Amit's numbers', 'Credit Ops must...'). "
        "Do NOT take names from From/To/Cc. If the only name is a header, write the function (Credit, Credit Ops) or Unassigned.\n\n"
        "TOPIC: short label for the shared series/thread.\n\n"
        "Return ONLY the JSON object, nothing else.\n"
    ])

    return "".join(prompt_parts)


def parse_cluster_synthesis_response(response_text: str, source_text: str = "") -> dict:
    """Parse cluster synthesis LLM JSON with tolerant repair and source grounding."""
    text = response_text.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    else:
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            text = obj_match.group(0).strip()

    text = sanitize_llm_json_text(text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        repaired = repair_llm_json_text(text)
        try:
            data = json.loads(repaired)
            logger.info("parse_cluster_synthesis_response: recovered via repair pass")
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse cluster synthesis JSON: {first_error}\nResponse: {text[:200]}"
            ) from first_error

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    summary = data.get("summary", [])
    questions = data.get("top_questions", data.get("questions", []))
    actions = data.get("action_items", [])
    if not isinstance(summary, list):
        summary = [str(summary)] if summary else []
    if not isinstance(questions, list):
        questions = [str(questions)] if questions else []
    if not isinstance(actions, list):
        actions = []

    logger.info(
        "parse_cluster_synthesis_response: raw LLM output: summary=%d questions=%d actions=%d",
        len(summary), len(questions), len(actions)
    )

    summary_lines = [re.sub(r"\s+", " ", str(s)).strip() for s in summary if str(s).strip()]
    summary_lines = [
        s for s in summary_lines
        if not _NON_EXECUTIVE_SUMMARY_RE.search(s)
        and not _PROCESS_META_SUMMARY_RE.search(s)
        and not _PLACEHOLDER_FIGURE_RE.search(s)
        and not _PURE_MEETING_INVITE_PLACEHOLDER_RE.search(s)
        and not _MEETING_LOGISTICS_LEAK_RE.search(s)
        and not _is_subject_restatement(s, source_text)
    ]
    summary_lines = _drop_ungrounded_numbers(dedupe_preserving_order(summary_lines), source_text)
    summary_lines = [s for s in summary_lines if not is_low_confidence_text(s)][:5]

    question_lines = [re.sub(r"\s+", " ", str(q)).strip() for q in questions if str(q).strip()]
    question_lines = quality_filter_questions(question_lines)
    question_lines = _drop_ungrounded_numbers(question_lines, source_text)
    question_lines = [q for q in question_lines if not is_low_confidence_text(q)][:6]

    filtered_actions = []
    for item in actions:
        if isinstance(item, dict):
            task = str(item.get("task", "")).strip()
            owner = str(item.get("owner", "")).strip()
            blob = f"{owner} {task}"
            if task and not _NON_ACTION_RE.search(task) and not is_low_confidence_text(blob):
                filtered_actions.append(item)
        elif isinstance(item, str) and item.strip() and not _NON_ACTION_RE.search(item) and not is_low_confidence_text(item):
            filtered_actions.append(item)

    # If every summary bullet was filtered out (or the model returned none),
    # surface the fallback sentence and force questions/actions empty too.
    if not summary_lines:
        summary_lines = ["No relevant email discussion available for this topic"]
        question_lines = []
        filtered_actions = []

    return {
        "department": str(data.get("department", "")).strip(),
        "topic": str(data.get("topic", "")).strip(),
        "summary": summary_lines,
        "top_questions": question_lines,
        "action_items": filtered_actions,
    }
