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
        "in a thread. One source email is NOT a reason to write only one summary bullet "
        "or one question. "
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
        "Write one combined pre-brief using only facts written in the email bodies, MoM, "
        "or CEO notes below. Synthesize the cluster; do not list subjects or write process "
        "commentary.\n\n"
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
        "SUMMARY: Return 3-5 distinct source-backed business facts. Each pointer is one sentence: "
        "a real source fact or figure plus why it matters. Never invent numbers, percentages, "
        "crore/day amounts, placeholders, or subject restatements. If every email is only meeting "
        "logistics, return summary: [\"No relevant email discussion available for this topic\"] "
        "and empty questions/actions.\n\n"
        "QUESTIONS: Return at least 4 short questions when facts support them, otherwise fewer. "
        "Each question must cite a source figure, gap, ask, decision, or unresolved item. Banned: "
        "Can the CEO confirm..., industry benchmarks not in source, duplicate questions, meeting "
        "purpose/ID/passcode/dial-in questions, and generic risk/timeline/current-position questions.\n\n"
        "ACTION ITEMS: No maximum. Return every unique concrete deliverable explicitly requested "
        "or committed in the source, then deduplicate cluster-wide by owner/team plus deliverable. "
        "If the request/commitment is not written anywhere in BODY, MoM, or CEO note, do not create "
        "an action item. Do not turn discussion points, facts, FYIs, sender/recipient names, risks, "
        "observations, or inferred helpful steps into actions. Join/attend/prepare for the meeting "
        "is never an action item. Owner, task, and deadline must come from BODY, MoM, or CEO note; "
        "use Unassigned/null only when the source gives a task but no owner/deadline. Never fabricate "
        "owner, task, or deadline.\n\n"
        "NAMES: Use a person/team only if they appear as an actor in BODY, MoM, or CEO note. "
        "Do NOT use names from From, To, or Cc as actors.\n\n"
        "TOPIC: short label for the shared series/thread, max 8 words.\n\n"
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
