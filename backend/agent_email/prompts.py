"""
agent_email/prompts.py
=========================
Single-email summariser prompt for the Email & Summary agent, plus the
public prompt surface for the whole agent.
 
This module owns the per-email briefing prompt (``BU_EMAIL_SYSTEM_PROMPT``,
``build_bu_email_prompt``, ``parse_bu_email_response``,
``evaluate_bu_response_quality``). Shared JSON/regex/quality helpers live in
``prompts_common``; the structured-context extractors live in
``prompts_semantics``; the cluster-synthesis prompt lives in
``prompts_cluster``. All of those public names are re-exported here so
existing imports such as ``from agent_email.prompts import X`` keep working
unchanged, regardless of which split module X now lives in.
"""
 
import json
import re
 
from core.logging_config import get_agent_logger
 
# Import fact extraction module
from agent_email import fact_extraction
 
# Shared helpers (JSON sanitize/repair, quality filters, boilerplate strip).
from agent_email.prompts_common import (
    _NON_ACTION_RE,
    _NON_EXECUTIVE_QUESTION_RE,
    _NON_EXECUTIVE_SUMMARY_RE,
    _PLACEHOLDER_FIGURE_RE,
    dedupe_preserving_order as _dedupe_preserving_order,
    quality_filter_actions as _quality_filter_actions,
    quality_filter_questions as _quality_filter_questions,
    quality_filter_summary as _quality_filter_summary,
    repair_llm_json_text as _repair_llm_json_text,
    sanitize_and_drop_empty as _sanitize_and_drop_empty,
    sanitize_and_normalize_action_items as _sanitize_and_normalize_action_items,
    sanitize_llm_json_text as _sanitize_llm_json_text,
    strip_meeting_boilerplate as _strip_meeting_boilerplate,
)
 
# Re-exported structured-context extractors (used by service/clustering/relevance).
from agent_email.prompts_semantics import (  # noqa: F401
    MEETING_CLUSTER_RELEVANCE_SYSTEM_PROMPT,
    MEETING_CONTEXT_SYSTEM_PROMPT,
    RELEVANCE_CONTEXT_SYSTEM_PROMPT,
    SPECIFIC_MEETING_RELEVANCE_SYSTEM_PROMPT,
    build_relevance_context_prompt,
    build_semantic_context_text,
    parse_relevance_context,
    render_relevance_context,
    topic_key_from_semantics,
)
 
# Re-exported cluster-synthesis prompt/parser.
from agent_email.prompts_cluster import (  # noqa: F401
    CLUSTER_SYNTHESIS_SYSTEM_PROMPT,
    build_cluster_synthesis_prompt,
    parse_cluster_synthesis_response,
)
 
logger = get_agent_logger("agent_email")
 
BU_EMAIL_SYSTEM_PROMPT = (
    "You are a sharp business analyst writing a briefing that a senior banking executive (CEO/MD "
    "level) will actually read before a meeting. You are given exactly ONE email (not a thread, "
    "not a digest) - subject, body, and any attachment text extracted from it. Your job is to "
    "surface the substance and business implication that is explicitly supported by the email or attachment., in the third person, using "
    "real names and roles that appear in the text (e.g. 'Finance team', 'Bharat Rungta'). "
    "Allow SIMPLE SOURCE-GROUNDED SYNTHESIS, including comparisons, changes over time, trends, and relationships between facts stated" "in the source. Preserve the relevant time period, status, and context when multiple values or states are present."
    "Do not infer unsupported implications, causality, urgency, judgment, recommendations, risks, or outcomes"
    "Never use 'I', 'we', 'you', or 'our'. Never state who the email was sent by, forwarded by, "
    "or addressed to - the executive already sees the sender and subject separately; your job is "
    "the content, not the routing. "
    "TREAT MEETING/CALENDAR INVITES AS CARRYING NO BRIEFING CONTENT. Many emails in this mailbox "
    "are Teams/Outlook/Webex meeting invites (subjects like 'FW: Review - X', bodies containing "
    "phrases such as 'Microsoft Teams meeting', 'Join on a video conferencing device', 'Join the "
    "meeting now', a Meeting ID, a Passcode, a dial-in phone number, a video ID, an organizer's "
    "email, or 'For organizers: Meeting options'). NEVER extract, quote, transcribe, or paraphrase "
    "any of that logistics text into summary/action_items/top_questions - not the meeting ID, not "
    "the passcode, not the dial-in number, not the organizer's contact details, not 'join on a "
    "video conferencing device'. If, after removing all of that boilerplate, an email has no real "
    "content left (it is purely a meeting invite/forward with no substantive body text of its "
    "own - e.g. a bare Teams/Outlook/Webex calendar invite with nothing beyond scheduling "
    "logistics), NO SUMMARY MUST BE PRODUCED AT ALL: return an empty array for summary as well as "
    "empty arrays for action_items and top_questions. Do not write a bullet stating that it is a "
    "meeting invite, do not write a placeholder sentence noting there is no content, and do not "
    "describe the invite in any way - a pure meeting invite gets summary: [], action_items: [], "
    "top_questions: [], with nothing else. Only fall back to a single descriptive summary bullet "
    "in the rarer case where the email is short/routine but is NOT a pure meeting invite (e.g. a "
    "bare forward or an FYI with no added context). Do not invent a topic, purpose, or discussion point for a meeting invite just "
    "because the subject line mentions a product or project name - the invite itself contains no "
    "information about what will actually be discussed unless the body explicitly says so. "
    "NEVER write meta-referential filler that describes the email as an object instead of "
    "stating its content. Banned openings/constructions (in any tense or phrasing) include: "
    "'This email summarizes...', 'This message stated...', 'The email is a follow-up...', 'The "
    "email states/mentions/indicates/discusses...', 'This is an email about...', 'The email "
    "contains no further info...', 'The message pertains to...', and any other sentence whose "
    "subject is 'the email'/'the message'/'this mail' rather than the actual business fact. A "
    "senior executive reading this wants the substance itself, not a description of the "
    "document. State the fact, decision, number, or ask directly, in third person, as if "
    "reporting it as news - e.g. write 'The Q4 collections target was revised to 92%' instead of "
    "'The email states that the Q4 target was revised to 92%'; write 'Finance is following up on "
    "the pending vendor invoice' instead of 'This email is a follow-up regarding the vendor "
    "invoice'. This applies equally to summary bullets, action items, and top questions - never "
    "frame an action item as 'The email asks X to...' (just write 'X must...'), and never frame a "
    "question as 'The email raises a question about...' (just ask the question). The only "
    "exception is the specific no-content case already described above (a pure meeting invite "
    "with nothing else in the body), where the single required sentence stating that fact is "
    "allowed to reference 'meeting invite/logistics email' directly. "
    "Ground every sentence in specific facts stated in the email or its attachment: names, "
    "figures, dates, targets, and decisions. Never invent, assume, or generalize content that is "
    "not explicitly present. Do not just restate the email line by line - identify the crux: what "
    "actually changed, what is being asked for and why it matters, what number or trend stands "
    "out, or what decision is pending. Every bullet must add distinct information; never repeat "
    "the same fact in different words across summary points, action items, or questions. If the "
    "email is short, routine, or purely informational with no real content to analyze (e.g. a "
    "bare forward with no added context, an automated notice, a calendar/meeting logistics note), "
    "say so plainly in one sentence and return empty arrays for action_items/top_questions rather "
    "than manufacturing filler. "
    "Decision Weight: 80% business outcomes, 20% technical details. Focus on business impact: "
    "decisions, timelines, accountabilities, financials, risk, and exceptions. "
    "TAT (Time To Action) context: 5-day reduction for success, 3-day reduction for partial success. "
    "Tone: Formal, clear, and concise. Length: 4-5 bullet points maximum, 200-250 words total. "
    "Return ONLY valid JSON, no markdown fences, no commentary. Each bullet point must be a "
    "single concise sentence: no sub-bullets, no paragraphs."
)
 
 
 
# Matches whole lines that are pure meeting-invite/dial-in boilerplate
# generated by Teams/Webex/Outlook ("Meeting ID: ...", "Passcode: ...",
# "Join on a video conferencing device", "For organizers: Meeting
# options", a dial-in phone number line, a video ID line, etc.). Applied
# BEFORE the email body ever reaches the LLM - relying on the model to
# reliably ignore this text in-context proved unreliable in practice
# (it repeatedly surfaced meeting IDs/passcodes into the summary), so
# it is stripped mechanically here as the primary guarantee, with the
# system prompt's own instruction to ignore any that slips through as
# a second line of defense.
_MEETING_BOILERPLATE_LINE_RE = re.compile(
    r"^\s*("
    r"microsoft teams meeting"
    r"|join (on|the) (a )?(video conferencing device|meeting now)"
    r"|meeting id\s*:?.*"
    r"|passcode\s*:?.*"
    r"|video id\s*:?.*"
    r"|dial[- ]?in.*"
    r"|phone conference id.*"
    r"|find a local number.*"
    r"|for organizers\s*:.*"
    r"|meeting options.*"
    r"|\+?\d[\d ()-]{7,}\d\s*,,.*"  # dial-in number lines often end in ",,<conf id>#"
    r"|_{5,}\s*$"  # the long underscore divider Outlook inserts around invite blocks
    r")\s*$",
    re.IGNORECASE,
)
 
 
def _strip_meeting_boilerplate(body_text: str) -> str:
    """
    Remove Teams/Webex/Outlook meeting-invite logistics lines (meeting
    ID, passcode, dial-in numbers, "join on a video conferencing
    device", organizer options, etc.) from an email body before it is
    handed to the LLM, so that boilerplate can never be quoted/
    paraphrased into a summary bullet, action item, or question -
    regardless of how the model interprets the system prompt's
    instruction to ignore it. Lines that don't match this specific
    boilerplate shape are left completely untouched.
    """
    kept_lines = [line for line in body_text.splitlines() if not _MEETING_BOILERPLATE_LINE_RE.match(line)]
    # Collapse runs of blank lines left behind by the removed boilerplate
    # block down to a single blank line, so the remaining text doesn't
    # end up with large gaps.
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines))
    return collapsed.strip()
 
 
def format_tables_for_prompt(facts_result: fact_extraction.FactExtractionResult) -> str:
    """
    Format extracted facts from tables for inclusion in LLM prompts.
   
    Args:
        facts_result: Result from fact extraction
   
    Returns:
        Formatted string suitable for LLM input
    """
    return fact_extraction.format_facts_for_llm(facts_result)
 
 
def build_bu_email_prompt(subject: str, body_text: str, attachment_texts: list[str]) -> str:
    """
    Build the single-LLM-call prompt for one individual email: given its
    subject, body, and any attachment text, ask the model to return summary
    + action_items + top_questions together in one JSON object. Used by
    ``agent_email.service.summarize_thread`` - exactly ONE call per email.
 
    ``body_text`` has Teams/Webex/Outlook meeting-invite boilerplate stripped
    via ``strip_meeting_boilerplate`` before it's included in the prompt.
    """
    body_text = _strip_meeting_boilerplate(body_text)
 
    # Extract facts from tables in body text
    tables_from_body = fact_extraction.extract_tables_from_text(body_text)
    facts_from_body = fact_extraction.FactExtractionResult(
        tables=tables_from_body,
        facts=[fact for table in tables_from_body for fact in table.facts],
        raw_text=body_text,
    )
    facts_block = fact_extraction.format_facts_for_llm(facts_from_body)
    attachments_block = ""
    if attachment_texts:
        attachment_blocks = []
        for i, text in enumerate(attachment_texts):
            tables_from_att = fact_extraction.extract_tables_from_text(text)
            facts_from_att = fact_extraction.FactExtractionResult(
                tables=tables_from_att,
                facts=[fact for table in tables_from_att for fact in table.facts],
                raw_text=text,
            )
            att_facts_block = fact_extraction.format_facts_for_llm(facts_from_att)
            if att_facts_block != "(no facts extracted from tables)":
                attachment_blocks.append(f"Attachment {i + 1} (extracted text with facts):\n{text}\n\n{att_facts_block}")
            else:
                attachment_blocks.append(f"Attachment {i + 1} (extracted text):\n{text}")
        attachments_block = "\n\n".join(attachment_blocks)
    else:
        attachments_block = "(no attachments with extractable text)"
    return (
        f"Below is ONE email only (subject: '{subject}'). Do not assume there are other emails "
        "in a thread - analyze only what is shown here. It may include text extracted from a "
        "PDF/Excel attachment with structured facts extracted from tables.\n\n"
        "The email may include EXTRACTED FACTS FROM TABLES in the following format:\n"
        "  ENTITY | VALUE | TIME/CONTEXT | STATUS | RELATION | SOURCE_TEXT\n"
        "These facts have been pre-processed to normalize statuses, dates, and relationships.\n\n"
        f"=== Email Body ===\n{body_text}\n"
        f"=== Extracted Facts from Email Body ===\n{facts_block}\n\n"
        f"=== Attachments ===\n{attachments_block}\n\n"
        "=== YOUR TASK ===\n"
        "Prepare an executive brief for THIS source only. The brief must help a bank CEO decide "
        "what warrants attention and what to ask the accountable executive. Synthesize rather than "
        "describe the document. Prioritize material facts, quantified movement, missed or upcoming "
        "commitments, exceptions, dependencies, risks, and decisions. If the source contains no "
        "substantive business information after boilerplate and signatures are ignored, return empty "
        "arrays for all three fields. Every output must be traceable to the source; do not add generic "
        "banking commentary or infer details from the subject alone.\n\n"
        "=== RESPONSE FORMAT ===\n"
        "Return ONLY valid JSON (no markdown or commentary) with exactly this structure:\n"
        "{\n"
        '  "summary": ["<material business fact and its executive implication>"],\n'
        '  "action_items": [\n'
        "    {\n"
        '      "text": "<named owner/team>: <specific deliverable>",\n'
        '      "due_date": "<date explicitly stated or clearly implied in the source, in YYYY-MM-DD format if a full '
        'date is given, or the exact weekday/relative phrase as written (e.g. \'Friday\', \'next week\') if only that '
        "is given, or null if no date is stated anywhere in the source>\",\n"
        '      "status": "<\'done\' if the source text itself reports this item as already completed/sent/'
        'closed/resolved, otherwise \'pending\'>"\n'
        "    }\n"
        "  ],\n"
        '  "top_questions": ["<CEO question anchored to a specific source fact or gap>"]\n'
        "}\n\n"
        "RULES:\n"
        "1. summary: 4-5 crisp senior-management-ready pointers (fewer if the email cannot "
        "support 4 with high confidence; never more than 5; never pad). Only include facts you "
        "are certain are explicitly in the source. If unsure, omit. Write to these rules:\n"
        "   (a) Preserve every fact, number, date, name, and decision exactly as stated in the "
        "source. Copy digits and units verbatim (e.g. if the source says Rs 5,972 Cr, write "
        "Rs 5,972 Cr). Never invent, round, annualise, assume, or replace a missing figure with "
        "a placeholder such as X, Y, Rs X Cr, or \u20b9X Cr.\n"
        "   (b) Lead with the single most important business outcome, metric, or change - put it "
        "in the first bullet.\n"
        "   (c) Follow a clear flow where the source supports it: current position -> "
        "performance/gap -> key driver or root cause -> action or next step.\n"
        "   (d) Prioritize quantified impact; make improvements and gaps immediately visible. If "
        "both a current figure and a prior/target figure are in the source, state the comparison "
        "using those exact figures. If only one side is stated, do not invent the other.\n"
        "   (e) Use direct, active, third-person language. Convert passive or awkward wording into "
        "direct statements. Never write meta-referential framing ('This email summarizes...', "
        "'The email states...', 'This message stated...', 'The email is a follow-up...', 'The "
        "email contains no further info...') - state the actual fact, decision, or number "
        "directly. Never use 'I', 'we', 'you', or 'our'.\n"
        "   (f) Remove repetition, filler, unnecessary context, and jargon.\n"
        "   (g) Keep the source's own units (Rs/Cr, days, bps, %). Do not convert or restyle "
        "figures in a way that changes the digits.\n"
        "   (h) Qualify root causes only to the degree the source evidence supports - use "
        "'potential contributor', 'key contributor', or 'identified driver' accordingly, and "
        "never overstate causation.\n"
        "   (i) Keep any stated action clear and accountable, but never imply an action, "
        "objective, or conclusion that is not stated in the source.\n"
        "   (j) Correct obvious spelling and grammar errors from the source.\n"
        "   (k) Do not pad. Do not restate the subject line as a bullet.\n"
        "   If the email is NOTHING BUT a Teams/Outlook/Webex meeting invite (pure scheduling "
        "logistics - join link, meeting ID, passcode, dial-in, organizer options - with no "
        "substantive body text of its own), summary MUST be an empty array. Do NOT write a "
        "bullet announcing that it is a meeting invite; do NOT describe it at all. The entire "
        "output for a pure meeting invite is summary: [], action_items: [], top_questions: []. "
        "Only for a short/routine email that is NOT a pure meeting invite (a bare forward, an "
        "FYI, a brief status note) is a single descriptive bullet acceptable - and even then, do "
        "not pad it out with invented detail or with logistics (meeting IDs, passcodes, dial-in "
        "info, organizer notes, video IDs). Never guess at a topic, agenda, or purpose for a "
        "meeting invite from the subject line alone; only state what a meeting is about if the "
        "BODY text explicitly says so, and only when there is other real content to report "
        "alongside the invite.\n"
        "2. Never state who sent, forwarded, or received the email in the summary. Focus purely "
        "on the substance.\n"
        "3. action_items: list ALL concrete tasks explicitly stated or clearly implied in this "
        "email's own text (a request, an instruction, a 'please do X', a deliverable someone was "
        "asked to produce, data/analysis someone must send by a date), whoever the named owner is "
        "- the owner does not have to be the recipient of the briefing. A meeting invite/logistics "
        "email (nothing but scheduling a meeting, no requested deliverable in the body) has NO "
        "action item - 'join the meeting', 'attend the meeting', 'prepare for the meeting', or "
        "'be available on [date]' are NOT action items and must never appear here, even if the "
        "email nominally asks the recipient to attend. Only a specific, nameable piece of work "
        "(a document, a number, a decision, an analysis, a follow-up email) counts. Return an "
        "empty array if this email contains no such concrete deliverable - do not invent one. "
        "EXTRACT ALL DISTINCT CONCRETE TASKS - there is no upper limit and no target count, but "
        "extract everything you find. Each item's 'text' field must "
        "name the real owner and real action from the email text as '<owner>: <task>'; never output "
        "the literal words 'Owner', 'Action', or 'Deadline', and never use square-bracket "
        "placeholders or the example text verbatim. If no owner is named, write 'Unassigned'. Each "
        "item's 'due_date' field must be the date/deadline that specific task actually has in the "
        "source text (see the RESPONSE FORMAT above for the exact format), or null if the source "
        "states no date for it - never invent a date, and never put date text inside 'text' itself. "
        "Each item's 'status' field must be 'done' ONLY if the source text itself says this "
        "specific deliverable has already been completed/sent/closed/resolved (e.g. 'the report "
        "has been sent', 'this is now closed', 'already completed'); otherwise it must be "
        "'pending'. Never infer 'done' from a due date alone (a past due date does not mean the "
        "work was completed) - only from the source explicitly reporting the work as finished.\n"
        "Do not list the same action item more than once even if it is mentioned multiple times in "
        "the email. Never phrase an item as 'The email asks [owner] to...' - state the task "
        "directly as '<owner>: <task>'. If you are not certain a task, owner, or deadline is "
        "explicitly in the source, omit that item rather than guessing.\n"
        "4. top_questions: aim for at least 4 logical, relevant questions a CEO would actually put to the "
        "accountable executive in a pre-meet (minimum 4 whenever the email contains substantive facts; "
        "fewer only if the email truly cannot support 4 with high confidence; never more than 6; never pad). "
        "Prefer QUANT-ORIENTED questions whenever the source states a figure: name the metric, the exact source "
        "number or gap, and the decision it forces. Never copy sample questions from these instructions. Each must "
        "probe a specific number, gap, tradeoff, or decision point that is actually written "
        "in the text above, and each must be distinct. If you are not certain the question is "
        "grounded in this email, omit it. NEVER ask about the email/meeting itself "
        "as an object - questions like 'What is the purpose of this meeting?', 'What is the"
        "specific topic being discussed?', 'What is the expected outcome of the meeting?', 'What "
        "are the risks?', 'What is the timeline?', 'What is the current X position?', 'How do we "
        "manage X?', or 'What are the implications of X?' are banned regardless of phrasing. If "
        "the email has no real substantive fact, figure, or decision to question, return an empty "
        "array rather than forcing a generic one - a plain meeting invite ALWAYS gets an empty "
        "top_questions array. Never phrase a question as 'The email raises a question about...'.\n"
        "5. Return ONLY the JSON object, nothing else."
    )
 
 
def parse_bu_email_response(response_text: str) -> dict:
    """
    Parse the LLM's single-email JSON response (summary/action_items/
    top_questions) into a sanitized dict. Raises ValueError on parse
    failure so callers can retry.
    """
    text = response_text.strip()
 
    json_match = re.search(r'```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    else:
        json_obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_obj_match:
            text = json_obj_match.group(0).strip()
 
    text = _sanitize_llm_json_text(text.strip())
 
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        # Small models (e.g. llama3.2:3b) frequently emit *almost* valid
        # JSON with a structural slip - a missing comma between two array
        # elements on separate lines, a trailing comma, or an unescaped
        # double-quote inside a string value (which prematurely ends the
        # string so the next token looks like it needs a comma: the
        # "Expecting ',' delimiter" error). Try a conservative repair
        # pass before giving up, so a single stray quote/comma doesn't
        # cost us the whole summary.
        repaired = _repair_llm_json_text(text)
        try:
            data = json.loads(repaired)
            logger.info("parse_bu_email_response: recovered a malformed LLM JSON response via repair pass")
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {first_error}\nResponse: {text[:200]}"
            ) from first_error
 
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
 
    summary = data.get("summary", [])
    action_items_raw = data.get("action_items", [])
    top_questions = data.get("top_questions", [])
    if not isinstance(summary, list) or not isinstance(action_items_raw, list) or not isinstance(top_questions, list):
        raise ValueError("Expected 'summary', 'action_items', and 'top_questions' to each be a list")
 
    summary = _quality_filter_summary(_sanitize_and_drop_empty(summary))
    action_items = _quality_filter_actions(_sanitize_and_normalize_action_items(action_items_raw))
    top_questions = _quality_filter_questions(_sanitize_and_drop_empty(top_questions))
 
    return {
        "summary_text": "\n".join(f"\u2022 {item}" for item in summary),
        # Kept as "<text> - <due_date or TBD>" strings for backward compatibility with
        # every existing consumer (CSV column, frontend date-extraction regexes, action
        # tracker's due-date parsing) - see `_sanitize_and_normalize_action_items`.
        "action_items": action_items,
        "top_questions": top_questions,
    }
 
 
def evaluate_bu_response_quality(response_text: str, source_text: str = "") -> dict:
    """
    Inspect raw model JSON without changing it. Structural and heuristic
    checks for test review; factual grounding still requires a human reading
    the source alongside the output.
    """
    checks: dict[str, bool] = {}
    warnings: list[str] = []
 
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
 
    try:
        data = json.loads(_sanitize_llm_json_text(text.strip()))
    except (json.JSONDecodeError, TypeError):
        return {
            "passed": False,
            "checks": {"valid_json": False},
            "warnings": ["The model response is not valid JSON."],
            "grounding_requires_human_review": True,
        }
 
    checks["valid_json"] = isinstance(data, dict)
    if not isinstance(data, dict):
        return {
            "passed": False,
            "checks": checks,
            "warnings": ["The model response must be a JSON object."],
            "grounding_requires_human_review": True,
        }
 
    fields = {name: data.get(name) for name in ("summary", "action_items", "top_questions")}
    checks["required_fields_are_lists"] = all(isinstance(value, list) for value in fields.values())
    if not checks["required_fields_are_lists"]:
        warnings.append("summary, action_items, and top_questions must all be arrays.")
        return {
            "passed": False,
            "checks": checks,
            "warnings": warnings,
            "grounding_requires_human_review": True,
        }
 
    summary = _sanitize_and_drop_empty(fields["summary"])
    actions = _sanitize_and_normalize_action_items(fields["action_items"])
    questions = _sanitize_and_drop_empty(fields["top_questions"])
 
    checks["summary_count_is_0_to_5"] = len(summary) <= 5
    checks["question_count_is_0_to_4"] = len(questions) <= 4
    checks["summary_has_no_placeholder_figures"] = all(not _PLACEHOLDER_FIGURE_RE.search(s) for s in summary)
    checks["questions_have_no_placeholder_figures"] = all(not _PLACEHOLDER_FIGURE_RE.search(q) for q in questions)
    checks["questions_end_with_question_mark"] = all(q.rstrip().endswith("?") for q in questions)
    checks["summary_has_no_document_meta"] = all(not _NON_EXECUTIVE_SUMMARY_RE.search(s) for s in summary)
    checks["actions_have_no_meeting_logistics"] = all(not _NON_ACTION_RE.search(a) for a in actions)
    checks["questions_are_not_generic_or_document_meta"] = all(
        not _NON_EXECUTIVE_QUESTION_RE.search(q) for q in questions
    )
    checks["summary_has_no_duplicates"] = len(_dedupe_preserving_order(summary)) == len(summary)
    checks["actions_have_no_duplicates"] = len(_dedupe_preserving_order(actions)) == len(actions)
    checks["questions_have_no_duplicates"] = len(_dedupe_preserving_order(questions)) == len(questions)
 
    source_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9%.-]{3,}|\d+(?:\.\d+)?%?", source_text)
        if token.casefold() not in {
            "this", "that", "with", "from", "email", "message", "meeting", "what", "when",
            "where", "which", "would", "could", "should", "have", "been", "will", "bank",
        }
    }
    checks["questions_reference_source_terms"] = all(
        bool(source_terms.intersection({token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9%.-]{3,}|\d+(?:\.\d+)?%?", q)}))
        for q in questions
    )
 
    warning_by_check = {
        "summary_count_is_0_to_5": "The summary has more than five pointers.",
        "question_count_is_0_to_4": "The response contains more than four questions.",
        "questions_end_with_question_mark": "One or more questions are not phrased as questions.",
        "summary_has_no_document_meta": "A summary describes the email, signature, routing, image, or contact details.",
        "summary_has_no_placeholder_figures": "A summary contains placeholder figures (X/Y) rather than source numbers.",
        "questions_have_no_placeholder_figures": "A question contains placeholder figures (X/Y) rather than source numbers.",
        "actions_have_no_meeting_logistics": "An action item is only meeting attendance or access logistics.",
        "questions_are_not_generic_or_document_meta": "A question is generic or asks about the email/document itself.",
        "summary_has_no_duplicates": "The summary contains duplicate outcomes.",
        "actions_have_no_duplicates": "The action-item list contains duplicates.",
        "questions_have_no_duplicates": "The question list contains duplicates.",
        "questions_reference_source_terms": "A question has no recognizable anchor in the supplied email or attachment text.",
    }
    warnings.extend(message for check, message in warning_by_check.items() if not checks.get(check, True))
 
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "warnings": warnings,
        "grounding_requires_human_review": True,
    }
 
 
