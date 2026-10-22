"""Unit tests for cluster synthesis prompt constraints."""

from types import SimpleNamespace

from agent_email.prompts import CLUSTER_SYNTHESIS_SYSTEM_PROMPT, build_cluster_synthesis_prompt


def test_synthesis_prompt_locks_assigned_department_and_forbids_invention():
    depts = [
        SimpleNamespace(name="BB&RL", summary="Retail branch CASA only."),
        SimpleNamespace(
            name="Wholesale Banking",
            summary="Wholesale deposits, NIM pack, corporate credit TAT.",
        ),
    ]
    prompt = build_cluster_synthesis_prompt(
        appended_emails="--- Email 1 of 2 ---\nSubject: July MBR\nBody:\nPipeline Rs 3900 Cr\n",
        departments=depts,
        assigned_department="Wholesale Banking",
        email_count=2,
        phrases_by_dept={"Wholesale Banking": ["NIM pack funding cost assumption", "corporate credit TAT"]},
    )
    assert "already assigned: Wholesale Banking" in prompt
    assert "Do not pick a different department" in prompt
    assert "Never invent" in prompt or "never invent" in prompt.lower()
    assert "full body + full MoM" in prompt
    assert "NIM pack funding cost assumption" in prompt
    assert "must use ALL of them" in prompt
    assert "Do not let the longest body or MoM attachment dominate" in prompt
    assert "Join/attend/prepare for the meeting" in prompt
    assert "Rs 3900 Cr" in prompt
    assert "FIGURES PRESENT IN THIS CLUSTER" in prompt
    assert "3-5" in prompt
    assert ("3-4" in prompt or "at least 4" in prompt or "minimum 4" in prompt)
    assert "one source email" in prompt.lower()
    assert "From, To, or Cc" in prompt
    assert "Can the CEO" in prompt
    assert "Why is standard TAT 7.1" not in prompt
    assert "6.4 days" not in prompt
    assert "₹X Cr from ₹Y Cr" not in prompt
    assert "No maximum" in prompt


def test_synthesis_system_prompt_is_quantitative():
    text = CLUSTER_SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "never invent" in text
    assert "3-5" in text
    assert "at least 4" in text or "minimum 4" in text
    assert "one source email" in text or "only one" in text
    assert "from/to/cc" in text
    assert "can the ceo" in text


def test_parse_cluster_synthesis_keeps_insight_and_all_actions():
    from agent_email.prompts import parse_cluster_synthesis_response

    long_bullet = (
        "Wholesale CA is Rs 5,972 Cr against the Sep-26 target of Rs 9,625 Cr "
        "(gap Rs 3,653 Cr), so the quarter-end rebuild is the meeting decision."
    )
    raw = (
        '{"department":"Wholesale Banking","topic":"WB Review",'
        f'"summary":["{long_bullet}","Pipeline held at Rs 3750 Cr.",'
        '"Wholesale CA gap is probably about Rs 4000 Cr.",'
        '"Q3 funding cost assumption for the NIM pack: ₹X Cr from ₹Y Cr",'
        '"Wholesale Liabilities & Deposit Mix Review- Discussion","Extra sixth pointer on CASA mix."],'
        '"top_questions":['
        '"Why has Product CA only reached Rs 5,972 Cr against Rs 9,625 Cr?",'
        '"What is the purpose of this meeting?",'
        '"How do we manage exposure?",'
        '"Why is TAT 7.1 days vs 6?",'
        '"Which names close the Rs 3,653 Cr CA gap by the 30th?"'
        "],"
        '"action_items":['
        '{"owner":"Credit","task":"Sign off exception logic","deadline":"19-Jul-2026"},'
        '{"owner":"Kavya Iyer","task":"Finalise ORION pilot scope","deadline":"30 Jun"},'
        '{"owner":"Amit Sethi","task":"Prepare exception-case root-cause breakdown","deadline":"30-Aug-2026"},'
        '{"owner":"Neeraj Bhatia","task":"Review documentation checklist","deadline":"21-Sep-2026"},'
        '{"owner":"Bharat Rungta","task":"Name the accounts that fund the CA gap","deadline":"30th"},'
        '{"owner":"TBA","task":"Attend the meeting","deadline":null}'
        "]}"
    )
    source = (
        "Subject: Wholesale Liabilities & Deposit Mix Review- Discussion\n"
        "Subject: Q3 funding cost assumption for the NIM pack\n"
        "Body: Product CA Rs 5,972 Cr against Sep-26 target Rs 9,625 Cr. Gap Rs 3,653 Cr by the 30th.\n"
        "Pipeline held at Rs 3750 Cr.\n"
    )
    parsed = parse_cluster_synthesis_response(raw, source_text=source)
    assert parsed["summary"][0] == long_bullet
    assert "Pipeline held at Rs 3750 Cr." in parsed["summary"]
    assert len(parsed["summary"]) <= 5
    assert not any("probably" in s.lower() for s in parsed["summary"])
    assert not any("₹X" in s or "Rs X" in s for s in parsed["summary"])
    assert not any("Deposit Mix Review" in s for s in parsed["summary"])
    joined_q = " ".join(parsed["top_questions"]).lower()
    assert "purpose of this meeting" not in joined_q
    assert "how do we manage" not in joined_q
    assert "7.1" not in joined_q
    assert any("5,972" in q or "5972" in q.replace(",", "") for q in parsed["top_questions"])
    assert 1 <= len(parsed["top_questions"]) <= 4
    tasks = [item["task"] for item in parsed["action_items"] if isinstance(item, dict)]
    assert "Attend the meeting" not in tasks
    assert len(tasks) == 5
    assert "Sign off exception logic" in tasks
    assert "Finalise ORION pilot scope" in tasks


def test_parse_cluster_synthesis_drops_process_meta_summary():
    from agent_email.prompts import parse_cluster_synthesis_response

    raw = (
        '{"department":"Wholesale Banking","topic":"Corporate Credit TAT",'
        '"summary":['
        '"The explanation given is that the credit team is working to improve the process, but the timeline is not yet met.",'
        '"There is a challenge to the explanation, as the TAT has not improved in the past quarter.",'
        '"Standard-case TAT is 6.4 days versus the 9.2-day April baseline, while exception cases remain at 13.1 days.",'
        '"Credit has not given a dated plan to bring the client-facing average down."'
        "],"
        '"top_questions":['
        '"Can the CEO confirm the target for Corporate Credit TAT and ensure it is aligned for industry benchmarks?",'
        '"Why is exception-case TAT still 13.1 days when standard-case TAT is 6.4 days?",'
        '"When will Credit commit a date to close the gap from 9.2 days?",'
        '"Which exception types still drive the 13.1-day average?"'
        "],"
        '"action_items":[]}'
    )
    source = (
        "Subject: RE: Corporate Credit TAT\n"
        "Body: Standard TAT 6.4 days from 9.2 days in April. Exception cases 13.1 days.\n"
    )
    parsed = parse_cluster_synthesis_response(raw, source_text=source)
    joined = " ".join(parsed["summary"]).lower()
    assert "explanation given is" not in joined
    assert "challenge to the explanation" not in joined
    assert any("6.4" in s for s in parsed["summary"])
    assert not any("ceo confirm" in q.lower() for q in parsed["top_questions"])
    assert len(parsed["top_questions"]) == 3
