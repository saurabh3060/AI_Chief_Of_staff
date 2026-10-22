"""
tests/chatbot/test_build_context.py
======================================
Unit tests for chatbot.retriever.build_context - the prompt-size
reduction that keeps ALL retrieved threads but sends only deduplicated,
question-relevant sentences from each thread's subject + body.

The sample threads below deliberately contain:
  * repeated quoted reply history (the same sentences across all 3 threads)
  * header lines (From/Sent/To/Subject/Date), separators, signatures and
    an address block, which must all be stripped
  * a "Sample N" prefix that must be removed
so the test can assert the output is compact, header-free, dedup-free and
still includes content from every thread.
"""

import pytest

from chatbot.retriever import build_context


# A block of quoted reply history that is pasted verbatim into every thread.
# build_context must keep it only ONCE across all three threads.
_QUOTED_HISTORY = (
    "-----Original Message-----\n"
    "From: deepak.kumar@rblbank.com\n"
    "Sent: Monday, April 14, 2025 9:00 AM\n"
    "To: ceo.office@rblbank.com\n"
    "Subject: RE: Wholesale Banking Review\n"
    "\n"
    "The wholesale CASA book grew by eighteen percent during the first quarter.\n"
    "Corporate credit turnaround time improved to four days on average last month.\n"
)


def _thread(subject: str, unique_body: str) -> dict:
    """Build a retrieval-shaped thread chunk with headers + shared history."""
    return {
        "subject": subject,
        "body": (
            f"From: analyst@rblbank.com\n"
            f"Sent: Tuesday, April 15, 2025 10:30 AM\n"
            f"To: ceo.office@rblbank.com\n"
            f"Subject: {subject}\n"
            f"========================================\n"
            f"Sample 1: {unique_body}\n"
            f"Regards,\n"
            f"The Analytics Team\n"
            f"Tel: +91 22 1234 5678\n"
            f"{_QUOTED_HISTORY}"
        ),
        # Fields build_context must NEVER use - poisoned so any leak is obvious.
        "thread_raw_content": "LEAKED_RAW_CONTENT should never appear",
        "messages_json": '[{"body": "LEAKED_JSON should never appear"}]',
        "metadata_summary": "LEAKED_METADATA should never appear",
        "attachment_data": "LEAKED_ATTACHMENT should never appear",
        "message_id": "LEAKED_MESSAGE_ID",
        "conversation_id": "LEAKED_CONVERSATION_ID",
    }


@pytest.fixture
def sample_threads() -> list[dict]:
    return [
        _thread(
            "Wholesale Banking Review",
            "Wholesale sanctions for April reached nine hundred and twelve crore in total.",
        ),
        _thread(
            "Retail Liabilities Update",
            "Retail CASA deposits mobilised through branches rose by twelve percent in April.",
        ),
        _thread(
            "Credit Cards Performance",
            "Credit card spends crossed two thousand crore for the month of April this year.",
        ),
    ]


def test_output_under_budget(sample_threads):
    """The built context must not exceed the configured char budget."""
    budget = 2000
    out = build_context(sample_threads, "April figures", budget_chars=budget)
    assert len(out) <= budget


def test_no_header_or_noise_lines(sample_threads):
    """Header/separator/signature/address lines must be stripped."""
    out = build_context(sample_threads, "April figures", budget_chars=4000)
    lowered = out.lower()

    # Header labels at line starts
    for line in out.splitlines():
        assert not line.lower().startswith(("from:", "sent:", "to:", "cc:", "bcc:", "subject:", "date:"))

    assert "-----original message-----" not in lowered
    assert "====" not in out
    assert "regards" not in lowered
    assert "tel:" not in lowered
    # The "Sample N" prefix must be removed from kept sentences.
    assert "sample 1:" not in lowered


def test_never_uses_forbidden_fields(sample_threads):
    """build_context must use subject + body only, never raw/json/metadata/ids."""
    out = build_context(sample_threads, "April figures", budget_chars=4000)
    assert "LEAKED_RAW_CONTENT" not in out
    assert "LEAKED_JSON" not in out
    assert "LEAKED_METADATA" not in out
    assert "LEAKED_ATTACHMENT" not in out
    assert "LEAKED_MESSAGE_ID" not in out
    assert "LEAKED_CONVERSATION_ID" not in out


def test_quoted_history_deduplicated(sample_threads):
    """The shared quoted-history sentence must appear at most once overall."""
    out = build_context(sample_threads, "wholesale CASA", budget_chars=4000)
    sentinel = "wholesale casa book grew by eighteen percent"
    assert out.lower().count(sentinel) <= 1


def test_every_thread_included(sample_threads):
    """No thread may be dropped - each subject header block must be present."""
    out = build_context(sample_threads, "April figures", budget_chars=4000)
    assert "[Email 1: Wholesale Banking Review]" in out
    assert "[Email 2: Retail Liabilities Update]" in out
    assert "[Email 3: Credit Cards Performance]" in out

    # And the unique sentence from each thread survives.
    assert "nine hundred and twelve crore" in out
    assert "twelve percent in april" in out.lower()
    assert "two thousand crore" in out


def test_before_after_sizes_printed(sample_threads, capsys):
    """Print before/after char counts + estimated tokens (chars/4) for a query."""
    question = "What were the April figures across business units?"

    # "Before": the raw thread text we would otherwise have sent (bodies joined).
    before_chars = sum(len(t["body"]) for t in sample_threads)

    out = build_context(sample_threads, question, budget_chars=8000)
    after_chars = len(out)

    with capsys.disabled():
        print("\n--- build_context before/after ---")
        print(f"query:         {question}")
        print(f"before: {before_chars:>6} chars  (~{before_chars // 4} tokens)")
        print(f"after:  {after_chars:>6} chars  (~{after_chars // 4} tokens)")
        print(f"reduction: {100 * (before_chars - after_chars) / before_chars:.1f}%")

    assert after_chars < before_chars
