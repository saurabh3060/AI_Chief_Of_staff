"""
tests/chatbot/test_build_context.py
======================================
Unit tests for the lossless cleaning stage of chatbot.retriever.build_context:
headers, separators, signatures, address blocks and quoted repeats are removed,
and nothing else is rewritten.

The sample threads below deliberately contain:
  * repeated quoted reply history (the same paragraph across all 3 threads)
  * header lines (From/Sent/To/Subject/Date), separators, signatures and
    an address block, which must all be stripped
so the test can assert the output is compact, header-free and dedup-free and
still includes content from every thread.
"""

import pytest

from chatbot import retriever
from chatbot.retriever import ContextConfig


def make_cfg(**overrides) -> ContextConfig:
    """A ContextConfig with every value explicit (no Settings, no .env)."""
    values = dict(
        max_chars=4000, top_threads=10, neighbor_blocks=1, min_thread_share=0.2,
        figure_bonus=0.1, use_embeddings=False, embedding_weight=0.5,
        dedup_similarity=0.9, dedup_min_chars=60, short_line_chars=60,
        table_min_lines=2, table_min_numbers=3, block_max_chars=800,
        partial_head_share=0.5, thin_body_chars=300, attachment_max_chars=1500,
    )
    values.update(overrides)
    return ContextConfig(**values)


def build_context(threads, question, budget_chars=None, **cfg_overrides):
    return retriever.build_context(threads, question, cfg=make_cfg(**cfg_overrides), budget_chars=budget_chars)


# A block of quoted reply history that is pasted verbatim into every thread.
# build_context must keep it only ONCE across all three threads.
_QUOTED_HISTORY = (
    "-----Original Message-----\n"
    "From: deepak.kumar@example.com\n"
    "Sent: Monday, April 14, 2025 9:00 AM\n"
    "To: ceo.office@example.com\n"
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
            f"From: analyst@example.com\n"
            f"Sent: Tuesday, April 15, 2025 10:30 AM\n"
            f"To: ceo.office@example.com\n"
            f"Subject: {subject}\n"
            f"========================================\n"
            f"{unique_body}\n"
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
    out = build_context(sample_threads, "April figures")
    lowered = out.lower()

    for line in out.splitlines():
        assert not line.lower().startswith(("from:", "sent:", "to:", "cc:", "bcc:", "subject:", "date:"))

    assert "-----original message-----" not in lowered
    assert "====" not in out
    assert "regards" not in lowered
    assert "analytics team" not in lowered   # the signature that follows the sign-off
    assert "tel:" not in lowered


def test_text_is_not_rewritten(sample_threads):
    """Cleaning only removes: the kept sentence is byte-for-byte what was written."""
    out = build_context(sample_threads, "April figures")
    assert "Wholesale sanctions for April reached nine hundred and twelve crore in total." in out


def test_never_uses_forbidden_fields(sample_threads):
    """build_context must use subject + body only, never raw/json/metadata/ids."""
    out = build_context(sample_threads, "April figures")
    for leaked in ("LEAKED_RAW_CONTENT", "LEAKED_JSON", "LEAKED_METADATA", "LEAKED_ATTACHMENT",
                   "LEAKED_MESSAGE_ID", "LEAKED_CONVERSATION_ID"):
        assert leaked not in out


def test_quoted_history_deduplicated(sample_threads):
    """The shared quoted-history paragraph must appear at most once overall."""
    out = build_context(sample_threads, "wholesale CASA")
    assert out.lower().count("wholesale casa book grew by eighteen percent") <= 1


def test_every_thread_included(sample_threads):
    """Each matching thread keeps its own header block and unique sentence."""
    out = build_context(sample_threads, "April figures")
    assert "[Email 1: Wholesale Banking Review]" in out
    assert "[Email 2: Retail Liabilities Update]" in out
    assert "[Email 3: Credit Cards Performance]" in out
    assert "nine hundred and twelve crore" in out
    assert "twelve percent in april" in out.lower()
    assert "two thousand crore" in out


def test_smaller_than_the_raw_threads(sample_threads):
    before = sum(len(t["body"]) for t in sample_threads)
    out = build_context(sample_threads, "What were the April figures across business units?")
    assert len(out) < before
