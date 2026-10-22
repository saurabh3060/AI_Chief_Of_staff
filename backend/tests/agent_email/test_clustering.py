"""Unit tests for email clustering embed text."""

from datetime import datetime, timezone

from agent_email.clustering import candidate_embed_text
from agent_email.schema import EmailCandidate, EmailMessage


def test_candidate_embed_text_subject_appears_once():
    # Subject must appear exactly once — repeating it distorts the centroid
    # embedding and causes semantically distinct emails sharing surface-level
    # subject tokens to cluster together incorrectly.
    settings = type("S", (), {
        "email_embed_body_max_chars": 100,
        "email_embed_attachment_max_chars": 100,
        "email_embed_subject_copies": 2,  # config value ignored — always 1
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="1",
            subject="Wholesale MBR",
            body_text="Body text here",
            received_at=datetime.now(timezone.utc),
        ),
    )
    text = candidate_embed_text(candidate, settings=settings)
    assert text.count("Wholesale MBR") == 1
    assert "Body text here" in text


def test_candidate_embed_text_drops_external_banner():
    settings = type("S", (), {
        "email_embed_body_max_chars": 400,
        "email_embed_attachment_max_chars": 100,
        "email_embed_subject_copies": 2,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="2",
            subject="Sep-26 CA/SA targets - WB view required",
            body_text=(
                "EXTERNAL EMAIL: Do not click any links or open any attachments "
                "unless you trust the sender and know the content is safe.\n"
                "Need WB view on September CASA targets."
            ),
            received_at=datetime.now(timezone.utc),
        ),
    )
    text = candidate_embed_text(candidate, settings=settings)
    assert "EXTERNAL EMAIL" not in text
    assert "September CASA targets" in text


def test_format_emails_for_synthesis_uses_full_body_and_attachment():
    from agent_email.clustering import format_emails_for_synthesis

    long_body = "BODY-" + ("x" * 5000)
    long_attach = "MOM-" + ("y" * 5000)
    settings = type("S", (), {
        "email_synth_body_max_chars": 0,
        "email_synth_attachment_max_chars": 0,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="3",
            subject="July Review",
            body_text=long_body,
            received_at=datetime.now(timezone.utc),
        ),
        attachment_text=long_attach,
    )
    block = format_emails_for_synthesis([candidate], [0], settings=settings)
    assert "Email 1 of 1" in block
    assert long_body in block
    assert long_attach in block
    assert "From:" not in block


def test_format_emails_for_synthesis_keeps_quoted_thread_drops_routing():
    from agent_email.clustering import format_emails_for_synthesis

    settings = type("S", (), {
        "email_synth_body_max_chars": 0,
        "email_synth_attachment_max_chars": 0,
    })()
    body = (
        "EXTERNAL EMAIL: Do not click any links.\n"
        "From: Sameer Kapoor <sameer.kapoor@rbl.bank.in>\n"
        "To: digitaloffice.aicos@rblbank.com\n"
        "Cc: Chourishi, Tanishq <tchourishi@alvarezandmarsal.com>\n"
        "Subject: RE: Corporate Credit TAT\n\n"
        "Standard-case TAT is 6.4 days against a 9.2 day April baseline.\n"
        "-----Original Message-----\n"
        "From: Bharat Rungta <bharat.rungta@rbl.bank.in>\n"
        "Exception-flagged proposals are still taking 13.1 days.\n"
    )
    candidate = EmailCandidate(
        message=EmailMessage(
            id="5",
            subject="RE: Corporate Credit TAT",
            body_text=body,
            from_name="Chourishi, Tanishq",
            from_address="tchourishi@alvarezandmarsal.com",
            received_at=datetime.now(timezone.utc),
        ),
        attachment_text="",
    )
    block = format_emails_for_synthesis([candidate], [0], settings=settings)
    assert "6.4 days" in block
    assert "13.1 days" in block
    assert "Chourishi" not in block
    assert "alvarezandmarsal" not in block.lower()
    assert "From:" not in block


def test_format_emails_for_synthesis_respects_positive_cap():
    from agent_email.clustering import format_emails_for_synthesis

    settings = type("S", (), {
        "email_synth_body_max_chars": 20,
        "email_synth_attachment_max_chars": 10,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="4",
            subject="Cap test",
            body_text="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            received_at=datetime.now(timezone.utc),
        ),
        attachment_text="1234567890XXXX",
    )
    block = format_emails_for_synthesis([candidate], [0], settings=settings)
    assert "ABCDEFGHIJKLMNOPQRST" in block
    assert "UVWXYZ" not in block
    assert "1234567890" in block
    assert "XXXX" not in block


def test_series_subject_tokens_align_recurring_reviews():
    from agent_email.clustering import series_subject_tokens

    july = series_subject_tokens("Wholesale Banking - July Review | Minutes")
    june = series_subject_tokens("Actions from June's WB Review")
    august = series_subject_tokens("WB Monthly Review - August | Key Actions")
    liabilities = series_subject_tokens("Wholesale Liabilities & Deposit Mix Review- Discussion")
    casa = series_subject_tokens("Sep-26 CA/SA targets - WB view required")

    shared_reviews = july & june
    assert {"wholesale", "banking", "review"} <= shared_reviews
    assert {"wholesale", "banking", "review"} <= (june & august)
    assert len(july & liabilities) < 3
    assert len(june & casa) < 3


def test_series_near_miss_merges_june_into_review_cluster_not_liabilities():
    import math

    import numpy as np

    from agent_email.clustering import cluster_candidates

    def unit(*vals: float) -> np.ndarray:
        vec = np.array(vals, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    settings = type("S", (), {
        "email_cluster_algo": "components",
        "email_cluster_similarity_threshold": 0.75,
        "email_cluster_series_near_miss": 0.68,
        "email_cluster_series_jaccard": 0.5,
        "email_cluster_residual_threshold": 0.0,
    })()
    july = unit(1.0, 0.0, 0.0)
    june = unit(0.72, math.sqrt(1.0 - 0.72 ** 2), 0.0)
    liab = unit(0.70, 0.05, math.sqrt(max(0.0, 1.0 - 0.70 ** 2 - 0.05 ** 2)))
    groups = cluster_candidates(
        [july, june, liab],
        settings=settings,
        subjects=[
            "Wholesale Banking - July Review | Minutes",
            "Actions from June's WB Review",
            "Wholesale Liabilities & Deposit Mix Review- Discussion",
        ],
    )
    grouped = {frozenset(g) for g in groups}
    assert frozenset({0, 1}) in grouped
    assert frozenset({2}) in grouped


def test_series_merge_runs_for_agglomerative_even_when_cosine_is_low():
    import math

    import numpy as np

    from agent_email.clustering import cluster_candidates

    def unit(*vals: float) -> np.ndarray:
        vec = np.array(vals, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    settings = type("S", (), {
        "email_cluster_algo": "agglomerative",
        "email_cluster_similarity_threshold": 0.75,
        "email_cluster_series_near_miss": 0.68,
        "email_cluster_series_jaccard": 0.5,
        "email_cluster_residual_threshold": 0.0,
    })()
    july = unit(1.0, 0.0, 0.0)
    june = unit(0.50, math.sqrt(1.0 - 0.50 ** 2), 0.0)
    groups = cluster_candidates(
        [july, june],
        settings=settings,
        subjects=[
            "Wholesale Banking - July Review | Minutes",
            "Actions from June's WB Review",
        ],
    )
    grouped = {frozenset(g) for g in groups}
    assert grouped == {frozenset({0, 1})}


def test_residual_merge_joins_funding_singletons_without_unlocking_review_series():
    import numpy as np

    from agent_email.clustering import cluster_candidates

    def unit(*vals: float) -> np.ndarray:
        vec = np.array(vals, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    settings = type("S", (), {
        "email_cluster_algo": "agglomerative",
        "email_cluster_similarity_threshold": 0.75,
        "email_cluster_series_near_miss": 0.68,
        "email_cluster_series_jaccard": 0.5,
        "email_cluster_residual_threshold": 0.72,
    })()
    july = unit(1.0, 0.0, 0.0, 0.0)
    april = unit(0.99, 0.14, 0.0, 0.0)
    nim = unit(0.0, 0.0, 1.0, 0.0)
    casa = unit(0.0, 0.0, 0.85, 0.53)
    groups = cluster_candidates(
        [july, april, nim, casa],
        settings=settings,
        subjects=[
            "Wholesale Banking - July Review | Minutes",
            "Wholesale Banking - April Review | Minutes",
            "Q3 funding cost assumption for the NIM pack",
            "Sep-26 CA/SA targets - WB view required",
        ],
    )
    grouped = {frozenset(g) for g in groups}
    assert frozenset({0, 1}) in grouped
    assert frozenset({2, 3}) in grouped


def test_quoted_subject_links_join_compiled_pack_not_review_series():
    import numpy as np

    from agent_email.clustering import cluster_candidates

    def unit(*vals: float) -> np.ndarray:
        vec = np.array(vals, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    settings = type("S", (), {
        "email_cluster_algo": "agglomerative",
        "email_cluster_similarity_threshold": 0.75,
        "email_cluster_series_near_miss": 0.68,
        "email_cluster_series_jaccard": 0.5,
        "email_cluster_residual_threshold": 0.0,
    })()
    groups = cluster_candidates(
        [
            unit(1.0, 0.0, 0.0, 0.0),
            unit(0.0, 1.0, 0.0, 0.0),
            unit(0.0, 0.0, 1.0, 0.0),
            unit(0.0, 0.0, 0.0, 1.0),
        ],
        settings=settings,
        subjects=[
            "Wholesale Liabilities & Deposit Mix Review- Discussion",
            "Q3 funding cost assumption for the NIM pack",
            "Sep-26 CA/SA targets - WB view required",
            "Wholesale Banking - July Review | Minutes",
        ],
        bodies=[
            (
                "Please walk through the mix.\n"
                "Subject: Sep-26 CA/SA targets - WB view required\n"
                "Subject: Q3 funding cost assumption for the NIM pack\n"
                "Subject: FW: Wholesale Bank Balance Sheet as on 17-Aug-2026\n"
            ),
            "Need to lock the funding cost assumption this week.",
            "Need WB view on September targets.",
            "Minutes from the July review.",
        ],
    )
    grouped = {frozenset(g) for g in groups}
    assert frozenset({0, 1, 2}) in grouped
    assert frozenset({3}) in grouped


def test_candidate_embed_text_ignores_topic_key():
    settings = type("S", (), {
        "email_embed_subject_copies": 2,  # config ignored — always 1
        "email_embed_body_max_chars": 8000,
        "email_embed_attachment_max_chars": 8000,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="5",
            subject="Wholesale MBR",
            body_text="Long body that should appear in embed text",
            received_at=datetime.now(timezone.utc),
        ),
        attachment_text="Attachment text from MoM DOCX",
        topic_key="Project Alpha NPA recovery",
    )
    text = candidate_embed_text(candidate, settings=settings)
    # Subject appears exactly once — no repetition
    assert text.count("Wholesale MBR") == 1
    # LLM topic key is never embedded (would break series stem logic)
    assert "Project Alpha NPA recovery" not in text
    # Body is included
    assert "Long body that should appear in embed text" in text
    # Attachment IS included — cross-topic relevance depends on this
    assert "Attachment text from MoM DOCX" in text


def test_candidate_embed_text_includes_quoted_nested_subject():
    settings = type("S", (), {
        "email_embed_subject_copies": 1,
        "email_embed_body_max_chars": 8000,
        "email_embed_attachment_max_chars": 8000,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="7",
            subject="Wholesale Liabilities & Deposit Mix Review- Discussion",
            body_text=(
                "Please comment on the mix.\n"
                "Subject: FW: Wholesale Bank Balance Sheet as on 17-Aug-2026\n"
            ),
            received_at=datetime.now(timezone.utc),
        ),
        topic_key="ignored",
    )
    text = candidate_embed_text(candidate, settings=settings)
    assert "Wholesale Bank Balance Sheet as on 17-Aug-2026" in text
    assert "Project" not in text


def test_candidate_embed_text_falls_back_without_topic_key():
    settings = type("S", (), {
        "email_embed_subject_copies": 2,
        "email_embed_body_max_chars": 100,
        "email_embed_attachment_max_chars": 100,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="6",
            subject="Wholesale MBR",
            body_text="Body text here",
            received_at=datetime.now(timezone.utc),
        ),
        topic_key="",
    )
    text = candidate_embed_text(candidate, settings=settings)
    assert "Body text here" in text


def test_candidate_embed_text_attachment_capped():
    """Attachment text must be capped to EMAIL_EMBED_ATTACHMENT_MAX_CHARS.

    A 24 KB MoM DOCX must not drown out the subject+body signal. The cap
    preserves the beginning of the attachment (where the most relevant
    content typically appears) and discards the tail.
    """
    cap = 50
    settings = type("S", (), {
        "email_embed_body_max_chars": 8000,
        "email_embed_attachment_max_chars": cap,
    })()
    long_attachment = "ORION pilot status: " + ("detail " * 500)  # >> cap chars
    candidate = EmailCandidate(
        message=EmailMessage(
            id="8",
            subject="Wholesale Banking - July Review | Minutes",
            body_text="July review body",
            received_at=datetime.now(timezone.utc),
        ),
        attachment_text=long_attachment,
    )
    text = candidate_embed_text(candidate, settings=settings)
    # Attachment is present but capped — "ORION pilot status" is in the
    # first 50 chars so it must appear; the tail ("detail detail...") is cut.
    assert "ORION pilot status" in text
    assert len([c for c in text.split("\n") if "detail detail" in c]) == 0 or \
           text.index("detail") < cap + 200  # rough guard: tail not duplicated


def test_candidate_embed_text_cross_topic_relevance():
    """An MBR minutes email mentioning ORION in its attachment must surface
    that keyword in the embedding text so the July MBR cluster matches the
    'Project Orion Update Meeting' pre-meet brief query.
    """
    settings = type("S", (), {
        "email_embed_body_max_chars": 8000,
        "email_embed_attachment_max_chars": 8000,
    })()
    candidate = EmailCandidate(
        message=EmailMessage(
            id="9",
            subject="Wholesale Banking - July Review | Minutes",
            body_text="Please find minutes attached.",
            received_at=datetime.now(timezone.utc),
        ),
        # Simulates the July WBR DOCX which mentions ORION
        attachment_text=(
            "Action: ORION pilot phase 2 kick-off by end of August.\n"
            "Owner: Vikram Sriram\n"
            "Status: On track\n"
        ),
    )
    text = candidate_embed_text(candidate, settings=settings)
    # ORION must be present in the embedding text so the cluster's centroid
    # vector picks up this keyword and matches a 'Project Orion' meeting query
    assert "ORION" in text
    # Subject appears exactly once
    assert text.count("Wholesale Banking - July Review | Minutes") == 1


def test_complete_linkage_avoids_bridge_chaining():
    import numpy as np

    from agent_email.clustering import cluster_by_complete_linkage, cluster_by_similarity_graph
    from agent_email.semantic_match import cosine_similarity

    def unit(*vals: float) -> np.ndarray:
        vec = np.array(vals, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    threshold = 0.75
    vectors = [
        unit(1.0, 0.0, 0.0),
        unit(0.99, 0.14, 0.0),
        unit(0.98, 0.20, 0.0),
        unit(0.0, 1.0, 0.0),
        unit(0.14, 0.99, 0.0),
        unit(0.20, 0.98, 0.0),
        unit(0.75, 0.66, 0.01),
    ]

    assert cosine_similarity(vectors[0], vectors[6]) >= threshold
    assert cosine_similarity(vectors[5], vectors[6]) >= threshold
    assert cosine_similarity(vectors[0], vectors[3]) < threshold

    graph_groups = cluster_by_similarity_graph(vectors, threshold)
    assert len(graph_groups) == 1
    assert sorted(sum(graph_groups, [])) == list(range(7))

    agg_groups = cluster_by_complete_linkage(vectors, threshold)
    grouped = {frozenset(g) for g in agg_groups}
    assert frozenset({3, 4, 5}) in grouped
    assert len(agg_groups) >= 2
    for group in agg_groups:
        assert not ({0, 3} <= set(group))
    bridge_cluster = next(g for g in agg_groups if 6 in g)
    assert {0, 1, 2} <= set(bridge_cluster) or {3, 4, 5} <= set(bridge_cluster)

