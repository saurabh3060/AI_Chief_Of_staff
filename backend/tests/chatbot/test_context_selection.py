"""
tests/chatbot/test_context_selection.py
=========================================
Unit tests for the thread selection, protected table rows and model routing
added to the chatbot prompt builder. Every build_context call passes its
settings explicitly, so nothing here reads Settings or .env.
"""

from types import SimpleNamespace

import pytest

from chatbot import retriever


def _build(threads, question, **kwargs):
    kwargs.setdefault("budget_chars", 4000)
    kwargs.setdefault("max_threads", 10)
    kwargs.setdefault("table_fraction", 0.5)
    kwargs.setdefault("ignore_words", set())
    return retriever.build_context(threads, question, **kwargs)


def _thread(subject: str, body: str) -> dict:
    return {"subject": subject, "body": body}


_TREASURY = _thread(
    "Treasury Update",
    "Treasury income for the quarter was driven by trading gains on the bond portfolio.",
)
_CARDS = _thread(
    "Card Launch",
    "The Lumiere credit card programme issued about five hundred cards in the first month.",
)
_LOANS = _thread(
    "Loan Review",
    "Retail loan disbursements grew steadily across all regions during the quarter.",
)


# A vertical table: one label line, then one value line, repeated. Every line is
# shorter than the 25-char minimum for prose, so the old filter dropped all of it.
_VERTICAL_TABLE = "\n".join([
    "Particulars", "FY25",
    "CASA ratio", "34.1%",
    "Advances growth", "13%",
    "Deposits growth", "12%",
    "Cost of funds", "5.9%",
])


def test_threads_without_keyword_match_are_dropped():
    out = _build([_TREASURY, _CARDS, _LOANS], "Lumiere card issuance")
    assert "[Email 1: Card Launch]" in out
    assert "Treasury" not in out and "Loan Review" not in out


def test_email_numbers_count_the_threads_actually_sent():
    out = _build([_TREASURY, _CARDS, _LOANS], "loan disbursements")
    assert out.startswith("[Email 1: Loan Review]")
    assert "[Email 2" not in out


def test_top_threads_limit_keeps_best_ranked_matches():
    threads = [
        _thread("A", "Quarter results for the wholesale book were reviewed by the committee."),
        _thread("B", "Quarter results for the retail book were reviewed by the committee."),
        _thread("C", "Quarter results for the cards book were reviewed by the committee."),
    ]
    stats: dict = {}
    out = _build(threads, "quarter results book committee", max_threads=2, stats=stats)
    assert "[Email 1: A]" in out and "[Email 2: B]" in out
    assert "[Email 3" not in out and "cards book" not in out
    assert stats["threads_in"] == 3 and stats["threads_used"] == 2


def test_ignore_words_do_not_create_a_keyword_match():
    # "treasury" is the only shared word; ignoring it leaves nothing to match on
    # except the other words, which the email does not contain.
    out = _build([_TREASURY], "treasury sanctions", ignore_words={"treasury"})
    assert out == "(no relevant emails found)"


def test_subject_match_alone_keeps_a_thread():
    thread = _thread("Lumiere Cards Weekly", "Short note, details to follow shortly in a later mail.")
    out = _build([thread], "lumiere")
    assert "[Email 1: Lumiere Cards Weekly]" in out


def test_vertical_table_rows_are_joined_and_kept():
    thread = _thread("Business Data", "Please find the numbers below.\n" + _VERTICAL_TABLE)
    out = _build([thread], "CASA ratio decline")
    assert "CASA ratio: 34.1%" in out
    assert "Advances growth: 13%" in out
    assert "Cost of funds: 5.9%" in out


def test_table_rows_survive_dedup_and_scoring():
    # The same table in two threads: prose dedup would drop the second copy,
    # but table rows are protected.
    t1 = _thread("Business Data", _VERTICAL_TABLE)
    t2 = _thread("Business Data Resend", _VERTICAL_TABLE)
    out = _build([t1, t2], "CASA ratio business data")
    assert out.count("CASA ratio: 34.1%") == 2


def test_table_rows_cut_only_when_table_budget_is_exhausted_and_warned(monkeypatch):
    warnings: list[str] = []
    fake = SimpleNamespace(
        warning=lambda msg, *a: warnings.append(msg % a),
        info=lambda *a, **k: None,
    )
    monkeypatch.setattr(retriever, "logger", fake)

    thread = _thread("Business Data", _VERTICAL_TABLE)
    stats: dict = {}
    # 80 chars * 0.5 = 40 chars of table budget: only the first rows fit.
    out = _build([thread], "CASA ratio", budget_chars=80, stats=stats)

    assert stats["table_rows_cut"] > 0 and stats["table_rows_kept"] > 0
    assert any("Business Data" in w and "table rows" in w for w in warnings)
    assert "[Email 1: Business Data]" in out


def test_long_prose_is_not_mistaken_for_a_table():
    body = "\n".join([
        "The committee reviewed the proposal in detail and asked for a revised plan.",
        "Management agreed to share the revised plan with all members next week.",
        "Several members raised concerns about the timeline for the rollout phase.",
    ])
    prose, rows = retriever._extract_tables(body)
    assert rows == [] and "committee" in prose


def test_selected_thread_is_never_left_with_zero_lines():
    # Budget far smaller than any sentence: the old code kept nothing.
    threads = [_CARDS, _LOANS]
    out = _build(threads, "card loan quarter month", budget_chars=10, table_fraction=0.0)
    assert "[Email 1: Card Launch]" in out and "[Email 2: Loan Review]" in out
    for block in out.split("\n\n"):
        assert len(block.splitlines()) >= 2  # header + at least one line


def test_question_words_ignore_question_phrasing():
    words = retriever._question_words("How many Lumiere cards were sold?")
    assert "lumiere" in words and "cards" in words
    assert "many" not in words and "how" not in words


# ── Model routing (service) ───────────────────────────────────

def _settings(**over):
    base = dict(
        chatbot_model_routing=False,
        chatbot_llm_model="default-model",
        chatbot_llm_model_fast="fast-model",
        chatbot_llm_model_accurate="accurate-model",
        chatbot_numeric_question_pattern=r"\bhow\s+(much|many)\b|\bwhat\s+(percentage|percent)\b",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("routing,question,expected", [
    (False, "How many cards were sold?", "default-model"),
    (False, "Summarise the treasury update", "default-model"),
    (True, "How many cards were sold?", "accurate-model"),
    (True, "By how much did CASA decline?", "accurate-model"),
    (True, "What percentage of cards are active?", "accurate-model"),
    (True, "Summarise the treasury update", "fast-model"),
])
def test_model_routing(monkeypatch, routing, question, expected):
    service = pytest.importorskip("chatbot.service")

    monkeypatch.setattr(service, "get_settings", lambda: _settings(chatbot_model_routing=routing))
    assert service._select_model(question) == expected


def test_bad_numeric_pattern_falls_back_to_fast_model(monkeypatch):
    service = pytest.importorskip("chatbot.service")

    monkeypatch.setattr(
        service, "get_settings",
        lambda: _settings(chatbot_model_routing=True, chatbot_numeric_question_pattern="(unclosed"),
    )
    assert service._select_model("How many cards?") == "fast-model"
