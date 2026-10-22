"""
tests/chatbot/test_context_selection.py
=========================================
Unit tests for block formation, thread/block selection and model routing in the
chatbot prompt builder. Every build_context call passes a ContextConfig built
here, so nothing reads Settings or .env.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from chatbot import retriever
from chatbot.retriever import ContextConfig


def make_cfg(**overrides) -> ContextConfig:
    values = dict(
        max_chars=4000, top_threads=10, neighbor_blocks=1, min_thread_share=0.2,
        figure_bonus=0.1, use_embeddings=False, embedding_weight=0.5,
        dedup_similarity=0.9, dedup_min_chars=60, short_line_chars=60,
        table_min_lines=2, table_min_numbers=3, block_max_chars=800,
        partial_head_share=0.5, thin_body_chars=300, attachment_max_chars=1500,
        signature_max_lines=6, signature_max_words=6, signature_caps_share=0.5,
        signature_address_pattern=r"\b(road|street|floor)\b",
    )
    values.update(overrides)
    return ContextConfig(**values)


def build(threads, question, budget=None, stats=None, **cfg_overrides):
    return retriever.build_context(
        threads, question, cfg=make_cfg(**cfg_overrides), stats=stats, budget_chars=budget,
    )


@pytest.fixture
def warnings(monkeypatch):
    """Capture build_context's WARNING log lines."""
    captured: list[str] = []
    fake = SimpleNamespace(warning=lambda msg, *a: captured.append(msg % a), info=lambda *a, **k: None)
    monkeypatch.setattr(retriever, "logger", fake)
    return captured


def _thread(subject: str, body: str, **extra) -> dict:
    return {"subject": subject, "body": body, **extra}


# A block shaped like the failing real thread: a heading, a label, a value line
# with more than 4 letters ("<number>+ <3 words> <1 word>"), a quoted line,
# another label, then a "<number>%" value line.
_SHAPED_LINES = [
    "Targets for the coming year",
    "Accounts opened by year end",
    "5,000+ new savings accounts monthly",
    '"Expand the partner referral network"',
    "Time to onboard a customer",
    "35% faster than the current cycle",
]
_PAIRS = [
    ("Accounts opened by year end", "5,000+ new savings accounts monthly"),
    ("Time to onboard a customer", "35% faster than the current cycle"),
]
_LONG_FILLER_A = (
    "The weekly operations review covered staffing, facilities and the schedule of "
    "upcoming maintenance windows across all of the regional offices."
)
_LONG_FILLER_C = (
    "Please also note that the quarterly calendar for training sessions has been "
    "updated and shared with every team lead for their planning."
)


def _shaped_thread() -> dict:
    body = _LONG_FILLER_A + "\n\n" + "\n".join(_SHAPED_LINES) + "\n\n" + _LONG_FILLER_C
    return _thread("Planning Notes", body)


def _assert_pairs_intact(out: str) -> None:
    """A label is never present without the value line right under it, and a
    value line is never present without its label right above it."""
    lines = out.splitlines()
    for label, value in _PAIRS:
        if label in lines:
            i = lines.index(label)
            assert i + 1 < len(lines) and lines[i + 1] == value, f"label without its value:\n{out}"
        if value in lines:
            i = lines.index(value)
            assert i > 0 and lines[i - 1] == label, f"value without its label:\n{out}"


def _assert_block_whole(out: str, block_lines: list[str], warnings: list[str]) -> None:
    """A block is all there, contiguous and in order, unless a cut was logged."""
    lines = out.splitlines()
    present = [line in lines for line in block_lines]
    if all(present):
        idx = [lines.index(line) for line in block_lines]
        assert idx == list(range(idx[0], idx[0] + len(idx))), f"block split:\n{out}"
    elif any(present):
        assert any("larger than the room left" in w for w in warnings), f"silent cut:\n{out}"


# ── The failing thread (Case A and Case B) ────────────────────

@pytest.mark.parametrize("budget", range(100, 1500, 50))
def test_labels_stay_with_their_values_at_every_budget(budget, warnings):
    out = build(
        [_shaped_thread()],
        "What are the targets for accounts opened and time to onboard?",
        budget=budget,
    )
    _assert_pairs_intact(out)
    _assert_block_whole(out, _SHAPED_LINES, warnings)


def test_shaped_block_is_one_atomic_table_block():
    paragraphs = retriever._clean_paragraphs("\n".join(_SHAPED_LINES), make_cfg())
    blocks = retriever._form_blocks(paragraphs, make_cfg())
    assert len(blocks) == 1 and blocks[0].kind == "table" and blocks[0].lines == _SHAPED_LINES


def test_short_value_lines_are_not_dropped_by_a_length_rule():
    body = (
        "Please find the planned targets for the coming year below.\n"
        "Target for segment one is set to grow to about 1,200 units\n+15%\n"
        "Target for segment two is set to grow to about 800 units\n+9%"
    )
    out = build([_thread("Segment Targets", body)], "growth target for segment one and segment two", budget=250)
    assert "segment one is set to grow to about 1,200 units\n+15%" in out
    assert "segment two is set to grow to about 800 units\n+9%" in out


def test_table_cells_separated_by_blank_lines_stay_together():
    body = "Metric alpha\n\n34.1%\n\nMetric beta\n\n13%\n\nMetric gamma\n\n12%"
    out = build([_thread("Key Ratios", body)], "metric beta")
    assert "Metric beta\n13%" in out


def test_bullet_group_and_its_intro_line_are_one_block():
    body = (
        "Please review the long note above before the call so that everyone has the same background.\n\n"
        "Key actions agreed:\n- first action item here\n- second action item here\n- third action item here"
    )
    blocks = retriever._form_blocks(retriever._clean_paragraphs(body, make_cfg()), make_cfg())
    assert [b.kind for b in blocks] == ["prose", "bullets"]
    assert blocks[1].lines[0] == "Key actions agreed:" and len(blocks[1].lines) == 4


# ── Step 3: selection ─────────────────────────────────────────

def test_everything_that_fits_is_sent_unchanged():
    stats: dict = {}
    out = build([_shaped_thread()], "targets for accounts", stats=stats)
    assert stats["mode"] == "everything_fit"
    assert "\n".join(_SHAPED_LINES) in out and _LONG_FILLER_A in out and _LONG_FILLER_C in out
    assert stats["blocks_kept"] == stats["blocks_total"] == 3


def test_selection_is_used_when_the_text_is_too_big():
    stats: dict = {}
    out = build([_shaped_thread()], "targets for accounts opened", budget=300, stats=stats)
    assert stats["mode"] == "selection" and stats["blocks_kept"] < stats["blocks_total"]
    assert len(out) <= 300
    assert "\n".join(_SHAPED_LINES) in out   # the best-matching block wins


def test_every_part_of_a_multi_part_question_is_covered():
    targets = [
        "The plan sets the target for segment one at roughly twelve hundred units over the coming year.",
        "The plan sets the target for segment two at roughly eight hundred units over the coming year.",
        "The plan sets the target for segment three at roughly twenty five hundred units over the coming year.",
    ]
    fillers = [
        f"A review of the regional segment targets was postponed until committee {name} has more time to meet."
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    ]
    paragraphs = [fillers[0], targets[0], fillers[1], fillers[2], targets[1], fillers[3], fillers[4], targets[2], fillers[5]]
    out = build(
        [_thread("Plan", "\n\n".join(paragraphs))],
        "What are the targets for segment one, segment two and segment three?",
        budget=520,
    )
    for target in targets:
        assert target in out


def test_sub_questions_split_on_separators_but_not_inside_numbers():
    queries = retriever._sub_questions("sales in 1,200 branches, margin and costs; 2) risk", frozenset())
    assert queries[0].startswith("sales in 1,200 branches")
    assert "sales in 1,200 branches" in queries and "margin" in queries and "costs" in queries and "risk" in queries


def test_every_thread_sent_gets_at_least_one_line():
    threads = [
        _thread("Card Launch", "The new card programme issued about five hundred cards in the first month."),
        _thread("Loan Review", "Retail loan disbursements grew steadily across all regions during the quarter."),
    ]
    out = build(threads, "card loan programme disbursements", budget=60)
    for block in out.split("\n\n"):
        assert len(block.splitlines()) >= 2   # header + at least one line


def test_oversize_block_is_cut_with_a_warning_and_keeps_labels_with_values(warnings):
    lines = []
    for i in range(1, 31):
        lines += [f"Metric {i}", f"{i * 3}.5%"]
    stats: dict = {}
    out = build([_thread("Big Table", "\n".join(lines))], "metric 7", budget=260, stats=stats)

    assert len(out) <= 260
    assert any("Big Table" in w and "larger than the room left" in w for w in warnings)
    assert stats["cuts"] and stats["cuts"][0]["lines_kept"] < stats["cuts"][0]["lines_total"]
    kept = out.splitlines()
    for i, line in enumerate(kept):
        if line.endswith("%"):
            n = int(float(line[:-1]) / 3)
            assert kept[i - 1] == f"Metric {n}", "a value line was kept without its label"
    # the table's first part is kept
    assert "Metric 1" in kept and "3.5%" in kept


def test_neighbour_block_comes_along_with_the_best_block():
    body = "\n\n".join([
        "An unrelated paragraph about the cafeteria menu and the schedule for next week's lunches.",
        "The savings programme opened about 500 accounts in the first month of the launch period.",
        "A second unrelated paragraph about parking permits and the office shuttle timetable today.",
        "A third unrelated paragraph about visitor badges and the front desk opening hours daily.",
    ])
    out = build([_thread("Update", body)], "how many accounts were opened", budget=300, neighbor_blocks=1)
    assert "opened about 500 accounts" in out
    assert "cafeteria menu" in out and "parking permits" in out      # one neighbour each side
    assert "visitor badges" not in out


# ── Thread choice ─────────────────────────────────────────────

_TREASURY = _thread("Treasury Update", "Treasury income for the quarter was driven by trading gains on the portfolio.")
_CARDS = _thread("Card Launch", "The Lumiere credit card programme issued about five hundred cards in the first month.")
_LOANS = _thread("Loan Review", "Retail loan disbursements grew steadily across all regions during the quarter.")


def test_threads_without_keyword_match_are_dropped():
    out = build([_TREASURY, _CARDS, _LOANS], "Lumiere card issuance")
    assert "[Email 1: Card Launch]" in out
    assert "Treasury" not in out and "Loan Review" not in out


def test_email_numbers_count_the_threads_actually_sent():
    out = build([_TREASURY, _CARDS, _LOANS], "loan disbursements")
    assert out.startswith("[Email 1: Loan Review]") and "[Email 2" not in out


def test_top_threads_limit_keeps_best_ranked_matches():
    threads = [
        _thread("A", "Quarter results for the wholesale book were reviewed by the committee."),
        _thread("B", "Quarter results for the retail book were reviewed by the committee."),
        _thread("C", "Quarter results for the cards book were reviewed by the committee."),
    ]
    stats: dict = {}
    out = build(threads, "quarter results book committee", stats=stats, top_threads=2)
    assert "[Email 1: A]" in out and "[Email 2: B]" in out and "[Email 3" not in out
    assert stats["threads_in"] == 3 and stats["threads_used"] == 2


def test_ignore_words_do_not_create_a_keyword_match():
    out = build([_TREASURY], "treasury sanctions", ignore_words=frozenset({"treasury"}))
    assert out == "(no relevant emails found)"


def test_subject_match_alone_keeps_a_thread():
    thread = _thread("Lumiere Cards Weekly", "Short note, details to follow shortly in a later mail.")
    assert "[Email 1: Lumiere Cards Weekly]" in build([thread], "lumiere")


def test_question_words_ignore_question_phrasing():
    words = retriever._question_words("How many Lumiere cards were sold?")
    assert "lumiere" in words and "cards" in words and "many" not in words


# ── Step 1: cleaning details ──────────────────────────────────

def test_near_duplicate_with_the_same_figures_is_removed_but_changed_figures_are_kept():
    base = "The committee approved the revised plan with an overall limit of {n} units for the coming year."
    same = [_thread("One", base.format(n=450)), _thread("Two", base.format(n=450))]
    changed = [_thread("One", base.format(n=450)), _thread("Two", base.format(n=475))]
    assert build(same, "overall limit committee").count("overall limit of 450 units") == 1
    out = build(changed, "overall limit committee")
    assert "limit of 450 units" in out and "limit of 475 units" in out


def test_mojibake_and_invisible_characters_are_repaired():
    body = "The target isnâ€™t changed this year​ and stays at five percent for the segment."
    out = build([_thread("Targets", body)], "target segment")
    assert "isn’t changed this year and stays" in out


def test_signature_after_a_sign_off_is_removed_but_content_sentences_are_not():
    body = (
        "Thanks for the update, the segment target is now 5 percent for the year.\n"
        "Next steps will be shared after the committee meets on Monday.\n\n"
        "Thanks and regards,\nA. Person\nHead of Planning\n\n"
        "A later paragraph with more detail about the segment target and its timeline."
    )
    out = build([_thread("Plan", body)], "segment target")
    assert "Thanks for the update, the segment target is now 5 percent" in out
    assert "Head of Planning" not in out and "A. Person" not in out
    assert "A later paragraph with more detail" in out


_BODY_LINE = "The segment target is confirmed for the coming year after the committee review."


def test_signature_block_with_a_figure_is_kept():
    out = build([_thread("Plan", f"{_BODY_LINE}\nRegards,\nA. Person\nTarget 5 percent")], "segment target")
    assert "A. Person" in out and "Target 5 percent" in out


def test_signature_with_name_title_and_address_is_removed():
    body = f"{_BODY_LINE}\nRegards,\nA. Person\nHead of Planning\n12 Main Road, Pune"
    out = build([_thread("Plan", body)], "segment target")
    assert _BODY_LINE in out
    assert "A. Person" not in out and "Head of Planning" not in out and "Main Road" not in out


def test_signature_longer_than_the_line_limit_is_kept():
    names = "\n".join(f"{word} Team" for word in ("First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"))
    out = build([_thread("Plan", f"{_BODY_LINE}\nRegards,\n{names}")], "segment target")
    assert "First Team" in out and "Seventh Team" in out


def test_signature_at_the_end_of_a_message_without_a_sign_off_is_removed():
    out = build([_thread("Plan", f"{_BODY_LINE}\n\nA. Person\nHead of Planning")], "segment target")
    assert _BODY_LINE in out and "A. Person" not in out and "Head of Planning" not in out


def test_the_only_paragraph_of_a_message_is_never_removed_as_a_signature():
    out = build([_thread("Plan Notes", "Segment Owners\nA. Person")], "segment owners")
    assert "A. Person" in out


def test_last_paragraph_with_a_figure_is_kept():
    out = build([_thread("Plan", f"{_BODY_LINE}\n\nTarget 5 percent")], "segment target")
    assert "Target 5 percent" in out


def test_thin_body_draws_matching_attachment_blocks_only():
    thread = _thread(
        "Short Mail", "See attached.",
        attachment_data="Revenue by segment rose to 12 percent in the year.\n\nThe cafeteria will be closed on Friday afternoon.",
    )
    out = build([thread], "revenue by segment")
    assert "Revenue by segment rose to 12 percent" in out and "cafeteria" not in out


# ── Embeddings (Step 3b) ──────────────────────────────────────

def _selection_case():
    return [_shaped_thread()], "targets for accounts opened"


def test_embeddings_are_used_for_selection_and_skipped_when_everything_fits(monkeypatch):
    calls = []

    def fake_embed(texts):
        calls.append(len(texts))
        return np.full((len(texts), 4), 0.5, dtype=np.float32)

    monkeypatch.setattr(retriever, "embed_texts", fake_embed)
    threads, question = _selection_case()

    stats: dict = {}
    build(threads, question, budget=300, stats=stats, use_embeddings=True)
    assert stats["embeddings"] == "on" and len(calls) == 1      # one batched call

    calls.clear()
    stats = {}
    build(threads, question, stats=stats, use_embeddings=True)  # everything fits
    assert stats["embeddings"] == "skipped" and calls == []


def test_embedding_failure_falls_back_to_bm25(monkeypatch, warnings):
    def broken(texts):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(retriever, "embed_texts", broken)
    threads, question = _selection_case()
    stats: dict = {}
    out = build(threads, question, budget=300, stats=stats, use_embeddings=True)
    assert stats["embeddings"] == "failed" and "\n".join(_SHAPED_LINES) in out
    assert any("embedding failed" in w for w in warnings)


# ── Model routing (service) ───────────────────────────────────

def _settings(**over):
    base = dict(
        chatbot_model_routing=False,
        chatbot_llm_model="default-model",
        chatbot_llm_model_fast="fast-model",
        chatbot_llm_model_accurate="accurate-model",
        chatbot_numeric_question_pattern=r"\bhow\s+(much|many)\b|\bwhat\s+(percentage|percent)\b",
        chatbot_calc_question_pattern=r"\bcalculate\b|\btotal\b",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("routing,question,expected", [
    (False, "How many cards were sold?", "default-model"),
    (False, "Summarise the treasury update", "default-model"),
    (True, "How many cards were sold?", "accurate-model"),
    (True, "By how much did CASA decline?", "accurate-model"),
    (True, "What percentage of cards are active?", "accurate-model"),
    (True, "Calculate the combined figure", "accurate-model"),   # calculation pattern only
    (False, "Calculate the combined figure", "default-model"),   # routing off: unchanged
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


# ── Arithmetic in code (service) ──────────────────────────────

@pytest.mark.parametrize("expression,expected", [
    ("2*3+4/2", 8.0),
    ("(1+2)**2", 9.0),
    ("-5+2", -3.0),
    ("100 / (1 + 15/100)", 100 / 1.15),
])
def test_safe_eval_accepts_plain_arithmetic(expression, expected):
    service = pytest.importorskip("chatbot.service")
    assert service._safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('x')", "abs(-1)", "a + 1", "1 if 1 else 2", "[1, 2]", "'x' * 3",
    "1 < 2", "1e999 * 1e999", "9 ** 9 ** 9", "(-8) ** 0.5", "1 / 0", "",
])
def test_safe_eval_rejects_everything_else(expression):
    service = pytest.importorskip("chatbot.service")
    with pytest.raises(Exception):
        service._safe_eval(expression)


def test_expression_cleaning_and_number_formatting():
    service = pytest.importorskip("chatbot.service")
    assert service._clean_expression("```\n1,200 * 3\n```") == "1200 * 3"
    assert service._clean_expression("1,20,000 / 4\nextra words") == "120000 / 4"
    assert service._format_number(6521.7391) == "6521.74"
    assert service._format_number(1800.0) == "1800"
    assert service._format_number(0.5) == "0.5"


def _calc_settings(**over):
    values = dict(
        chatbot_calc_in_code=True, chatbot_calc_max_tokens=80, chatbot_calc_timeout_s=20.0,
        ollama_host="http://ollama.invalid", chatbot_max_tokens=256, chatbot_temperature=0.3,
        chatbot_repeat_penalty=1.0, chatbot_llm_num_ctx=4096, chatbot_num_thread=8,
        chatbot_keep_alive=-1,
    )
    values.update(over)
    return _settings(**values)


def test_calculation_result_is_appended_to_the_prompt(monkeypatch):
    service = pytest.importorskip("chatbot.service")
    monkeypatch.setattr(service, "get_settings", lambda: _calc_settings())
    monkeypatch.setattr(
        service, "_compute_in_code",
        lambda context, query: {"attempted": True, "used": True, "expression": "600*3",
                                "value": 1800.0, "error": None, "seconds": 0.5},
    )
    stats = {"threads_used": 1}
    out = service._apply_code_calculation("ctx", "Calculate the total", "BASE", stats)
    assert out == "BASE\n\nComputed result: 600*3 = 1800. Use this value; do not recompute."
    assert stats["calc"]["used"] is True and stats["calc"]["expression"] == "600*3"


def test_calculation_is_skipped_when_off_or_not_a_calculation_question(monkeypatch):
    service = pytest.importorskip("chatbot.service")

    def boom(context, query):
        raise AssertionError("the extra call must not happen")

    monkeypatch.setattr(service, "_compute_in_code", boom)
    stats = {"threads_used": 1}
    monkeypatch.setattr(service, "get_settings", lambda: _calc_settings(chatbot_calc_in_code=False))
    assert service._apply_code_calculation("ctx", "Calculate the total", "BASE", stats) == "BASE"
    monkeypatch.setattr(service, "get_settings", lambda: _calc_settings())
    assert service._apply_code_calculation("ctx", "Summarise the update", "BASE", stats) == "BASE"
    assert stats["calc"]["attempted"] is False


def test_failed_calculation_leaves_the_prompt_unchanged(monkeypatch):
    service = pytest.importorskip("chatbot.service")
    monkeypatch.setattr(service, "get_settings", lambda: _calc_settings())
    monkeypatch.setattr(
        service, "_compute_in_code",
        lambda context, query: {"attempted": True, "used": False, "expression": None,
                                "value": None, "error": "ValueError: bad", "seconds": 1.0},
    )
    stats = {"threads_used": 1}
    assert service._apply_code_calculation("ctx", "Calculate the total", "BASE", stats) == "BASE"
    assert stats["calc"]["used"] is False and stats["calc"]["error"] == "ValueError: bad"


def test_compute_in_code_uses_the_accurate_model_with_the_configured_limits(monkeypatch):
    service = pytest.importorskip("chatbot.service")
    seen = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            seen["client"] = (host, kwargs)

        def chat(self, **request):
            seen["request"] = request
            return {"message": {"content": "600 * 3"}}

    monkeypatch.setattr(service, "get_settings", lambda: _calc_settings())
    monkeypatch.setattr(service, "get_calc_prompt", lambda: "CALC PROMPT")
    monkeypatch.setattr(service.ollama, "Client", FakeClient)

    result = service._compute_in_code("context text", "Calculate the total")
    assert result["used"] is True and result["expression"] == "600 * 3" and result["value"] == 1800.0
    assert seen["client"] == ("http://ollama.invalid", {"timeout": 20.0})
    request = seen["request"]
    assert request["model"] == "accurate-model"
    assert request["messages"][0] == {"role": "system", "content": "CALC PROMPT"}
    assert request["options"]["num_predict"] == 80 and request["options"]["temperature"] == 0.0


def test_compute_in_code_never_raises(monkeypatch):
    service = pytest.importorskip("chatbot.service")

    class BrokenClient:
        def __init__(self, host=None, **kwargs):
            pass

        def chat(self, **request):
            raise TimeoutError("timed out")

    monkeypatch.setattr(service, "get_settings", lambda: _calc_settings())
    monkeypatch.setattr(service, "get_calc_prompt", lambda: "CALC PROMPT")
    monkeypatch.setattr(service.ollama, "Client", BrokenClient)
    result = service._compute_in_code("context text", "Calculate the total")
    assert result["used"] is False and "TimeoutError" in result["error"]
