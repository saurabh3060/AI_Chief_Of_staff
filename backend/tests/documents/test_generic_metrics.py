"""
tests/documents/test_generic_metrics.py
========================================
Tests for the database-driven calculation rules engine.

Tests are ported from the SQLite version to async PostgreSQL, matching the
project's production stack (asyncpg + store.acquire()).

Tests fall into two groups:
  - **Unit tests** — pure-logic functions (compile_formula, _matches, _evaluate, etc.)
    that need no database at all.
  - **Integration tests** — require a live PostgreSQL database with at least
    one ingested document (CC12.pdf) and its associated series/facts.  These
    tests are marked with ``@pytest.mark.integration`` so they can be skipped
    in CI when no database is available.

Run:
    pytest tests/documents/test_generic_metrics.py -v
    pytest tests/documents/test_generic_metrics.py -k "not integration" -v   # unit only
"""
from __future__ import annotations

import json

import pytest

from documents import generic_metrics
from documents import store


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Async DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
async def conn():
    """Acquire a pooled asyncpg connection for one test, then release it."""
    async with store.acquire() as c:
        yield c


@pytest.fixture
async def doc_cc12(conn):
    """Return the CC12.pdf document row (integration tests assume it's ingested)."""
    row = await conn.fetchrow(
        "SELECT * FROM documents WHERE file_name = 'CC12.pdf'"
    )
    if row is None:
        pytest.skip("CC12.pdf not ingested — skipping integration test")
    return row


# ---------------------------------------------------------------------------
# Unit tests — infer_document_scope (uses an in-memory mock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unscoped_portfolio_question_uses_newest_matching_report(conn):
    """infer_document_scope picks the newest matching family document."""
    # Insert temporary documents
    tx = conn.transaction()
    await tx.start()
    try:
        await conn.execute(
            "INSERT INTO documents (doc_id, file_name, title, as_of_key, status) "
            "VALUES ($1, $2, $3, $4, $5)",
            "test_old", "2025-12-03 - Review - HL-LAP.pptx", "", "2025-12-03", "ready",
        )
        await conn.execute(
            "INSERT INTO documents (doc_id, file_name, title, as_of_key, status) "
            "VALUES ($1, $2, $3, $4, $5)",
            "test_new", "2026-03-06 - Review - Mortgages - Feb26.pptx", "", "2026-03-06", "ready",
        )
        await conn.execute(
            "INSERT INTO documents (doc_id, file_name, title, as_of_key, status) "
            "VALUES ($1, $2, $3, $4, $5)",
            "test_agri", "2026-02-02 - MD Sir Review - Agri - Jan 2026.pptx", "", "2026-02-02", "ready",
        )

        assert await generic_metrics.infer_document_scope(conn, "What is PLAP book growth?") == ["test_new"]
        assert await generic_metrics.infer_document_scope(conn, "What is HL-LAP book growth?") == ["test_old"]
        assert await generic_metrics.infer_document_scope(conn, "What is Agri ANR YoY?") == ["test_agri"]
        assert await generic_metrics.infer_document_scope(conn, "What is total income?") is None
    finally:
        await tx.rollback()


# ---------------------------------------------------------------------------
# Unit tests — compile_formula
# ---------------------------------------------------------------------------

def test_q_dash_four_reference_compiles_as_quarter_four():
    operation, spec, *_ = generic_metrics.compile_formula(
        "Q4 estimate", "Disbursement(Estimate,Q-4)", "463"
    )
    assert operation == "fact"
    assert spec["period_quarter"] == 4


def test_unsupported_business_formula_fails_closed():
    with pytest.raises(ValueError, match="fact reference"):
        generic_metrics.compile_formula("Ambiguous metric", "Do the appropriate adjustment", "")
    with pytest.raises(ValueError, match="period selector"):
        generic_metrics.compile_formula("Selected-period growth", "metric(m) − metric(m,base)", "")


def test_multidimensional_reference_keeps_dimensions_and_period_separate():
    _, spec, _ = generic_metrics.compile_formula("Cell amount", "A(rate bucket, maturity bucket, q-1)")
    assert spec["symbol"] == "a"
    assert spec["selectors"] == ["rate bucket", "maturity bucket"]
    assert spec["period_offset"] == -1


def test_explicit_period_is_compiled_as_a_fact_period():
    _, spec, _ = generic_metrics.compile_formula("Growth", "B(Jun-26) / B(Sep-25) - 1")
    refs = generic_metrics._refs(spec)
    assert [ref["period_label"] for ref in refs] == ["Jun-26", "Sep-25"]


def test_business_period_shorthand_keeps_scenario_and_month_or_quarter():
    _, spec, _ = generic_metrics.compile_formula(
        "Budget movement", "AUM(Total,Jan A) - AUM(Total,YTD Dec B)", "Jan A: 1"
    )
    left, right = generic_metrics._refs(spec)
    assert left["selectors"] == ["Total", "Actual"]
    assert left["period_month"] == 1
    assert right["selectors"] == ["Total", "Budget", "YTD"]
    assert right["period_month"] == 12

    _, quarter, _ = generic_metrics.compile_formula(
        "Projection", "Disbursement(RVF,Q4 E)", "Q4 E: 1"
    )
    assert quarter["selectors"] == ["RVF", "Estimate"]
    assert quarter["period_quarter"] == 4


def test_scenario_prefix_and_qualified_period_are_compiled_separately():
    _, actual, _ = generic_metrics.compile_formula(
        "Actual book", "EOP(Agri,Actuals Jan'25)", "Jan-25: 1"
    )
    assert actual["selectors"] == ["Agri", "Actual"]
    assert actual["period_label"] == "Jan-25"

    _, budget, _ = generic_metrics.compile_formula(
        "Budget book", "PLAP(Book,AOP Mar-26)", "Mar-26: 1"
    )
    assert budget["selectors"] == ["Book", "Budget", "AOP"]
    assert budget["period_label"] == "Mar-26"

    _, growth, _ = generic_metrics.compile_formula(
        "Growth", "RetailAgriKCC(Disbursement,YoY Growth % Feb'26 YTD)", "Feb-26: 1"
    )
    assert growth["selectors"] == ["Disbursement", "YTD", "YoY Growth %"]
    assert growth["period_label"] == "Feb-26"


def test_executable_percent_scale_is_not_applied_twice():
    _, spec, _ = generic_metrics.compile_formula(
        "Achievement %", "actual(m) / budget(m) * 100", "96.43%"
    )
    assert generic_metrics._has_division(spec)
    assert generic_metrics._has_explicit_percent_scale(spec)


def test_generated_nested_wb_nfb_reference_is_flattened():
    operation, spec, _ = generic_metrics.compile_formula(
        "NFB", "TotalNFBOS(NFB OS(*),FY25 A)", "21,979"
    )
    assert operation == "fact"
    assert spec["symbol"] == "totalnfbos"
    assert spec["period_label"] == "FY25"
    assert "Actual" in spec["selectors"]

    _, bare_spec, _ = generic_metrics.compile_formula(
        "NFB", "TotalNFBOS(NFB OS,FY25 A)", "21,979"
    )
    assert bare_spec["symbol"] == "totalnfbos"


def test_generated_mfi_resolution_reference_is_flattened():
    operation, spec, _ = generic_metrics.compile_formula(
        "Resolution rate", "ResolutionRate(SMA(*)-0,Nov-25) * 100", "46.26%"
    )
    assert operation == "arithmetic"
    ref = generic_metrics._refs(spec)[0]
    assert ref["symbol"] == "resolutionrate"
    assert ref["selectors"] == ["Bucket0"]
    assert ref["period_label"] == "Nov-25"

    _, flat_spec, _ = generic_metrics.compile_formula(
        "Resolution rate", "ResolutionRate(SMA-0,Nov-25) * 100", "46.26%"
    )
    assert generic_metrics._refs(flat_spec)[0]["selectors"] == ["Bucket0"]


def test_mfi_compact_sma_symbol_carries_its_encoded_period():
    ref = generic_metrics._reference("SMANov25", "BR, SMA0 RR")
    assert ref["period_label"] == "Nov-25"
    assert ref["percent_fraction"] is True

    pos = generic_metrics._reference("SMANov25", "BR, POS")
    assert pos["percent_fraction"] is False


def test_yearless_half_is_a_period_selector_not_entity_text():
    _operation, spec, _unit = generic_metrics.compile_formula(
        "H1 slippages", "Slippages(Agri projection,H-1)", "H-1: 28.50"
    )
    assert spec["period_half"] == 1
    assert spec["selectors"] == ["Estimate", "Agri projection"]


# ---------------------------------------------------------------------------
# Unit tests — import_rows
# ---------------------------------------------------------------------------

def test_imports_the_six_column_pasted_table(tmp_path):
    table = tmp_path / "metrics.tsv"
    table.write_text(
        "#\tMetric family\tPipeline-ready metric\tDefinition / formula\tApplies to\tExample\n"
        "1\tFact lookup\tMonthly Net Addition\tna(m), as reported\tAll months\tFeb-26: 25,670\n"
        "2\tTrend\tNet Addition MoM change\tna(m) − na(m−1)\tMonthly\tDerived\n",
        encoding="utf-8",
    )
    rows = generic_metrics.import_rows(str(table), default_page=4, apply=False)
    assert [row["rule_id"] for row in rows] == ["g_p4_001", "g_p4_002"]
    assert all(row["status"] == "approved" for row in rows)


def test_import_can_preserve_business_definition_with_safe_executable_formula(tmp_path):
    table = tmp_path / "metrics.tsv"
    table.write_text(
        "#\tMetric family\tPipeline-ready metric\tDefinition / formula\tExecutable formula\tApplies to\tExample\n"
        "1\tProduct mix\tMonthly Proprietary mix\tProprietary sourcing(m) ÷ Total sourcing(m) × 100\t"
        "proprietary(m) / total(m) * 100\tMonthly\tFeb-26: 57.93%\n",
        encoding="utf-8",
    )
    row = generic_metrics.import_rows(str(table), default_page=3, apply=False)[0]
    assert row["status"] == "approved"
    assert row["definition"] == "Proprietary sourcing(m) ÷ Total sourcing(m) × 100"
    assert row["compiled_spec"]["type"] == "binary"


def test_imports_optional_routing_and_interpretation_columns(tmp_path):
    table = tmp_path / "routed.tsv"
    table.write_text(
        "#\tMetric family\tPipeline-ready metric\tDefinition / formula\tApplies to\tExample\t"
        "Aliases\tRequired any\tPeriod scope\tSuccess condition\tTrue text\tFalse text\n"
        "1\tEfficiency\tCoverage ratio\tincome(m) / cost(m)\tYTD\t1.26x\tfully cover\t"
        "replace;exited\tYTD\tvalue >= 1\tYes at {formatted_value}\tNo at {formatted_value}\n",
        encoding="utf-8",
    )
    rows = generic_metrics.import_rows(str(table), default_page=18, apply=False)
    assert rows[0]["routing_spec"] == {"required_any": ["replace", "exited"], "period_scope": "YTD"}
    assert rows[0]["interpretation_spec"]["success_condition"] == "value >= 1"


# ---------------------------------------------------------------------------
# Unit tests — _matches (pure logic, no DB)
# ---------------------------------------------------------------------------

def test_compact_workbook_symbol_matches_spaced_report_label():
    series = generic_metrics.Series(
        1, "Review › Actual › Interest Income", "Interest Income", "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "Q1FY26", "Q1FY26", "Q", "2025-04-01", 10.0, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "interestincome", "entity": "Actual", "selectors": ["Actual"],
        "period_label": "Q1FY26", "period_offset": 0,
    })


def test_source_portfolio_selector_is_scoped_and_projection_still_matches():
    series = generic_metrics.Series(
        1, "FLOW AND RECOVERY › Proj › Slippages", "Slippages", "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "H:FY26-H1", "H1FY26", "H", "2025-04-01", 28.5, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "slippages", "entity": "Agri projection",
        "selectors": ["Agri projection"], "source_name": "2026 Review Agri.pptx",
        "period_offset": 0,
    })


def test_ytd_fiscal_year_matches_partial_source_period():
    series = generic_metrics.Series(
        1, "Prime OMT P&L › Actuals YTD › Disbursement", "Disbursement", "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "FY:FY26", "FY26 (partial)", "FY", "2025-04-01", 2978.0, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "disbursement", "entity": "YTD", "selectors": ["YTD"],
        "period_label": "FY26", "period_offset": 0,
    })
    assert generic_metrics._period_label_matches(
        "FY26 (partial)", "FY26", {"selectors": ["YTD"]}
    )


def test_bbg_pool_symbol_maps_to_anonymized_tat_row():
    series = generic_metrics.Series(
        1, "TAT Reduction Project for BBG › Pool 1 › Punjab › [names omitted]",
        "[names omitted]", "", "days", 1.0, [],
        [generic_metrics.Fact(1, "M:2026-02", "Feb-26", "M", "2026-02-01", 8.0, "days")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "punjabsachinpool", "entity": "E2E TAT", "selectors": ["E2E TAT"],
        "source_name": "2026 Review BBG.pptx", "period_label": "Feb-26", "period_offset": 0,
    })


def test_mfi_compound_partner_symbol_maps_to_structured_path():
    series = generic_metrics.Series(
        1, "Disbursement FY26 › BC Partner › RFL › Actual", "Actual", "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "M:2025-04", "Apr-25", "M", "2025-04-01", 332.0, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "rflactual", "entity": "RBL MFI", "selectors": ["RBL MFI"],
        "source_name": "MFI.pdf", "period_label": "Apr-25", "period_offset": 0,
    })


def test_rbg_central_functions_cost_accepts_correct_source_spelling():
    series = generic_metrics.Series(
        1, "RBG P&L › Actual › YTD › Central functions cost", "Central functions cost",
        "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "M:2026-01", "Jan-26", "M", "2026-01-01", 12.0, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "centralfunctionscost", "entity": "RBG | Actual | YTD",
        "selectors": ["RBG", "Actual", "YTD"],
        "source_name": "2026 Review RBG.pptx", "period_label": "Jan-26",
        "period_offset": 0,
    })


def test_total_selector_can_mean_the_primary_metric_row():
    series = generic_metrics.Series(
        1, "Vehicle Finance › YTD Actual › AUM", "AUM", "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "M:2025-12", "Dec-25", "M", "2025-12-01", 3771, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "aum", "entity": "Total", "selectors": ["Total"],
        "source_name": "2026 Review RVF.pptx", "period_offset": 0,
    })


def test_wb_untagged_january_series_is_treated_as_actual():
    series = generic_metrics.Series(
        1, "Profit & Loss Statement- Corporate › Other Income › PPOP", "PPOP", "", "Cr", 1.0, [],
        [generic_metrics.Fact(1, "M:2026-01", "Jan-26", "M", "2026-01-01", 29.0, "Cr")],
    )
    assert generic_metrics._matches(series, {
        "symbol": "ppop", "entity": "Corporate | Actual",
        "selectors": ["Corporate", "Actual"],
        "source_name": "2026 Review Wholesale Banking.pptx",
        "period_label": "Jan-26", "period_offset": 0,
    })


# ---------------------------------------------------------------------------
# Unit tests — _evaluate
# ---------------------------------------------------------------------------

def test_arithmetic_normalizes_percent_facts_to_fractions():
    fact = generic_metrics.Fact(1, "M:2025-11", "Nov-25", "M", "2025-11-01", 53.72, "%")
    series = generic_metrics.Series(1, "Resolution › BR", "BR", "", "%", 1.0, [], [fact])
    ref = {"type": "ref", "period_offset": 0, "percent_fraction": True}
    spec = {"type": "binary", "op": "*", "left": ref,
            "right": {"type": "number", "value": 100.0}}
    bindings = {id(ref): {"*": series}}

    result = generic_metrics._evaluate(
        spec, bindings, "*", fact, normalize_percent=True
    )
    assert result.value == pytest.approx(53.72)


# ---------------------------------------------------------------------------
# Unit tests — document_family / document_pattern
# ---------------------------------------------------------------------------

def test_recurring_document_family_ignores_report_date_and_delivery_suffix():
    assert generic_metrics.document_family(
        "2026-03-06 - Review - RBG - Feb 2026.pptx"
    ) == generic_metrics.document_family(
        "2026-05-08 - Review - RBG - Apr 2026.pptx"
    ) == "rbg_review"
    assert generic_metrics.document_family(
        "01_Book_Growth_Analysis_4786.pdf"
    ) == generic_metrics.document_family(
        "01_Book_Growth_Analysis_9012.pdf"
    ) == "book_growth_analysis"


def test_family_rule_applies_to_new_filename_but_not_another_portfolio():
    rule = {
        "source_name": "2026-03-06 - Review - RVF - Feb 2026.pptx",
        "document_pattern": "family:rvf_review",
    }
    assert generic_metrics._rule_applies_to_document(rule, {
        "file_name": "2026-04-06 - Review - RVF - Mar 2026.pptx", "title": "Vehicle Finance",
    })
    assert not generic_metrics._rule_applies_to_document(rule, {
        "file_name": "2026-04-06 - Review - RBG - Mar 2026.pptx", "title": "RBG Review",
    })


# ---------------------------------------------------------------------------
# Unit tests — _validate_example
# ---------------------------------------------------------------------------

def test_numeric_example_validation():
    rule = {"example": "Nov-25: +0.33 pp"}
    assert generic_metrics._validate_example(rule, [{"period_label": "Nov-25", "value": 0.33}])
    assert not generic_metrics._validate_example(rule, [{"period_label": "Nov-25", "value": 0.21}])
    assert generic_metrics._validate_example(
        {"example": "−1,67,323"}, [{"period_label": "Apr-25 to Feb-26", "value": -167323}]
    )
    assert generic_metrics._validate_example(
        {"example": "Jun 73.3% → Feb 88.5%"},
        [{"period_label": "Jun-25", "value": 73.30}, {"period_label": "Feb-26", "value": 88.48}],
    )


def test_interpretation_condition_renders_business_conclusion():
    row = {"formatted_value": "1.26x", "value": 1.26, "entity": "Portfolio",
           "period_label": "YTD", "metric_name": "Coverage ratio",
           "answer": "Coverage ratio is 1.26x.", "source_page": 18,
           "interpretation_spec": {"success_condition": "value >= 1",
                                   "true_text": "Yes, fully replaced at {formatted_value}",
                                   "false_text": "No"}}
    assert generic_metrics.deterministic_answer([row]) == "Yes, fully replaced at 1.26x. (p.18)"


# ---------------------------------------------------------------------------
# Integration tests — require live PostgreSQL with CC12.pdf ingested
# ---------------------------------------------------------------------------

@integration
@pytest.mark.asyncio
async def test_generic_materializer_combines_uniquely_named_rows_on_a_page(conn, doc_cc12):
    operation, spec, unit = generic_metrics.compile_formula(
        "Monthly Proprietary mix", "proprietary(m) / total(m) * 100", "Feb-26: 57.93%"
    )
    rule = {
        "source_page": 3, "compiled_spec": json.dumps(spec), "applies_to": "Monthly",
        "output_type": unit, "rule_id": "test_product_mix", "metric_name": "Monthly Proprietary mix",
        "metric_family": "Product mix", "definition": "Proprietary / Total * 100", "version": 1,
    }
    rows = await generic_metrics._materialize_rule(conn, rule, doc_cc12)
    february = next(row for row in rows if row["period_label"] == "Feb-26")
    assert abs(february["value"] - 57.93) < 0.01
    assert len(json.loads(february["source_fact_ids"])) == 2


@integration
@pytest.mark.asyncio
async def test_cif_and_net_addition_symbols_match_their_own_page4_rows(conn, doc_cc12):
    series = await generic_metrics._series_for_page(conn, doc_cc12["doc_id"], 4)
    cif = [s.name for s in series if generic_metrics._matches(
        s, {"symbol": "cif", "entity": "*", "period_offset": 0})]
    additions = [s.name for s in series if generic_metrics._matches(
        s, {"symbol": "na", "entity": "*", "period_offset": 0})]
    assert cif == ["CIF"]
    assert additions == ["Net Addition"]


@integration
@pytest.mark.asyncio
async def test_fixed_result_month_reuses_same_slot_for_next_family_document(conn, doc_cc12):
    ref = generic_metrics.compile_formula("Partner", "partner(m)", "1")[1]
    ref["source_name"] = "CC12.pdf"
    source_series = await generic_metrics._series_for_page(conn, doc_cc12["doc_id"], 3)
    matching = next(s for s in source_series if generic_metrics._matches(s, ref))
    shifted_facts = [generic_metrics.Fact(
        f.fact_id, f.period_key, f.period_label.replace("-26", "-27"), f.period_kind,
        f.period_start.replace("2026-", "2027-"), f.value, f.unit,
    ) for f in matching.facts]
    target = generic_metrics.Series(
        matching.series_id, matching.label, matching.name, matching.entity,
        matching.unit, matching.confidence, matching.path, shifted_facts,
    )
    rule = {"source_name": "CC12.pdf", "source_page": 3}
    doc = {"file_name": "CC13.pdf"}
    result = await generic_metrics._relative_result_periods(
        conn, rule, doc, ref, target, {"Feb-26"}
    )
    assert result == {"Feb-27"}


@integration
@pytest.mark.asyncio
async def test_grouped_row_sum_materializes_each_channel_without_total(conn, doc_cc12):
    operation, spec, unit = generic_metrics.compile_formula(
        "9-month channel totals", "sum rows_channel_mix(m)", "Branch: 4,73,823"
    )
    rule = {
        "source_page": 3, "compiled_spec": json.dumps(spec), "applies_to": "All months",
        "output_type": unit, "rule_id": "test_channel_totals", "metric_name": "9-month channel totals",
        "metric_family": "Volume", "definition": "Σ S(c)", "version": 1,
    }
    rows = await generic_metrics._materialize_rule(conn, rule, doc_cc12)
    assert {row["entity"] for row in rows} == {"Branch + DST", "Online/Digital", "Partner Sourcing"}
    branch = next(row for row in rows if row["entity"] == "Branch + DST")
    assert branch["value"] == 473823


@integration
@pytest.mark.asyncio
async def test_result_period_limits_fixed_window_metrics(conn, doc_cc12):
    operation, spec, unit = generic_metrics.compile_formula(
        "Partner trailing 3-month run-rate",
        "(partner(m) + partner(m-1) + partner(m-2)) / 3", "12,558"
    )
    rule = {
        "source_page": 3, "compiled_spec": json.dumps(spec), "applies_to": "Partner",
        "output_type": "count", "rule_id": "test_partner_run_rate",
        "metric_name": "Partner trailing 3-month run-rate", "metric_family": "Run-rate",
        "definition": "Average of Partner, Dec–Feb", "version": 1,
        "routing_spec": json.dumps({"result_period": "Feb'26"}),
    }
    rows = await generic_metrics._materialize_rule(conn, rule, doc_cc12)
    assert [(row["period_label"], row["value"]) for row in rows] == [("Feb-26", 12558)]


@integration
@pytest.mark.asyncio
async def test_descriptive_ytd_result_period_is_not_shown_as_anchor_quarter(conn, doc_cc12):
    operation, spec, unit = generic_metrics.compile_formula(
        "YTD total", "total(m) + total(m-1)", "1"
    )
    rule = {
        "source_page": 3, "compiled_spec": json.dumps(spec), "applies_to": "",
        "output_type": unit, "rule_id": "test_ytd_display_period",
        "metric_name": "YTD total", "metric_family": "Total",
        "definition": "Current plus previous", "version": 1,
        "routing_spec": json.dumps({"result_period": "YTD Feb (Q1+Q2+Q3+Jan+Feb)"}),
    }
    rows = await generic_metrics._materialize_rule(conn, rule, doc_cc12)
    assert rows
    assert {row["period_label"] for row in rows} == {"YTD"}


@integration
@pytest.mark.asyncio
async def test_example_mismatch_warns_but_keeps_formula_results(conn, doc_cc12):
    operation, spec, unit = generic_metrics.compile_formula("Test total", "total(m)", "Feb-26: 1")
    tx = conn.transaction()
    await tx.start()
    try:
        await conn.execute(
            """INSERT INTO metric_rules
               (rule_id, source_name, source_page, metric_family, metric_name, definition,
                applies_to, example, operation, compiled_spec, aliases, output_type,
                enabled, status, version)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,1,'approved',1)""",
            "test_example_warning", "CC12.pdf", 3, "Test", "Test total", "total(m)",
            "All months", "Feb-26: 1", operation, json.dumps(spec), '["test total"]', unit,
        )
        report = await generic_metrics.materialize(
            conn, doc_ids=[doc_cc12["doc_id"]],
            rule_ids=["test_example_warning"], dry_run=True,
        )
        assert report["results"] == 9
        assert report["failures"] == []
        assert report["warnings"] == [
            "test_example_warning: supplied example did not match the formula result; formula result retained"
        ]
    finally:
        await tx.rollback()


@integration
@pytest.mark.asyncio
async def test_database_rule_aliases_are_available_to_question_router(conn):
    tx = conn.transaction()
    await tx.start()
    try:
        await conn.execute(
            """INSERT INTO metric_rules
               (rule_id, source_name, source_page, metric_family, metric_name, definition,
                applies_to, example, operation, compiled_spec, aliases, output_type,
                enabled, status, version)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,1,'approved',1)
               ON CONFLICT (rule_id) DO NOTHING""",
            "test_rule_alias", "", 18, "Trend", "Customer growth", "A/B-1",
            "Monthly", "", "arithmetic", "{}", '["customer month-on-month growth"]', "%",
        )
        result = await generic_metrics.intended_metrics(
            conn, "What was customer month-on-month growth?", page=18
        )
        assert "test_rule_alias" in result
    finally:
        await tx.rollback()


@integration
@pytest.mark.asyncio
async def test_needs_review_metric_is_discoverable_but_not_executable(conn):
    tx = conn.transaction()
    await tx.start()
    try:
        await conn.execute(
            """INSERT INTO metric_rules
               (rule_id, source_name, source_page, metric_family, metric_name, definition,
                applies_to, example, operation, compiled_spec, aliases, output_type,
                enabled, status, review_reason, version)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,1,'needs_review',$13,1)
               ON CONFLICT (rule_id) DO NOTHING""",
            "test_review_rule", "", 18, "Trend", "Unsafe growth", "A/B-1",
            "Monthly", "", "unsupported", "{}", '["unsafe month-on-month growth"]', "%",
            "formula did not resolve to PDF facts",
        )
        matches = await generic_metrics.reviewed_metric_matches(
            conn, "What was unsafe month-on-month growth?", page=18
        )
        assert [item["metric_id"] for item in matches] == ["test_review_rule"]
        assert "did not resolve" in generic_metrics.reviewed_metric_answer(matches)
        intended = await generic_metrics.intended_metrics(
            conn, "What was unsafe month-on-month growth?", page=18
        )
        assert "test_review_rule" not in intended
    finally:
        await tx.rollback()


@integration
@pytest.mark.asyncio
async def test_generic_intent_group_keeps_the_most_specific_candidate(conn):
    tx = conn.transaction()
    await tx.start()
    try:
        for rule_id, name, aliases, routing in [
            ("test_broad", "YoY growth", '["yoy growth"]',
             '{"intent_group":"growth","priority":1}'),
            ("test_specific", "Like-for-like YoY growth", '["like for like yoy growth"]',
             '{"intent_group":"growth","priority":10}'),
        ]:
            await conn.execute(
                """INSERT INTO metric_rules
                   (rule_id, source_name, source_page, metric_family, metric_name,
                    definition, applies_to, example, operation, compiled_spec,
                    aliases, routing_spec, output_type, enabled, status, version)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,1,'routing_only',1)
                   ON CONFLICT (rule_id) DO NOTHING""",
                rule_id, "", 88, "Growth", name, "", "Monthly", "",
                "arithmetic", "{}", aliases, routing, "%",
            )
        result = await generic_metrics.intended_metrics(
            conn, "What was like-for-like YoY growth?", page=88
        )
        assert result == {"test_specific"}
    finally:
        await tx.rollback()


# ---------------------------------------------------------------------------
# Integration tests — resolve (full pipeline)
# ---------------------------------------------------------------------------

@integration
@pytest.mark.asyncio
async def test_plural_category_question_returns_all_metric_entities(conn):
    rows = await generic_metrics.resolve(
        conn, "What were the 9-month channel totals?",
        doc_ids=["2a7323461ad9438b"], page=3,
    )
    assert {row["entity"] for row in rows} == {"Branch + DST", "Online/Digital", "Partner Sourcing"}


@integration
@pytest.mark.asyncio
async def test_two_period_growth_is_calculated_from_validated_fact_rows(conn):
    rows = await generic_metrics.resolve(
        conn,
        "What is the growth rate for proprietary sourcing from Jun-25 to Aug-25?",
        doc_ids=["2a7323461ad9438b"], page=3,
    )
    assert len(rows) == 1
    assert rows[0]["entity"] == "Proprietary"
    assert rows[0]["comparison_period_label"] == "Jun-25"
    assert rows[0]["period_label"] == "Aug-25"
    assert rows[0]["calculation_status"] == "runtime_validated"
    assert len(json.loads(rows[0]["source_fact_ids"])) == 2
    assert rows[0]["value"] == pytest.approx((33476 / 28184 - 1) * 100)


@integration
@pytest.mark.asyncio
async def test_reviewed_quarterly_metric_uses_declared_validated_base_series(conn):
    rows = await generic_metrics.resolve(
        conn,
        "What was quarterly channel sourcing for Branch + DST in Q2FY26?",
        doc_ids=["2a7323461ad9438b"], page=3,
    )
    assert len(rows) == 1
    assert rows[0]["metric_name"] == "Quarterly channel sourcing"
    assert rows[0]["entity"] == "Branch + DST"
    assert rows[0]["period_label"] == "Q2FY26"
    assert rows[0]["value"] == 42438 + 38728 + 45697
    assert rows[0]["calculation_status"] == "runtime_validated"
    assert len(json.loads(rows[0]["source_fact_ids"])) == 3


@integration
@pytest.mark.asyncio
async def test_longer_named_entity_wins_over_its_substring(conn):
    rows = await generic_metrics.resolve(
        conn, "What share of ENR was Non-Term in Feb'26?",
        doc_ids=["2a7323461ad9438b"], page=None,
    )
    assert [(row["entity"], row["formatted_value"]) for row in rows] == [("Non-Term", "56.00%")]


@integration
@pytest.mark.asyncio
async def test_lowest_entity_value_is_selected_deterministically_from_fact_series(conn):
    rows = await generic_metrics.resolve(
        conn, "In which month was CIF lowest?",
        doc_ids=["2a7323461ad9438b"], page=None,
    )
    assert [(row["entity"], row["period_label"], row["value"]) for row in rows] == [
        ("CIF", "Aug-25", 4541171.0)
    ]
    assert generic_metrics.deterministic_answer(rows) == (
        "The lowest Month-end CIF was 4,541,171.00 in Aug-25. (p.4)"
    )


@integration
@pytest.mark.asyncio
async def test_highest_registered_metric_is_selected_without_model_comparison(conn):
    rows = await generic_metrics.resolve(
        conn, "In which month was the active rate highest?",
        doc_ids=["2a7323461ad9438b"], page=None,
    )
    assert [(row["entity"], row["period_label"], row["value"]) for row in rows] == [
        ("Active Rate", "Oct-25", 50.0)
    ]


@integration
@pytest.mark.asyncio
async def test_lowest_product_entity_uses_the_same_generic_extremum_path(conn):
    rows = await generic_metrics.resolve(
        conn, "Which month had the lowest Xpress Cash disbursal?",
        doc_ids=["2a7323461ad9438b"], page=None,
    )
    assert [(row["entity"], row["period_label"], row["value"]) for row in rows] == [
        ("Xpress Cash", "Feb-26", 57.0)
    ]


@integration
@pytest.mark.asyncio
async def test_grouped_period_comparison_sums_windows_from_monthly_facts(conn):
    rows = await generic_metrics.resolve(
        conn, "How does net addition compare between H1 and Oct–Feb?",
        doc_ids=["2a7323461ad9438b"], page=None,
    )
    assert [(row["period_label"], row["value"]) for row in rows] == [
        ("H1", -242916.0), ("Oct–Feb", 23617.0)
    ]
    assert json.loads(rows[0]["inputs_json"]) == {
        "Apr-25": -51976.0, "May-25": -40842.0, "Jun-25": -47329.0,
        "Jul-25": -41061.0, "Aug-25": -105727.0, "Sep-25": 44019.0,
    }
    assert generic_metrics.deterministic_answer(rows).endswith(
        "This represents a shift from negative to positive. (p.4)"
    )


@integration
@pytest.mark.asyncio
async def test_passive_write_off_question_resolves_source_fact_and_displays_magnitude(conn):
    rows = await generic_metrics.resolve(
        conn, "How much was written off in Q3 FY26?",
        doc_ids=["2a7323461ad9438b"], page=None,
    )
    assert [(row["entity"], row["period_label"], row["value"]) for row in rows] == [
        ("Write Off", "Q3FY26", 663.0)
    ]
    assert "reported as -663.00 Cr in the source walk" in rows[0]["answer"]


# ---------------------------------------------------------------------------
# Integration tests — resolve_direct_fact and unscoped document routing
# ---------------------------------------------------------------------------

@integration
@pytest.mark.asyncio
async def test_unscoped_rfl_recovery_rate_uses_exact_source_fact(conn):
    rows = await generic_metrics.resolve(conn, "What is RFL's NPA recovery rate?")

    assert len(rows) == 1
    assert rows[0]["metric_name"] == "NPA / Recovery Rate %"
    assert rows[0]["entity"] == "RFL"
    assert rows[0]["value"] == pytest.approx(0.82)
    assert rows[0]["source_name"] == "08_SMA_and_Collection_Efficiency_5659.pdf"


@integration
@pytest.mark.asyncio
async def test_unscoped_june_total_advances_does_not_use_generic_total_fallback(conn):
    rows = await generic_metrics.resolve(conn, "What were total advances in June 2026?")

    assert len(rows) == 1
    assert rows[0]["entity"] == "Grand Total"
    assert rows[0]["period_label"] == "Jun-26"
    assert rows[0]["value"] == 116226
    assert rows[0]["source_name"] == "09_Net_Advances_Insights_7222.pdf"


@integration
@pytest.mark.asyncio
async def test_exact_catalog_metric_remains_authoritative_over_source_fact_search(conn):
    rows = await generic_metrics.resolve(conn, "What is the INSIGNIA grand total?")

    assert len(rows) == 1
    assert rows[0]["entity"] == "INSIGNIA › Grand Total"
    assert rows[0]["value"] == pytest.approx(68026.83)


@integration
@pytest.mark.asyncio
async def test_natural_entity_plus_calculation_name_resolves_scoped_metric(conn):
    rows = await generic_metrics.resolve(
        conn, "What is Affordable Housing QoQ growth in Jun-26?"
    )

    assert len(rows) == 1
    assert rows[0]["metric_name"] == "Product QoQ growth"
    assert rows[0]["entity"] == "Affordable Housing"
    assert rows[0]["value"] == pytest.approx(8.40918942349371)


@integration
@pytest.mark.asyncio
async def test_scenario_letter_is_interpreted_only_as_a_complete_path_component(conn):
    rows = await generic_metrics.resolve(
        conn, "What was Wholesale Banking CA average balance in Q2 FY27 Budget?"
    )

    assert len(rows) == 1
    assert rows[0]["metric_name"] == "Budget CA AVG"
    assert rows[0]["period_label"] == "Q2FY27"
    assert rows[0]["value"] == 8165


@integration
@pytest.mark.asyncio
async def test_metric_phrase_wins_over_metric_words_used_in_report_scope(conn):
    rows = await generic_metrics.resolve(
        conn, "In the Retail Disbursement MIS report, what is ATS change?"
    )

    assert len(rows) == 1
    assert rows[0]["metric_id"] == "g_271980e6ab_015"
    assert rows[0]["formatted_value"] == "GT -0.7%; LAP -9.7%"


@integration
@pytest.mark.asyncio
async def test_report_date_does_not_override_an_explicit_metric_name(conn):
    rows = await generic_metrics.resolve(
        conn, "In the Agri January 2026 review report, what is ANR (Agri) — % YOY?"
    )

    assert len(rows) == 1
    assert rows[0]["metric_id"] == "g_5de3209b9c_026"
    assert rows[0]["formatted_value"] == "16.20%"


@integration
@pytest.mark.asyncio
async def test_explicit_overall_advances_report_name_beats_shared_advance_vocabulary(conn):
    rows = await generic_metrics.resolve(
        conn, "In the Overall Advances report, what is 3-month growth?"
    )

    assert len(rows) == 1
    assert rows[0]["metric_id"] == "g_cb7b350655_008"
    assert rows[0]["doc_id"] == "2f7b0e39c77a7c6e"
