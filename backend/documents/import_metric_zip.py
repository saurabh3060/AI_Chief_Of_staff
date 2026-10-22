"""Import business calculation sheets from a ZIP of Excel workbooks.

Only sheets ending in ``_Metrics`` are read.  The source PDF is taken from each workbook's
README sheet, resolved to the exact filename already present in the document library, and bound
to PDF page 1.  The importer is a dry run unless ``--apply`` is supplied.

Usage:
  python import_metric_zip.py /path/to/business_metrics.zip
  python import_metric_zip.py /path/to/business_metrics.zip --apply
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook

from . import generic_metrics
from . import store


REQUIRED_COLUMNS = {"number", "family", "name", "formula", "applies", "example"}
SOURCE_RE = re.compile(r"(?i)\bsource\s*:\s*([^|\n]+?\.pdf)\b")
PAGE_COUNT_RE = re.compile(r"(?i)\((\d+)\s+pages?\b")
SOURCE_PAGE_RE = re.compile(r"(?im)\bsource\s+page\s*:\s*(\d+)\b")


def _filename_key(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"^\d+[\s_-]+", "", stem)  # delivery order, e.g. 11_Overall_Advances
    stem = re.sub(r"_\d{4}$", "", stem)  # delivery suffix, e.g. _4786
    key = re.sub(r"[^a-z0-9]+", "", stem)
    # Preserve compatibility with the already-ingested business filename's known spelling typo.
    return key.replace("liablity", "liability")


def _readme_text(workbook) -> str:
    if "README" not in workbook.sheetnames:
        raise ValueError("README sheet is missing")
    return "\n".join(
        " | ".join(str(value).strip() for value in row if value is not None and str(value).strip())
        for row in workbook["README"].iter_rows(values_only=True)
    )


def _source_metadata(workbook) -> tuple[str, int | None]:
    text = _readme_text(workbook)
    match = SOURCE_RE.search(text)
    if not match:
        raise ValueError("README does not contain 'Source: <file>.pdf'")
    source_page = SOURCE_PAGE_RE.search(text)
    if source_page:
        return Path(match.group(1).strip()).name, int(source_page.group(1))
    page_match = PAGE_COUNT_RE.search(text)
    data_pages = int(page_match.group(1)) if page_match else 1
    # Multi-page workbooks may instead provide a Page column in each metric table.
    return Path(match.group(1).strip()).name, 1 if data_pages == 1 else None


async def _library_documents() -> list[str]:
    async with store.acquire() as conn:
        return [row[0] for row in await conn.fetch("SELECT file_name FROM documents")]


def _declared_page_count(workbook) -> int | None:
    """Return the page count advertised by the workbook, when present."""
    match = PAGE_COUNT_RE.search(_readme_text(workbook))
    return int(match.group(1)) if match else None


def _resolve_source(expected: str, documents: list[str]) -> str | None:
    key = _filename_key(expected)
    matches = sorted({name for name in documents if _filename_key(name) == key})
    # Prefer a known recurring-report family over a loose filename token. For example WB.pdf
    # means the Wholesale Banking review, not an unrelated legacy file which happens to contain
    # the token "WB" in its filename.
    if not matches:
        family = generic_metrics.document_family(expected)
        family_matches = {
            name for name in documents
            if generic_metrics.document_family(name) == family
        }
        if family_matches:
            matches = sorted(family_matches)
    # Business catalogues often use the short portfolio name (for example RVF.pdf) while the
    # ingested file keeps the dated review-deck name. Accept that shorthand only when it is a
    # standalone token and resolves to exactly one library document; never guess between decks.
    if not matches and key:
        token_matches = {
            name for name in documents
            if key in {
                re.sub(r"[^a-z0-9]+", "", token)
                for token in re.findall(r"[A-Za-z0-9]+", Path(name).stem.lower())
            }
        }
        matches = sorted(token_matches)
    if len(matches) > 1:
        raise ValueError(f"source {expected!r} matches multiple ingested PDFs: {', '.join(matches)}")
    return matches[0] if matches else None


def _header_row(sheet) -> tuple[int, list[str]]:
    for index, row in enumerate(sheet.iter_rows(values_only=True), 1):
        headers = ["" if value is None else str(value).strip() for value in row]
        if REQUIRED_COLUMNS <= set(generic_metrics._column_map(headers)):
            return index, headers
    raise ValueError("six-column metric table header was not found")


def _sheet_csv(formula_sheet, value_sheet) -> str:
    header_at, headers = _header_row(formula_sheet)
    column_map = generic_metrics._column_map(headers)
    example_index = headers.index(column_map["example"])
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    # ``ReadOnlyWorksheet.cell()`` streams from the start of the sheet for every lookup.
    # Calling it once per cell makes large business catalogues effectively quadratic. Walk the
    # formula and cached-value sheets in lockstep so each worksheet XML stream is read once.
    formula_rows = formula_sheet.iter_rows(
        min_row=header_at + 1, max_col=len(headers), values_only=True,
    )
    value_rows = value_sheet.iter_rows(
        min_row=header_at + 1, max_col=len(headers), values_only=True,
    )
    for formula_values, cached_values in zip(formula_rows, value_rows):
        values = list(formula_values)
        cached = cached_values[example_index]
        if cached is not None:
            values[example_index] = cached
        writer.writerow("" if value is None else value for value in values)
    return stream.getvalue()


def _compile_sheet(formula_sheet, value_sheet, *, source: str, page: int,
                   workbook_name: str) -> list[dict]:
    text = _sheet_csv(formula_sheet, value_sheet)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", newline="", delete=False) as tmp:
        tmp.write(text)
        temp_path = Path(tmp.name)
    try:
        rules = generic_metrics.import_rows(
            str(temp_path), default_page=page, source_name=source, apply=False,
            rule_namespace=f"{source}|{workbook_name}|{formula_sheet.title}",
        )
        for rule in rules:
            rule["routing_spec"]["metric_sheet"] = formula_sheet.title
            rule["routing_spec"]["source_workbook"] = Path(workbook_name).name
        return rules
    finally:
        temp_path.unlink(missing_ok=True)


async def inspect_zip(zip_path: Path) -> tuple[list[dict], list[dict]]:
    documents = await _library_documents()
    async with store.acquire() as conn:
        library_page_counts = {
            row["file_name"]: int(row["pages"] or 0)
            for row in await conn.fetch("SELECT file_name, pages FROM documents")
        }
    sheets, rules = [], []
    with ZipFile(zip_path) as archive:
        workbook_names = sorted(name for name in archive.namelist()
                                if name.lower().endswith((".xlsx", ".xlsm")) and not name.startswith("__MACOSX/"))
        if not workbook_names:
            raise ValueError("ZIP contains no .xlsx or .xlsm workbooks")
        for workbook_name in workbook_names:
            payload = archive.read(workbook_name)
            formulas = load_workbook(io.BytesIO(payload), read_only=True, data_only=False)
            values = load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
            expected, page = _source_metadata(formulas)
            source = _resolve_source(expected, documents)
            declared_pages = _declared_page_count(formulas)
            metric_sheets = [name for name in formulas.sheetnames if name.lower().endswith("_metrics")]
            if not metric_sheets:
                raise ValueError(f"{workbook_name}: no *_Metrics sheet found")
            for sheet_name in metric_sheets:
                populated_rows = sum(
                    1 for row in formulas[sheet_name].iter_rows(values_only=True)
                    if any(value is not None and str(value).strip() for value in row)
                )
                # Business templates keep paired page/metric tabs even when a page has no
                # calculations. A header-only _Metrics tab is an intentional empty section,
                # not a malformed workbook, and must not block all populated sheets.
                if populated_rows <= 1:
                    sheets.append({
                        "workbook": Path(workbook_name).name,
                        "sheet": sheet_name,
                        "expected_source": expected,
                        "source": source,
                        "pages": [],
                        "rules": 0,
                        "approved": 0,
                        "needs_review": 0,
                    })
                    continue
                compiled = _compile_sheet(
                    formulas[sheet_name], values[sheet_name], source=source or expected, page=page,
                    workbook_name=Path(workbook_name).name,
                )
                # The WB rules were authored against a PDF export containing a cover page that
                # is not present in the ingested review deck. Infer that stable export/deck
                # difference from the declared and actual page counts instead of tying rules to
                # a dated filename. This keeps the corrected page routing valid for later decks
                # with the same layout while avoiding a shift when the source already has all
                # advertised pages.
                if (source and declared_pages and
                        generic_metrics.document_family(source) == "wholesale_banking_review" and
                        declared_pages == library_page_counts.get(source, 0) + 1):
                    for rule in compiled:
                        if rule["source_page"] > 1:
                            rule["source_page"] -= 1
                # The corrected Mortgages catalogue labels its Total LAP+HL P&L tab P12, while
                # the supplied review deck contains that table on slide 13 (slide 12 is RFL).
                # Bind by stable family and sheet identity so a dated replacement deck with the
                # same layout continues to use the actual Total LAP+HL table.
                if (source and generic_metrics.document_family(source) == "mortgages_lago_review"
                        and sheet_name == "P12_Metrics"
                        and library_page_counts.get(source, 0) >= 13):
                    for rule in compiled:
                        if rule["source_page"] == 12:
                            rule["source_page"] = 13
                sheets.append({
                    "workbook": Path(workbook_name).name,
                    "sheet": sheet_name,
                    "expected_source": expected,
                    "source": source,
                    "pages": sorted({rule["source_page"] for rule in compiled}),
                    "rules": len(compiled),
                    "approved": sum(rule["status"] == "approved" for rule in compiled),
                    "needs_review": sum(rule["status"] != "approved" for rule in compiled),
                })
                rules.extend(compiled)
    return sheets, rules


async def validate_pdf_bindings(sheets: list[dict], rules: list[dict]) -> None:
    """Approve only rules proven against facts extracted from their named PDF."""
    async with store.acquire() as conn:
        documents = {row["file_name"]: row for row in await conn.fetch(
            "SELECT * FROM documents WHERE status IN ('ready','indexing')"
        )}
        for rule in rules:
            if rule["status"] != "approved":
                continue
            document = documents.get(rule["source_name"])
            if not document:
                rule["status"] = "needs_review"
                rule["error"] = "source PDF is not ingested"
                continue
            routing = rule["routing_spec"]
            if not routing.get("result_period"):
                text = f"{rule['metric_name']} {rule['applies_to']}"
                month = re.search(
                    r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                    text,
                )
                if month and ("only" in text.lower() or re.search(r"\([^)]*" + re.escape(month.group(1)) + r"[^)]*\)", rule["metric_name"], re.I)):
                    prefix = month.group(1)[:3].title() + "-"
                    available = [row[0] for row in await conn.fetch(
                        "SELECT DISTINCT period_label, period_end FROM facts "
                        "WHERE doc_id=$1 AND page=$2 AND period_kind='M' "
                        "AND period_label LIKE $3 ORDER BY period_end DESC",
                        document["doc_id"], rule["source_page"], prefix + "%",
                    )]
                    if available:
                        routing["result_period"] = available[0]
            candidate = dict(rule)
            candidate["compiled_spec"] = json.dumps(rule["compiled_spec"])
            candidate["routing_spec"] = json.dumps(rule["routing_spec"])
            candidate["version"] = 1
            try:
                rows = await generic_metrics._materialize_rule(conn, candidate, document)
            except Exception as exc:
                rule["status"] = "needs_review"
                rule["error"] = f"PDF binding failed: {exc}"
                continue
            if not rows:
                rule["status"] = "needs_review"
                rule["error"] = "formula did not resolve to PDF facts"
                continue
            example_ok = generic_metrics._validate_example(candidate, rows)
            if example_ok is not True:
                rule["status"] = "needs_review"
                rule["error"] = ("PDF result does not match the supplied numeric example"
                                 if example_ok is False else
                                 "no numeric example is available to validate the PDF binding")
        for sheet in sheets:
            selected = [rule for rule in rules
                        if rule["source_name"] == (sheet["source"] or sheet["expected_source"])
                        and rule["routing_spec"].get("metric_sheet") == sheet["sheet"]]
            sheet["approved"] = sum(rule["status"] == "approved" for rule in selected)
            sheet["needs_review"] = len(selected) - sheet["approved"]


def main() -> int:
    import asyncio
    from core.db import init_db, close_db

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_file", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.zip_file.is_file():
        parser.error(f"ZIP file does not exist: {args.zip_file}")

    async def runner():
        await init_db()
        try:
            sheets, rules = await inspect_zip(args.zip_file)
            await validate_pdf_bindings(sheets, rules)
            print("MODE:", "APPLY" if args.apply else "DRY RUN (nothing written)")
            for item in sheets:
                target = item["source"] or f"UNRESOLVED ({item['expected_source']})"
                pages = ",".join(str(page) for page in item["pages"])
                print(f"{item['workbook']} :: {item['sheet']} -> {target} p.{pages} | "
                      f"{item['approved']}/{item['rules']} approved, {item['needs_review']} needs review")
            unresolved = [item for item in sheets if not item["source"]]
            if args.verbose:
                for rule in rules:
                    if rule["status"] != "approved":
                        print(f"REVIEW {rule['rule_id']} {rule['metric_name']}: {rule['error']}")
            if not args.apply:
                print(f"\nDry run complete: {len(rules)} rules found.")
                if unresolved:
                    print("Ingest the unresolved source PDFs before using --apply.")
                return 0
            if unresolved:
                names = ", ".join(sorted({item["expected_source"] for item in unresolved}))
                raise ValueError(f"source PDFs are not ingested: {names}")
            async with store.acquire() as conn:
                async with conn.transaction():
                    await generic_metrics.save_rules(rules, conn=conn)
            report = await generic_metrics.materialize(rule_ids=[rule["rule_id"] for rule in rules])
            print(f"\nImported {len(rules)} rules; materialized {report['results']} results "
                  f"across {report['documents']} documents.")
            if report["warnings"]:
                print(f"Warnings: {len(report['warnings'])}")
            if report["failures"]:
                print("Failures:")
                for failure in report["failures"]:
                    print(" -", failure)
                return 1
            return 0
        finally:
            await close_db()

    try:
        return asyncio.run(runner())
    except (BadZipFile, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
