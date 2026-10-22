"""One-command ingestion pipeline for business metric workbooks.

Dry run (default):
  python -m documents.business_rules_pipeline rules.xlsx

Apply dynamically validated rules:
  python -m documents.business_rules_pipeline rules.zip --apply

Apply and preserve formally approved Excel outputs for unsupported definitions:
  python -m documents.business_rules_pipeline rules.zip --apply --approve-examples

NOTE: the SQLite ``.backup()`` step is gone — PostgreSQL has its own backup tooling
(``pg_dump``). Take a dump externally if you need a rollback point before --apply.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from . import generic_metrics
from . import import_metric_zip
from . import store


WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}


def _package(input_path: Path, temp_dir: Path) -> Path:
    if input_path.suffix.lower() == ".zip":
        return input_path
    if input_path.suffix.lower() not in WORKBOOK_SUFFIXES:
        raise ValueError("input must be an .xlsx, .xlsm, or .zip file")
    output = temp_dir / "business_rules.zip"
    from zipfile import ZIP_DEFLATED
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.write(input_path, input_path.name)
    return output


def _copy_workbooks(package: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    seen: set[str] = set()
    with ZipFile(package) as archive:
        for name in archive.namelist():
            source = Path(name)
            if source.suffix.lower() not in WORKBOOK_SUFFIXES or source.name.startswith("~$"):
                continue
            if source.name in seen:
                raise ValueError(f"ZIP contains duplicate workbook filename: {source.name}")
            seen.add(source.name)
            target = destination / source.name
            with archive.open(name) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            copied.append(str(target))
    return copied


def _print_plan(sheets: list[dict], rules: list[dict], approve_examples: bool) -> None:
    for item in sheets:
        target = item["source"] or f"UNRESOLVED ({item['expected_source']})"
        pages = ",".join(str(page) for page in item["pages"])
        print(f"{item['workbook']} :: {item['sheet']} -> {target} p.{pages} | "
              f"{item['approved']}/{item['rules']} dynamically approved, "
              f"{item['needs_review']} need review")
    counts = {"approved": 0, "needs_review": 0}
    for rule in rules:
        counts[rule["status"]] += 1
    print(json.dumps({"rules": len(rules), **counts,
                      "approved_example_fallback_requested": approve_examples}, indent=2))


async def _verify(rule_ids: list[str]) -> dict:
    async with store.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM metric_rules WHERE rule_id = ANY($1::text[]) ORDER BY rule_id",
            rule_ids,
        )
        documents = {row["file_name"]: row["doc_id"] for row in await conn.fetch(
            "SELECT file_name,doc_id FROM documents"
        )}
        failures = []
        for rule in rows:
            result_count = await conn.fetchval(
                "SELECT COUNT(*) FROM calculated_metrics "
                "WHERE metric_id = $1 AND calculation_version = $2",
                rule["rule_id"], rule["version"],
            )
            if rule["status"] != "approved":
                failures.append(f"{rule['rule_id']}: status={rule['status']}")
                continue
            if result_count == 0:
                failures.append(f"{rule['rule_id']}: no current materialized result")
                continue
            resolved = await generic_metrics.resolve(
                conn, rule["metric_name"],
                doc_ids=[documents[rule["source_name"]]],
                page=rule["source_page"], limit=100,
            )
            if rule["rule_id"] not in {row["metric_id"] for row in resolved}:
                failures.append(f"{rule['rule_id']}: metric name did not resolve to its result")
        return {"rules": len(rows), "ready": len(rows) - len(failures), "failures": failures}


async def _run(args) -> int:
    if args.pdf:
        from . import library
        for pdf in args.pdf:
            if not pdf.is_file():
                raise ValueError(f"PDF does not exist: {pdf}")
            print(f"Ingesting {pdf} ...", flush=True)
            await library.add_pdf(str(pdf), use_vision=not args.no_vision)

    with tempfile.TemporaryDirectory(prefix="business_rules_pipeline_") as temp:
        package = _package(args.input, Path(temp))
        sheets, rules = await import_metric_zip.inspect_zip(package)
        await import_metric_zip.validate_pdf_bindings(sheets, rules)
        _print_plan(sheets, rules, args.approve_examples)
        if args.verbose:
            for rule in rules:
                if rule["status"] != "approved":
                    print(f"REVIEW {rule['rule_id']} {rule['metric_name']}: {rule['error']}")
        unresolved = sorted({item["expected_source"] for item in sheets if not item["source"]})
        if unresolved:
            raise ValueError("source PDFs are not ingested: " + ", ".join(unresolved))
        if not args.apply:
            print("Dry run only. Re-run with --apply after checking this report.")
            return 0

        copied = _copy_workbooks(package, Path("business_metrics_source"))
        async with store.acquire() as conn:
            async with conn.transaction():
                await generic_metrics.save_rules(rules, conn=conn)
        rule_ids = [rule["rule_id"] for rule in rules]
        dynamic = await generic_metrics.materialize(rule_ids=rule_ids)
        fallback = None
        if args.approve_examples:
            fallback = await generic_metrics.approve_business_examples(
                apply=True, rule_ids=rule_ids,
            )
        verification = await _verify(rule_ids)
        print(json.dumps({
            "copied_workbooks": copied,
            "dynamic_materialized_results": dynamic["results"],
            "business_example_fallback": fallback, "verification": verification,
        }, indent=2))
        if dynamic["failures"] or verification["failures"]:
            return 1
        print("READY: every imported rule is approved, materialized, and resolves by name.")
        return 0


def main() -> int:
    from core.db import init_db, close_db

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one business workbook or a ZIP of workbooks")
    parser.add_argument("--pdf", type=Path, action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approve-examples", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.pdf and not args.apply:
        parser.error("--pdf requires --apply because PDF ingestion writes data")

    async def runner():
        await init_db()
        try:
            return await _run(args)
        finally:
            await close_db()

    try:
        return asyncio.run(runner())
    except (BadZipFile, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
