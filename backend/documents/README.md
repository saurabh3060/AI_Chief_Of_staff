# Financial PDF Q&A — Database-driven calculations

The application uses one database-driven calculation engine for every page and every PDF. Business
calculation tables are imported as rules; no page-specific Python materializer is required.

## Included data

- `MFI.pdf` and `CC12.pdf` in `data/uploads/`
- Their existing OCR/extraction caches and page images in readable
  `data/docs/<pdf-name>__<hash>/` folders
- Their populated SQLite fact store and embeddings in `data/library.db`
- The active, database-driven calculation catalogues for CC12

The included reports do **not** need to be processed again. The retired MFI page-7 and page-14
calculation assets are preserved under `archive/mfi_page7_page14/` and are not loaded at runtime.
Month, quarter and fiscal-year windows are discovered from each report's stored fact metadata;
they are not tied to the dates in the included sample reports.

## Start the application

```bash
cd /Users/j.verma/Documents/rethink_det
./run.sh
```

The launcher starts the bundled local model server when necessary, then opens Streamlit at
`http://localhost:8501`.

Useful check:

```bash
./run.sh list
```

## Database-driven calculations

Import the business six-column Excel/TSV
table into the shared calculation registry, review the dry run, and apply it:

```bash
./run.sh calculations import page18_metrics.xlsx --page 18 --dry-run
./run.sh calculations import page18_metrics.xlsx --page 18 --apply
```

Clipboard/tab-separated input is also supported on macOS:

```bash
pbpaste | ./run.sh calculations import - --page 18 --dry-run
```

Definitions that compile to the safe generic expression language are stored in `metric_rules` and
materialized for every compatible PDF. Numeric example mismatches are reported as review issues but
do not suppress a valid formula result. Ambiguous or unsupported formulas are marked `needs_review`;
the engine never executes arbitrary Excel/Python expressions. Inspect or rerun the registry with:

Optional routing/answer columns can be added to the same workbook: `Aliases`, `Required terms`,
`Required any`, `Excluded terms`, `Period scope`, `Priority`, `Intent group`, `Answer template`,
`Success condition`, `True text`, and `False text`. Phrase lists use `;` as the separator. Success
conditions accept only safe comparisons such as `value >= 1`; they are not evaluated as Python.

```bash
./run.sh calculations list
./run.sh calculations materialize --all
```

After formal business sign-off, generated definitions that cannot be represented by the generic
formula language can retain their exact approved Excel output as a traceable, read-only result:

```bash
./run.sh calculations approve-business-examples          # preview
./run.sh calculations approve-business-examples --apply  # persist and materialize
```

These rules use the `business_example` operation. Their original compiler state and review reason
are retained in `compiled_spec`, and the answer records the source workbook. They do not pretend
that unsupported free-form notation was independently recomputed by the PDF formula engine.

## One-command business-rule pipeline

For a new workbook in the same format, preview it first:

```bash
./run.sh business-rules /path/to/New_Metrics.xlsx --approve-examples
```

After checking the preview and confirming formal business approval, apply it:

```bash
./run.sh business-rules /path/to/New_Metrics.xlsx --apply --approve-examples
```

For a PDF that is not yet in the document library, ingest it in the same run:

```bash
./run.sh business-rules /path/to/New_Metrics.xlsx \
  --pdf /path/to/New_Report.pdf --apply --approve-examples
```

The workbook `README` must contain `Source: New_Report.pdf`. Use `Source page: 3` for one source
page, or add a `Page` column to the metric table when rows target different pages. The pipeline
backs up `data/library.db`, copies the workbook into `business_metrics_source/`, validates dynamic
formulas, scopes the approved-output fallback to only the imported rule IDs, materializes results,
and verifies that every imported metric resolves by name. Without `--apply` it is always read-only.

## Add a new PDF

From the terminal:

```bash
./run.sh ingest /path/to/new-report.pdf
```

Or use **Documents → Add a PDF** in the application. New PDFs go through extraction/OCR and are
added to this project's own database. Use `--no-vision` only when charts and pictured tables can be
skipped.

OCR now uses the separate docTR adapter in `ocr_doctr.py`. RapidOCR/Paddle is inactive and retained
in `ocr_rapidocr.py` only for rollback. The commented import beside the active docTR import at the
top of `gridocr.py` shows the one-line switch.

The extraction cache records its OCR backend, so results produced by different backends are not
silently reused together.

## Capability boundary

- Every calculation must be represented by an approved database rule.
- The safe expression engine supports fact references, period offsets, arithmetic and standard
  aggregates. Unsupported definitions are marked `needs_review` instead of running guessed logic.
- New reusable operators are added once to `generic_metrics.py`, never as page-specific modules.

## Clean project contents

The project contains runtime application code, ingestion/OCR code, the financial tools, required
models, the local model server, the Python environment, included document data, and the active
calculation validation suite. Retired MFI page-specific calculation assets are isolated under
`archive/` and are not imported by the application.
