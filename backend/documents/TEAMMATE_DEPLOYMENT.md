# Complete ingestion and business-rule update

This archive is intended to be extracted over an existing copy of the same project.

## Install

1. Stop the Streamlit application and any process using `data/library.db`.
2. Back up the teammate's existing `data/library.db` and `data/docs/` if they contain local
   changes. This archive contains the complete validated database and readable document-cache
   folder names, so those paths must be replaced together.
3. Extract this ZIP at the project root, preserving paths and replacing existing files.
4. Start the application normally with `./run.sh`.

## Verify

```bash
.venv/bin/python -m pytest -q tests/test_business_examples.py tests/test_ingestion_robustness.py
sqlite3 data/library.db "SELECT COUNT(*) FROM metric_rules WHERE status='needs_review';"
sqlite3 data/library.db "SELECT COUNT(*) FROM facts WHERE doc_id='fa2376ada2b40e80';"
sqlite3 data/library.db "SELECT COUNT(*) FROM metric_rules WHERE json_extract(routing_spec, '$.source_workbook') IN ('11_Overall_Advances_Metrics.xlsx','12_Liability_Summary_Metrics.xlsx','13_Collections_daily_refresh_Metrics.xlsx','14_Retail_Collections_Efficiency_Metrics.xlsx');"
```

Expected results: the selected tests pass, `needs_review` is `0`, the WB fact count is `600`,
and the final query returns `158`.

## Import future business-rule workbooks

Preview, then apply after formal approval:

```bash
./run.sh business-rules /path/to/New_Metrics.xlsx --approve-examples
./run.sh business-rules /path/to/New_Metrics.xlsx --apply --approve-examples
```

Use `--pdf /path/to/New_Report.pdf` on the apply command when the source PDF is not yet ingested.

## Included behavior

- All 787 formerly reviewed rules return their formally approved Excel/TSV outputs.
- The audit also repaired 66 older approved CC12 rules that had no materialized result.
- `business_metrics_source/` contains all source workbooks used for provenance, including the four
  newly supplied workbooks for Overall Advances, Liability Summary, Collections Daily Refresh,
  and Retail Collections Efficiency.
- All 158 newly supplied rules resolve: 145 approved workbook-defined outputs and 13 dynamic
  calculations.
- The matching four PDFs, parsed document caches, and database records are included and kept in
  sync in this package.
- Document-cache folders use readable `<pdf-name>__<hash>` names without sacrificing hash-based
  uniqueness.
- The WB Balance Sheet is treated as native tables rather than 19 false charts. All six columns,
  including `Julʼ26 P`, are ingested: 300 series and 600 validated facts.
- Fresh ingestion removes stale chart-crop PNGs only inside the document being rebuilt.

The source workbooks are copied artifacts. The originals outside the project were not modified.
