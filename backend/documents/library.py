"""Document library: add PDFs/PPTX files, load their extraction into PostgreSQL, list them.

Usage:
  python -m documents.library add input/CC12.pdf [--no-vision] [--fresh]
  python -m documents.library add input/Review.pptx [--no-vision] [--fresh]
  python -m documents.library import data/pages.json --pdf input/CC12.pdf
  python -m documents.library reload
  python -m documents.library list

PDF extraction (``ingest.ingest``) is unchanged.  PPTX files are
automatically converted to PDF via LibreOffice before extraction.
Row data goes to PostgreSQL via ``documents/store.py``; block + series
embeddings go to Weaviate via ``documents/weaviate_store.py``.
No SQLite anywhere.
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from . import config
from . import facts
from . import generic_metrics
from . import llm
from . import periods
from . import store

DOC_TYPES = [  # first match on the opening pages wins
    ("annual_report", r"annual report|integrated report"),
    ("basel_disclosure", r"basel|pillar\s*3|pillar iii"),
    ("investor_presentation", r"investor presentation|earnings (presentation|call)|analyst (meet|presentation)"),
    ("financial_results", r"financial results|results for the quarter|audited|unaudited"),
    ("business_update", r"business update|\bmis\b|dashboard|review"),
]

_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def head_tokens(text: str, budget: int) -> str:
    """The start of text within about ``budget`` nomic tokens."""
    for n, m in enumerate(_PIECE.finditer(text)):
        if n == budget:
            return text[:m.start()]
    return text


def document_folder(file_name: str, doc_id: str) -> Path:
    """Readable, cross-platform cache folder with the content hash retained for uniqueness."""
    stem = Path(file_name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .") or "document"
    return config.DOC_IMG_DIR / f"{stem}__{doc_id}"


@asynccontextmanager
async def doc_paths(doc_id: str, file_name: str):
    """Point ingest.py's cache and page images at this document's own folder."""
    folder = document_folder(file_name, doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    saved = config.PAGES_JSON, config.PAGE_IMG_DIR
    config.PAGES_JSON, config.PAGE_IMG_DIR = folder / "pages.json", folder / "pages"
    try:
        yield folder
    finally:
        config.PAGES_JSON, config.PAGE_IMG_DIR = saved


def describe(pages: list[dict], file_name: str) -> dict:
    """Title, type and as-of period from the opening pages."""
    first = [p for p in pages[:3] if p.get("text")]
    lines = [ln.strip() for p in first[:1] for ln in p["text"].splitlines() if ln.strip()]
    title = " — ".join(lines[:2])[:120] or Path(file_name).stem
    opening = " ".join(p["text"] for p in first)
    doc_type = next((t for t, pat in DOC_TYPES if re.search(pat, opening, re.I)), "other")
    found = [p for p, _ in periods.find_all(" ".join(lines[:6]))]
    if not found:
        header_text = " ".join(
            str(cell)
            for page in pages[:3] if not page.get("exclude_from_index")
            for table in page.get("tables", [])
            for cell in table.get("header", [])
        )
        found = [p for p, _ in periods.find_all(header_text)]
    as_of = max(found, key=lambda p: p.end) if found else None
    return {"title": title, "doc_type": doc_type, "as_of": as_of}


def embed_text(b: dict) -> str:
    """What a block's vector is made from."""
    c = b.get("content") or {}
    if b["kind"] not in ("table", "chart") or not c.get("rows"):
        return b["text"]
    context = b["text"].split("\n", 1)[0]
    cols = [h for h in c.get("header", []) if h]
    rows = [r[0] for r in c["rows"] if r and r[0]]
    return "\n".join(x for x in (context, b.get("title", ""), "Columns: " + " | ".join(cols),
                                 "Rows: " + " | ".join(rows), b["text"]) if x)


def _embed_all(texts: list[str], prefix: str, batch: int = 32) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        out += llm.embed([prefix + head_tokens(t, 1500) for t in texts[i:i + batch]])
    return out


async def load(conn, doc_id: str, file_name: str, pages: list[dict], progress=print) -> dict:
    """(Re)build a document's rows from its page records.  Returns counts."""
    meta = describe(pages, file_name)
    await store.delete_document(conn, doc_id, keep_row=True)
    as_of = meta["as_of"]
    await conn.execute(
        """
        INSERT INTO documents
            (doc_id, file_name, title, doc_type, entity, as_of_key, as_of_label,
             pages, status, progress)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (doc_id) DO UPDATE SET
            file_name    = EXCLUDED.file_name,
            title        = EXCLUDED.title,
            doc_type     = EXCLUDED.doc_type,
            as_of_key    = EXCLUDED.as_of_key,
            as_of_label  = EXCLUDED.as_of_label,
            pages        = EXCLUDED.pages,
            status       = 'indexing',
            progress     = EXCLUDED.progress
        """,
        doc_id, file_name, meta["title"], meta["doc_type"], "",
        as_of.end.isoformat() if as_of else "",
        as_of.label if as_of else "",
        len(pages), "indexing", "building facts",
    )

    block_texts, series_texts, n_facts = [], [], 0
    excluded_pages = 0
    for rec in pages:
        await conn.execute(
            "INSERT INTO pages (doc_id, page, heading, text, image, record) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            doc_id, rec["page"], rec.get("heading", ""), rec.get("text", ""),
            rec.get("image", ""), json.dumps(rec, ensure_ascii=False),
        )
        if rec.get("exclude_from_index"):
            excluded_pages += 1
            continue
        out = facts.page_facts(rec, meta["title"])
        ids = []
        for b in out.blocks:
            bid = await store.add_block(
                conn, doc_id, rec["page"], b["kind"], b["text"],
                section=rec.get("heading", ""),
                title=b.get("title", ""),
                content=b.get("content"),
                bbox=b.get("bbox"),
                source=b.get("source", ""),
                confidence=b.get("confidence", 1.0),
                section_path=b.get("section_path"),
            )
            ids.append(bid)
            block_texts.append((bid, embed_text(b)))
        for b, bid in zip(out.blocks, ids):
            if b.get("anchor") is not None:
                await conn.execute(
                    "UPDATE blocks SET anchor_block_id = $1 WHERE block_id = $2",
                    ids[b["anchor"]], bid,
                )
        for s in out.series:
            sid = await store.add_series(
                conn, doc_id, rec["page"],
                ids[s["block"]] if s["block"] is not None else None,
                s["label"], s["name"], s["metric_id"], s["kind"], s["unit"], s["entity"],
                s["source"], s["confidence"], s.get("path"), s.get("scale"),
            )
            dated = [f["period"] for f in s["facts"] if f["period"]]
            span = f"{dated[0].label} to {dated[-1].label}" if dated else "period not stated"
            short = (" › ".join((s.get("path") or [s["label"]])[1:] or [s["label"]])
                     if rec.get("heading") else s["label"])
            series_texts.append((sid, f"{short} ({s['unit'] or s['kind'] or 'value'}), {span}"))
            for f in s["facts"]:
                await store.add_fact(
                    conn, sid, doc_id, rec["page"], f["period"], f["value"],
                    f["printed"], f["validated"], f["confidence"], f["note"],
                )
                n_facts += 1
        for kind, ref, msg, sugg in out.issues:
            await store.add_issue(conn, doc_id, rec["page"], kind, ref, msg, sugg)

    applied = await apply_corrections(conn, doc_id)
    conflicts = await cross_document_check(conn, doc_id)
    calculated = await generic_metrics.rebuild(conn, doc_id)
    progress(f"embedding {len(block_texts)} passages and {len(series_texts)} series")
    await conn.execute("UPDATE documents SET progress = $1 WHERE doc_id = $2",
                       "embedding", doc_id)

    # Embeddings — compute off-loop (CPU bound) via asyncio.to_thread.
    block_vecs = await asyncio.to_thread(
        _embed_all, [t for _, t in block_texts], "search_document: ")
    series_vecs = await asyncio.to_thread(
        _embed_all, [t for _, t in series_texts], "search_document: ")

    await store.set_embeddings(conn, "blocks", doc_id,
                               zip([i for i, _ in block_texts], block_vecs))
    await store.set_embeddings(conn, "series", doc_id,
                               zip([i for i, _ in series_texts], series_vecs))

    await conn.execute(
        "UPDATE documents SET status = 'ready', progress = '' WHERE doc_id = $1",
        doc_id,
    )
    return {"blocks": len(block_texts), "series": len(series_texts), "facts": n_facts,
            "excluded_pages": excluded_pages, "corrections": applied,
            "conflicts": conflicts, "calculated_metrics": calculated}


async def apply_corrections(conn, doc_id: str) -> int:
    """Reapply reviewer corrections saved in the ``corrections`` table."""
    n = 0
    rows = await conn.fetch("SELECT * FROM corrections WHERE doc_id = $1", doc_id)
    for c in rows:
        note = f"corrected by reviewer{': ' + c['note'] if c['note'] else ''}"
        tag = await conn.execute(
            """
            UPDATE facts SET value = $1, validated = 1, confidence = 1.0, note = $2
            WHERE doc_id = $3 AND page = $4 AND period_key = $5
              AND series_id IN (
                  SELECT series_id FROM series WHERE doc_id = $3 AND label = $6)
            """,
            float(c["value"]) if c["value"] is not None else None,
            note, doc_id, c["page"], c["period_key"], c["series_label"],
        )
        # asyncpg returns a command tag like "UPDATE N".
        try:
            n += int(tag.rsplit(" ", 1)[-1])
        except (ValueError, AttributeError):
            pass
    return n


async def cross_document_check(conn, doc_id: str) -> int:
    """Same metric/entity/period across documents with a different value: flag it."""
    rows = await conn.fetch("""
        SELECT s.label, s.metric_id, f.period_key, f.period_label, f.value, f.page,
               o.value AS other_value, o.page AS other_page,
               d.title AS other_title, d.as_of_key AS other_as_of
        FROM facts f
        JOIN series s  ON s.series_id = f.series_id
        JOIN series os ON os.metric_id = s.metric_id
                      AND os.entity    = s.entity
                      AND os.doc_id   <> s.doc_id
        JOIN facts o   ON o.series_id = os.series_id
                      AND o.period_key = f.period_key
                      AND o.partial    = f.partial
        JOIN documents d ON d.doc_id = os.doc_id
        WHERE f.doc_id = $1 AND s.metric_id <> ''
    """, doc_id)
    n = 0
    for r in rows:
        if r["value"] is None or r["other_value"] is None:
            continue
        if abs(r["value"] - r["other_value"]) > max(0.005 * abs(r["other_value"]), 0.01):
            await store.add_issue(
                conn, doc_id, r["page"], "cross_doc_conflict",
                f"{r['label']} @ {r['period_label']}",
                f"{r['value']:g} here vs {r['other_value']:g} in "
                f"'{r['other_title']}' p.{r['other_page']}",
                "restated? the newer document is preferred in answers",
            )
            n += 1
    return n


def copy_images(pages: list[dict], folder: Path) -> None:
    (folder / "pages").mkdir(parents=True, exist_ok=True)
    for p in pages:
        src = Path(p.get("image", ""))
        dst = folder / "pages" / f"page_{p['page']:03d}.png"
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        p["image"] = str(dst.relative_to(config.PROJECT_DIR)) if dst.exists() else p.get("image", "")


def _libreoffice_executable() -> str | None:
    """Find LibreOffice on PATH or in its usual macOS/Windows installation directory."""
    configured = os.getenv("SOFFICE_PATH", "").strip()
    bundled = sorted((config.DATA_DIR / "tools").glob("LibreOffice*.app"), reverse=True)
    candidates = [
        configured,
        shutil.which("soffice") or "",
        shutil.which("libreoffice") or "",
        *(str(app / "Contents" / "MacOS" / "soffice") for app in bundled),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for root in (os.getenv("PROGRAMFILES"), os.getenv("PROGRAMFILES(X86)")):
        if root:
            candidates.append(str(Path(root) / "LibreOffice" / "program" / "soffice.exe"))
    return next((p for p in candidates if p and Path(p).is_file()), None)


def ingestion_pdf(source_path: str, doc_id: str, fresh: bool = False) -> Path:
    """Return a PDF for the extractor, converting PPTX atomically when needed."""
    source = Path(source_path).resolve()
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return source
    if suffix != ".pptx":
        raise ValueError(f"Unsupported document type '{source.suffix}'; use PDF or PPTX")

    output_dir = config.DATA_DIR / "converted" / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / f"{source.stem}.pdf"
    if output_pdf.exists() and not fresh:
        return output_pdf

    with tempfile.TemporaryDirectory(prefix="convert_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        converted = temp_dir / f"{source.stem}.pdf"
        soffice = _libreoffice_executable()
        if soffice:
            # A private profile prevents simultaneous uploads from sharing LibreOffice's lock.
            profile = temp_dir / "lo_profile"
            result = subprocess.run(
                [soffice, f"-env:UserInstallation={profile.as_uri()}", "--headless",
                 "--convert-to", "pdf", "--outdir", str(temp_dir), str(source)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"PowerPoint-to-PDF conversion failed: {detail}")
        else:
            raise RuntimeError(
                "PPTX ingestion requires LibreOffice (set SOFFICE_PATH if it is installed)."
            )
        if not converted.is_file() or converted.stat().st_size == 0:
            raise RuntimeError("PowerPoint-to-PDF conversion produced no PDF")
        converted.replace(output_pdf)
    return output_pdf


async def add_pdf(document_path: str, use_vision: bool = True, fresh: bool = False,
                  progress=print, vision_model: str = config.VISION_MODEL) -> str:
    """Ingest a PDF or PPTX file into the document library.

    PPTX files are converted to PDF via LibreOffice before extraction.
    Backward compatible — all existing callers that pass a PDF path
    continue to work unchanged.
    """
    from . import ingest  # heavy imports (pdfplumber, pymupdf) only when extracting

    doc_id = ingest.file_hash(document_path)
    file_name = Path(document_path).name

    async with store.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (doc_id, file_name, title, status, progress)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (doc_id) DO UPDATE SET status = 'extracting'
            """,
            doc_id, file_name, Path(document_path).stem, "extracting", "starting",
        )

        async def show(num: int, total: int, note: str) -> None:
            await conn.execute(
                "UPDATE documents SET progress = $1 WHERE doc_id = $2",
                f"page {num}/{total}", doc_id,
            )
            progress(f"page {num}/{total} ({note})")

        # ingest.ingest() is sync and expects a sync progress callback; adapt.
        def sync_show(num: int, total: int, note: str) -> None:
            # Schedule the DB update on the running loop without blocking.
            try:
                asyncio.get_event_loop().create_task(show(num, total, note))
            except RuntimeError:
                progress(f"page {num}/{total} ({note})")

        try:
            await conn.execute(
                "UPDATE documents SET progress = $1 WHERE doc_id = $2",
                "preparing document", doc_id,
            )
            # Convert PPTX → PDF if needed (sync, runs in thread to avoid blocking).
            input_pdf = await asyncio.to_thread(
                ingestion_pdf, document_path, doc_id, fresh,
            )
            async with doc_paths(doc_id, file_name) as folder:
                pages = await asyncio.to_thread(
                    ingest.ingest, str(input_pdf),
                    use_vision=use_vision, fresh=fresh,
                    vision_model=vision_model, progress=sync_show,
                )
            copy_images(pages, folder)
            counts = await load(conn, doc_id, file_name, pages, progress)
        except Exception as e:
            await conn.execute(
                "UPDATE documents SET status = 'failed', progress = $1 WHERE doc_id = $2",
                str(e)[:200], doc_id,
            )
            raise

    progress(f"{file_name}: {counts}")
    return doc_id


async def import_cache(pages_json: str, pdf_path: str | None, progress=print) -> str:
    data = json.loads(Path(pages_json).read_text(encoding="utf-8"))
    doc_id = data["pdf_hash"]
    file_name = Path(pdf_path).name if pdf_path else Path(data.get("source", pages_json)).name
    folder = document_folder(file_name, doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    pages = [data["pages"][k] for k in sorted(data["pages"], key=int)]
    copy_images(pages, folder)
    shutil.copyfile(pages_json, folder / "pages.json")
    async with store.acquire() as conn:
        counts = await load(conn, doc_id, file_name, pages, progress)
    progress(f"{file_name}: {counts}")
    return doc_id


async def reload_all(progress=print) -> None:
    async with store.acquire() as conn:
        docs = await store.documents(conn)
        for d in docs:
            rows = await conn.fetch(
                "SELECT record FROM pages WHERE doc_id = $1 ORDER BY page",
                d["doc_id"],
            )
            pages = [json.loads(r["record"]) for r in rows]
            if pages:
                counts = await load(conn, d["doc_id"], d["file_name"], pages, progress)
                progress(f"{d['file_name']}: {counts}")


async def _list_docs() -> None:
    async with store.acquire() as conn:
        docs = await store.documents(conn)
        for d in docs:
            n = await conn.fetchval(
                "SELECT count(*) FROM facts WHERE doc_id = $1", d["doc_id"])
            print(f"{d['doc_id']}  {d['status']:8} {d['as_of_label'] or '-':8} "
                  f"{d['doc_type']:22} {n:6} facts  {d['file_name']}  "
                  f"{(d['title'] or '')[:50]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("document", help="PDF or PPTX file to ingest")
    a.add_argument("--no-vision", action="store_true")
    a.add_argument("--fresh", action="store_true")
    a.add_argument("--model", default=config.VISION_MODEL)
    i = sub.add_parser("import")
    i.add_argument("pages_json")
    i.add_argument("--pdf")
    sub.add_parser("reload")
    sub.add_parser("list")
    args = ap.parse_args()

    # Initialise the shared PostgreSQL pool before any store call.
    from core.db import init_db, close_db

    async def run():
        await init_db()
        try:
            if args.cmd == "add":
                await add_pdf(args.document, use_vision=not args.no_vision,
                              fresh=args.fresh, vision_model=args.model)
            elif args.cmd == "import":
                await import_cache(args.pages_json, args.pdf)
            elif args.cmd == "reload":
                await reload_all()
            else:
                await _list_docs()
        finally:
            await close_db()

    t0 = time.time()
    asyncio.run(run())
    print(f"({time.time() - t0:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
