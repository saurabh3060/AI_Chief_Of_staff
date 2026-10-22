"""Extract heading, tables, text and chart descriptions from each PDF page.

Usage: python ingest.py file.pdf [--no-vision] [--fresh]
Writes data/pages.json incrementally, so an interrupted run resumes where it stopped.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable

import pymupdf
import pdfplumber

try:
    from . import chartread
except ImportError:  # chartread.py is not part of the delivered files; new-PDF ingestion needs it
    chartread = None
from . import config
from . import llm
from . import tables

# granite-vision's own chart-to-CSV instruction (what its <chart2csv> tag expands to; the tag
# itself fails to tokenize in the GGUF build), extended for pages with several charts.
# Chart notes and tables are not needed here: tables.py reads the tables.
CHART_PROMPT = """Please examine this chart image. Consider you are a data visualization expert, and extract the data into a CSV table.

Your CSV should:
- Include a header row with clear column names
- Represent all data series/categories shown in the chart
- Use numeric values that match the chart as closely as possible
If the image contains several charts, output one CSV per chart, each preceded by a line
### Chart: <chart title as printed>
Ignore tables and body text. If there is no chart, reply exactly: NO_CHART

Output only the chart titles and CSV data, nothing else."""

def looks_like_chart_data(text: str) -> bool:
    """The model sometimes invents a one-row CSV for a page without charts (e.g. a cover page)."""
    rows = [line for line in text.splitlines() if "," in line and not line.startswith(("#", "```"))]
    return len(rows) >= 3  # header + at least 2 data rows


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def extract_heading(page: pdfplumber.page.Page) -> str:
    """Largest-font line in the top third of the page."""
    words = page.extract_words(extra_attrs=["size"], keep_blank_chars=False)
    top_words = [w for w in words if w["top"] < page.height / 3]
    if not top_words:
        return ""
    max_size = max(w["size"] for w in top_words)
    heading_words = [w for w in top_words if w["size"] >= max_size - 0.5]
    heading_words.sort(key=lambda w: (round(w["top"]), w["x0"]))
    return " ".join(w["text"] for w in heading_words).strip()


MIN_GRAPHIC_AREA = 0.03  # of the page; smaller images are logos and icons


def has_graphics(fitz_page: pymupdf.Page, table_boxes: list = ()) -> bool:
    """True if the page has, outside its tables, an image bigger than a logo or enough vector
    drawings to be a chart. A picture already read as a table, a table's own cell shading and
    rules, or a cover slide's background picture is not a chart and needs no vision pass."""
    page_area = fitz_page.rect.width * fitz_page.rect.height

    # Include ruled grids that were intentionally omitted from facts (for example a detached
    # numeric block with no row labels).  Their borders are layout, not a chart.
    layout_boxes = list(table_boxes)
    try:
        layout_boxes += [tuple(t.bbox) for t in fitz_page.find_tables(strategy="lines").tables]
    except Exception:
        pass

    def in_table(r: pymupdf.Rect) -> bool:
        # By centre, with a margin: rules are zero-width lines, and cell shading runs past the text.
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        return any(b[0] - 12 <= cx <= b[2] + 12 and b[1] - 12 <= cy <= b[3] + 12 for b in layout_boxes)

    words = fitz_page.get_text("words")
    for info in fitz_page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & fitz_page.rect
        if r.width * r.height >= MIN_GRAPHIC_AREA * page_area and not in_table(r) \
                and not tables.is_background(fitz_page, r, words):
            return True
    marks = []
    for drawing in fitz_page.get_drawings():
        r = drawing["rect"] & fitz_page.rect
        if in_table(r) or r.width <= 2 or r.height <= 2:
            continue
        # Full-width section bands and navigation tiles are common in dashboards.  A chart
        # needs a cluster of actual marks, not merely shallow decorative rectangles.
        if r.width >= 0.45 * fitz_page.rect.width and r.height <= 0.06 * fitz_page.rect.height:
            continue
        marks.append(r)
    return len(marks) > 20


def ocr_page_text(img_path: str) -> str:
    """Text of a page with no text layer (a scan, a slide pasted as one picture), line by line."""
    from . import gridocr
    return gridocr.page_text(img_path)


def describe_charts(image_path: str, model: str = config.VISION_MODEL) -> str:
    content = [llm.image_part(image_path), {"type": "text", "text": CHART_PROMPT}]
    text = llm.chat(model, [{"role": "user", "content": content}], temperature=0,
                    max_tokens=config.VISION_MAX_TOKENS).strip()
    return text if looks_like_chart_data(text) else ""


def page_charts(fpage: pymupdf.Page, img_path: str, use_vision: bool,
                model: str = config.VISION_MODEL, whole_page: bool = True,
                table_boxes: list = ()) -> tuple[str, list]:
    """Chart CSVs for one page and the regions of the charts read from the text layer.

    Charts whose labels are PDF text are read by chartread (number, period, and series by
    legend color). A chart it is not sure of goes to the vision model cropped to its region;
    a page with no readable chart axis goes to the vision model whole if `whole_page` (it has
    graphics outside its tables that may be a pictured chart).
    """
    charts = chartread.read_charts(fpage)
    # Rows of period-looking table headers can resemble chart axes.  Never send a region to
    # chart vision when a trusted native table already occupies that region.
    charts = [c for c in charts
              if not any(tables._overlap(c.bbox, box) >= 0.5 for box in table_boxes)]
    parts = [c.text for c in charts if c.text]
    if use_vision:
        for n, c in enumerate(c for c in charts if not c.text):
            crop = img_path.replace(".png", f"_chart{n + 1}.png")
            x0, y0, x1, y1 = c.bbox
            fpage.get_pixmap(dpi=config.RENDER_DPI, clip=pymupdf.Rect(x0 - 6, y0 - 4, x1 + 6, y1 + 4)).save(crop)
            text = describe_charts(crop, model=model)
            if text:
                parts.append(chartread.retitle(text, c.title))
        if not charts and whole_page:
            text = describe_charts(img_path, model=model)
            if text:
                parts.append(text)
    return "\n\n".join(parts), [c.bbox for c in charts if c.text]


def drop_chart_tables(found: list[dict], chart_boxes: list) -> list[dict]:
    """Picture tables the vision/OCR pass read out of a chart that chartread already read."""
    return [t for t in found
            if not (t["source"] in ("vision", "gridocr") and t.get("bbox")
                    and any(tables._overlap(t["bbox"], b) >= 0.7 for b in chart_boxes))]


def vision_models() -> list[str]:
    """Vision models from config that the llama.cpp server knows (models.ini)."""
    available = llm.models()
    return [m for m in config.VISION_MODELS if m in available]


def check_vision_model(model: str) -> None:
    available = vision_models()
    if model not in available:
        raise SystemExit(
            f"Vision model '{model}' is not available. Available vision models: "
            f"{', '.join(available) or 'none'}. Use --model <name>, or add it to models.ini "
            f"and config.VISION_MODELS."
        )


# Bump when the page record changes, so cached pages are extracted again.
CACHE_VERSION = 9  # 3: picture tables read by grid OCR; 4: charts read from the text layer;
# 5: side-by-side blocks split, header OCR noise dropped, B1/0_12 label misreads fixed;
# 6: wrapped/staggered whitespace headers, no chart pass for logos or table pictures,
#    background pictures not read as tables, OCR text for pages without a text layer
# 7: selectable docTR/RapidOCR backend; cache records which backend produced it
# 8: exclude generated reconstruction-note pages from report evidence
# 9: embedded-font numeric glyphs, sparse ruled tables, and table-aware chart filtering

_NON_REPORT_PAGE = re.compile(r"(?i)^\s*reconstruction notes\s*\(not part of the original report\)")


def non_report_page(text: str) -> str:
    """Reason a generated provenance/notes page must not become report evidence."""
    return "reconstruction notes" if _NON_REPORT_PAGE.search(text or "") else ""


def load_cache(pdf_hash: str, ocr_backend: str) -> dict:
    if config.PAGES_JSON.exists():
        data = json.loads(config.PAGES_JSON.read_text(encoding="utf-8"))
        if data.get("pdf_hash") == pdf_hash and data.get("version") == CACHE_VERSION \
                and data.get("ocr_backend") == ocr_backend:
            return data
    return {"pdf_hash": pdf_hash, "version": CACHE_VERSION, "ocr_backend": ocr_backend, "pages": {}}


def save_cache(data: dict) -> None:
    tmp = config.PAGES_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config.PAGES_JSON)


def ingest(
    pdf_path: str,
    use_vision: bool = True,
    fresh: bool = False,
    vision_model: str = config.VISION_MODEL,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    from . import gridocr

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Chart crops are disposable extraction artifacts. A fresh re-ingestion may determine that
    # an old chart candidate was really a table. Table-reader crops are equally disposable and
    # otherwise accumulate after extraction strategies improve. Keep only the main page renders;
    # this folder is already isolated to the one document being refreshed.
    if fresh:
        for pattern in ("page_*_chart*", "page_*_t*"):
            for stale in config.PAGE_IMG_DIR.glob(pattern):
                if stale.is_file():
                    stale.unlink()

    pdf_hash = file_hash(pdf_path)
    data = ({"pdf_hash": pdf_hash, "version": CACHE_VERSION, "ocr_backend": gridocr.OCR_BACKEND, "pages": {}}
            if fresh else load_cache(pdf_hash, gridocr.OCR_BACKEND))
    data["source"] = str(pdf_path)

    with pdfplumber.open(pdf_path) as pdf, pymupdf.open(pdf_path) as doc:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            num = i + 1
            if str(num) in data["pages"]:
                if progress:
                    progress(num, total, "cached")
                continue

            t0 = time.time()
            if progress:
                progress(num, total, "reading")
            fpage = doc[i]
            img_path = config.PAGE_IMG_DIR / f"page_{num:03d}.png"
            fpage.get_pixmap(dpi=config.RENDER_DPI).save(str(img_path))

            text, heading, text_source = (page.extract_text() or "").strip(), extract_heading(page), "pdf"
            excluded_reason = non_report_page(text)
            found, chart_desc = [], ""
            if not excluded_reason:
                found = tables.extract_page_tables(page, fpage, f"page_{num:03d}", use_vision, vision_model)
                # chartread always (text layer, cheap); the whole page goes to the vision model only
                # when it has graphics outside its tables.
                table_boxes = [t["bbox"] for t in found if t.get("bbox")]
                chart_desc, chart_boxes = page_charts(
                    fpage, str(img_path), use_vision, vision_model,
                    whole_page=has_graphics(fpage, table_boxes), table_boxes=table_boxes)
                found = drop_chart_tables(found, chart_boxes)
                if len(text) < 20 and any(
                        r.width * r.height >= 0.5 * fpage.rect.width * fpage.rect.height
                        for r in tables.image_regions(fpage, skip_backgrounds=False)):
                    # No text layer over a page-sized picture: a scan. Read its text by OCR.
                    ocr = ocr_page_text(str(img_path))
                    if len(ocr) > len(text):
                        text, text_source = ocr, "ocr"
                        heading = heading or ocr.splitlines()[0][:120]
            record = {
                "page": num,
                "heading": heading,
                "text": text,
                "text_source": text_source,
                "tables": found,
                "tables_md": [md for md in map(tables.to_markdown, found) if md],
                "chart_desc": chart_desc,
                "image": str(img_path),
                "exclude_from_index": bool(excluded_reason),
                "excluded_reason": excluded_reason,
            }

            data["pages"][str(num)] = record
            save_cache(data)
            if progress:
                progress(num, total, f"{time.time() - t0:.1f}s")

    return [data["pages"][k] for k in sorted(data["pages"], key=int)]


def load_pages() -> list[dict]:
    if not config.PAGES_JSON.exists():
        return []
    data = json.loads(config.PAGES_JSON.read_text(encoding="utf-8"))
    return [data["pages"][k] for k in sorted(data["pages"], key=int)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--no-vision", action="store_true", help="skip chart and picture-table reading")
    ap.add_argument("--fresh", action="store_true", help="ignore cache")
    ap.add_argument("--model", default=config.VISION_MODEL, help="vision model for charts")
    args = ap.parse_args()
    if not args.no_vision:
        check_vision_model(args.model)
        print(f"Reading charts with {args.model}", flush=True)

    def show(num: int, total: int, note: str) -> None:
        print(f"page {num}/{total} ({note})", flush=True)

    pages = ingest(
        args.pdf, use_vision=not args.no_vision, fresh=args.fresh, vision_model=args.model, progress=show
    )

    from index import build_index
    build_index(pages)
    print(f"Done: {len(pages)} pages -> {config.PAGES_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
