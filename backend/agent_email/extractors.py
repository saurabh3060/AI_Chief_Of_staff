"""
agent_email/extractors.py
=============================
Direct (non-vision) text extraction for email attachments, kept
deliberately separate from `agent_email.ocr` (which owns the
vision-model OCR pipeline for images/scanned pages).

For the common case of "born-digital" PDFs, Excel workbooks, and
PowerPoint decks, this module pulls text straight out of the file's own
structure instead of rasterizing pages and running them through the
vision model - cheaper and more accurate whenever the document actually
has extractable text (as opposed to a PDF that's just a scanned image).

- PDFs: `extract_pdf_text` uses PyMuPDF (`fitz`).
- Excel workbooks (.xlsx/.xlsm): `extract_excel_text` uses `openpyxl`.
- PowerPoint decks (.pptx): `extract_pptx_text` uses `python-pptx`.
- Word docs (.docx): `extract_docx_text` uses `python-docx`.

All functions support table fact extraction via `fact_extraction` module
to prepare structured facts for LLM summarization.

Neither function talks to Ollama or does any vision-model work - that
logic lives exclusively in `agent_email.ocr`.
"""

import io

import fitz  # PyMuPDF
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

# Import fact extraction module for table processing
from agent_email import fact_extraction


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract all selectable text from a PDF's pages, in order, using
    PyMuPDF. Returns an empty string if the PDF has no extractable text
    (e.g. it's a scanned image with no text layer) - callers should
    fall back to the vision OCR pipeline in that case.
    """
    pages_text: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pages_text.append(page.get_text())

    extracted_text = "\n\n".join(t.strip() for t in pages_text if t.strip())
    logger.info("PyMuPDF extracted %d chars of text from %d page(s)", len(extracted_text), len(pages_text))
    
    # Extract facts from tables and append to the extracted text
    table_facts = fact_extraction.extract_and_format_tables(extracted_text)
    if table_facts and table_facts != "(no facts extracted from tables)":
        extracted_text += "\n\n" + table_facts
        logger.info("Added table facts extraction (%d chars)", len(table_facts))
    
    return extracted_text


def extract_excel_text(excel_bytes: bytes) -> str:
    """
    Extract all cell values from every sheet of an Excel workbook, row
    by row, as plain tab-separated text using `openpyxl`. Empty cells
    are rendered as empty strings so column alignment is preserved;
    fully empty rows are skipped.
    """
    workbook = load_workbook(io.BytesIO(excel_bytes), data_only=True, read_only=True)
    sheet_blocks: list[str] = []

    for sheet in workbook.worksheets:
        rows_text: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            if row is None or all(cell is None for cell in row):
                continue
            rows_text.append("\t".join("" if cell is None else str(cell) for cell in row))
        if rows_text:
            sheet_blocks.append(f"Sheet: {sheet.title}\n" + "\n".join(rows_text))

    extracted_text = "\n\n".join(sheet_blocks)
    logger.info("openpyxl extracted %d chars of text from %d sheet(s)", len(extracted_text), len(workbook.worksheets))
    
    # Extract facts from tables and append to the extracted text
    table_facts = fact_extraction.extract_and_format_tables(extracted_text)
    if table_facts and table_facts != "(no facts extracted from tables)":
        extracted_text += "\n\n" + table_facts
        logger.info("Added table facts extraction (%d chars)", len(table_facts))
    
    return extracted_text


def extract_pptx_text(pptx_bytes: bytes) -> str:
    """
    Extract all text from a PowerPoint deck (.pptx) using `python-pptx`:
    every shape's text frame (titles, body text, text boxes) plus any
    table cell text, slide by slide, in slide order. Speaker notes are
    intentionally excluded - they are presenter-facing, not part of the
    content a recipient sees, and including them would mix
    presenter-only asides into what's meant to be an on-slide-content
    summary.

    Returns an empty string if the deck has no extractable text (e.g. a
    deck made entirely of slide images with no text shapes) - there is
    no vision-OCR fallback for PPTX in this pipeline, matching the
    existing Excel handling (direct extraction only).
    """
    slide_blocks: list[str] = []
    presentation = Presentation(io.BytesIO(pptx_bytes))

    for slide_index, slide in enumerate(presentation.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = "\n".join(
                    paragraph.text for paragraph in shape.text_frame.paragraphs if paragraph.text.strip()
                )
                if text.strip():
                    lines.append(text.strip())
            elif shape.has_table:
                for row in shape.table.rows:
                    row_text = "\t".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        lines.append(row_text)
        if lines:
            slide_blocks.append(f"Slide {slide_index}:\n" + "\n".join(lines))

    extracted_text = "\n\n".join(slide_blocks)
    logger.info("python-pptx extracted %d chars of text from %d slide(s)", len(extracted_text), len(presentation.slides))
    
    # Extract facts from tables and append to the extracted text
    table_facts = fact_extraction.extract_and_format_tables(extracted_text)
    if table_facts and table_facts != "(no facts extracted from tables)":
        extracted_text += "\n\n" + table_facts
        logger.info("Added table facts extraction (%d chars)", len(table_facts))
    
    return extracted_text


def extract_docx_text(docx_bytes: bytes) -> str:
    """
    Extract all text from a Word document (.docx) using `python-docx`:
    every paragraph in document order, plus every table's cell text
    (rows tab-separated). This is the format the Wholesale Banking
    "Minutes of Meeting" attachments arrive in, whose content was
    previously being dropped entirely (the pipeline only handled
    PDF/Excel/PPTX), so the LLM only ever saw the short forwarding note
    and never the actual minutes.

    Tables are included because MoM documents frequently record
    action-items/owners/deadlines in a table rather than prose. Returns
    an empty string if the document has no extractable text.
    """
    document = Document(io.BytesIO(docx_bytes))

    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)

    for table_index, table in enumerate(document.tables, start=1):
        rows_text: list[str] = []
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                rows_text.append(row_text)
        if rows_text:
            blocks.append(f"Table {table_index}:\n" + "\n".join(rows_text))

    extracted_text = "\n\n".join(blocks)
    logger.info(
        "python-docx extracted %d chars of text from %d paragraph(s) and %d table(s)",
        len(extracted_text), len(document.paragraphs), len(document.tables),
    )
    
    # Extract facts from tables and append to the extracted text
    table_facts = fact_extraction.extract_and_format_tables(extracted_text)
    if table_facts and table_facts != "(no facts extracted from tables)":
        extracted_text += "\n\n" + table_facts
        logger.info("Added table facts extraction (%d chars)", len(table_facts))
    
    return extracted_text
