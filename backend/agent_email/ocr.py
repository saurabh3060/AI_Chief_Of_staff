"""
agent_email/ocr.py
=====================
Attachment text-extraction pipeline for the Email & Summary agent,
owned exclusively by this agent (per project requirements - Action
Tracker never touches attachments or mail directly, it consumes this
agent's finished summaries instead).

Pipeline for a single attachment:
1. Decode the attachment's base64 `contentBytes` (as returned by Graph
   API) into raw bytes.
2. Dispatch by file type:
   - PDF: try direct text extraction first via
     `agent_email.extractors.extract_pdf_text` (PyMuPDF) - cheap and
     accurate for "born-digital" PDFs. If that yields no text (e.g. a
     scanned image with no text layer), fall back to rendering pages
     and running them through the vision OCR pipeline below.
   - Excel (.xlsx/.xlsm): direct extraction only, via
     `agent_email.extractors.extract_excel_text` (openpyxl) - never
     goes through vision OCR.
   - Image (png/jpg/jpeg): vision OCR only, since there's no text
     layer to extract directly.
   Anything else is skipped (not processed) - `ocr_applied=False`.
3. The vision OCR step (PDF fallback + images) converts pages into PNG
   images via `core.ollama_client.pdf_to_image.pdf_to_page_images` and
   runs each one through `core.ollama_client.OllamaClient.vision_ocr`.

The vision-model logic is intentionally isolated to steps that need it
(image attachments, or PDFs with no extractable text) - direct
PDF/Excel extraction never calls into Ollama.

Kept as a separate module (rather than folded into `service.py`) so the
extraction logic is easy to locate, test, and swap out in isolation.
"""

import base64

from core.logging_config import get_agent_logger
from core.ollama_client.client import OllamaClient
from core.ollama_client.pdf_to_image import pdf_to_page_images
from core.config import get_settings

from agent_email.attachment_filter import is_mom_attachment
from agent_email.extractors import extract_docx_text, extract_excel_text, extract_pdf_text, extract_pptx_text
from agent_email.schema import ParsedAttachment
from agent_email.pipeline_metrics import PipelineMetricsCollector

logger = get_agent_logger("agent_email")

# Attachment name suffixes this pipeline knows how to process.
_PDF_SUFFIXES = (".pdf",)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_EXCEL_SUFFIXES = (".xlsx", ".xlsm")
_PPTX_SUFFIXES = (".pptx", ".pptm")
# .docx/.docm are OOXML (read directly by python-docx). Legacy binary
# .doc is included here so it's routed to the same handler, which
# attempts the docx reader and logs/degrades cleanly if the file is a
# true legacy binary python-docx cannot open.
_DOCX_SUFFIXES = (".docx", ".docm", ".doc")


def _is_pdf(name: str) -> bool:
    """Return True if the attachment's filename suggests it's a PDF."""
    return name.lower().endswith(_PDF_SUFFIXES)


def _is_image(name: str) -> bool:
    """Return True if the attachment's filename suggests it's a directly-OCR-able image."""
    return name.lower().endswith(_IMAGE_SUFFIXES)


def _is_excel(name: str) -> bool:
    """Return True if the attachment's filename suggests it's an Excel workbook."""
    return name.lower().endswith(_EXCEL_SUFFIXES)


def _is_pptx(name: str) -> bool:
    """Return True if the attachment's filename suggests it's a PowerPoint deck."""
    return name.lower().endswith(_PPTX_SUFFIXES)


def _is_docx(name: str) -> bool:
    """Return True if the attachment's filename suggests it's a Word document (.docx/.docm/.doc)."""
    return name.lower().endswith(_DOCX_SUFFIXES)


async def _vision_ocr_pages(page_images: list[bytes], client: OllamaClient) -> str:
    """Run each page image through the vision OCR model and concatenate the extracted text."""
    # Check if pipeline is enabled before making Ollama calls
    settings = get_settings()
    if not settings.bu_email_pipeline_enabled:
        logger.debug("[FLAG-CHECK] BU_EMAIL_PIPELINE_ENABLED=false; skipping vision OCR")
        return ""
    
    extracted_pages = [await client.vision_ocr(page_image) for page_image in page_images]
    PipelineMetricsCollector.record_llm_call("attachment_ocr", tokens_out=len("\n\n".join(extracted_pages).split()))
    return "\n\n".join(extracted_pages)


async def parse_attachment(
    raw_attachment: dict,
    ollama_client: OllamaClient | None = None,
    source_message_id: str | None = None,
) -> ParsedAttachment:
    """
    Given a raw Microsoft Graph API attachment object (with
    `contentBytes` populated, e.g. from `GraphClient.get_attachment_content`),
    extract its text (direct extraction for PDF/Excel where possible,
    vision OCR for images or scanned PDFs) and return a
    `ParsedAttachment` with the extracted text.

    `source_message_id` records which message this attachment belongs
    to, so a per-email LLM call can pull in only its own attachment(s)
    instead of every attachment across an entire thread.

    Attachments that aren't PDFs/images/Excel workbooks are returned
    with `ocr_applied=False` and empty `extracted_text` - they are not
    considered a failure, just unsupported for extraction.
    """
    name = raw_attachment.get("name", "unknown")
    attachment_id = raw_attachment.get("id", "")
    content_type = raw_attachment.get("contentType")
    content_bytes_b64 = raw_attachment.get("contentBytes")

    if not content_bytes_b64:
        logger.warning("Attachment '%s' has no contentBytes; skipping extraction", name)
        return ParsedAttachment(attachment_id=attachment_id, name=name, content_type=content_type, source_message_id=source_message_id)

    if not is_mom_attachment(name):
        logger.info("Attachment '%s' does not fuzzy-match MoM keywords; skipping extraction", name)
        return ParsedAttachment(attachment_id=attachment_id, name=name, content_type=content_type, source_message_id=source_message_id)

    raw_bytes = base64.b64decode(content_bytes_b64)

    if _is_excel(name):
        # Excel workbooks never go through vision OCR - openpyxl reads
        # cell values directly.
        extracted_text = extract_excel_text(raw_bytes)
        logger.info("Excel extraction completed for attachment '%s' (%d chars extracted)", name, len(extracted_text))
        return ParsedAttachment(
            attachment_id=attachment_id,
            name=name,
            content_type=content_type,
            extracted_text=extracted_text,
            ocr_applied=bool(extracted_text),
            source_message_id=source_message_id,
        )

    if _is_pptx(name):
        # PowerPoint decks never go through vision OCR either -
        # python-pptx reads shape/table text directly, same treatment
        # as Excel workbooks above.
        extracted_text = extract_pptx_text(raw_bytes)
        logger.info("PPTX extraction completed for attachment '%s' (%d chars extracted)", name, len(extracted_text))
        return ParsedAttachment(
            attachment_id=attachment_id,
            name=name,
            content_type=content_type,
            extracted_text=extracted_text,
            ocr_applied=bool(extracted_text),
            source_message_id=source_message_id,
        )

    if _is_docx(name):
        # Word documents (.docx/.docm) are read directly by python-docx -
        # paragraphs + table cells, no vision OCR, same treatment as
        # Excel/PPTX above. This is the format the Wholesale Banking
        # "Minutes of Meeting" attachments arrive in. Legacy binary .doc
        # is routed here too: python-docx can't open a true binary .doc
        # (raises), so we catch it, log, and degrade to empty text rather
        # than crashing (a real legacy .doc has no direct extractor here).
        try:
            extracted_text = extract_docx_text(raw_bytes)
        except Exception as exc:  # noqa: BLE001 - one bad attachment must not fail the run
            if name.lower().endswith(".doc") and not name.lower().endswith((".docx", ".docm")):
                logger.warning(
                    "Attachment '%s' is a legacy binary .doc that python-docx cannot read "
                    "(only OOXML .docx/.docm is supported): %s",
                    name, exc,
                )
            else:
                logger.warning("Failed to extract DOCX text from attachment '%s': %s", name, exc)
            extracted_text = ""
        logger.info("DOCX extraction completed for attachment '%s' (%d chars extracted)", name, len(extracted_text))
        return ParsedAttachment(
            attachment_id=attachment_id,
            name=name,
            content_type=content_type,
            extracted_text=extracted_text,
            ocr_applied=bool(extracted_text),
            source_message_id=source_message_id,
        )

    if _is_pdf(name):
        # Try direct text extraction first (PyMuPDF) - cheap and
        # accurate for born-digital PDFs. Only fall back to rendering
        # pages through the vision model if no text layer was found.
        extracted_text = extract_pdf_text(raw_bytes)
        if extracted_text:
            logger.info("PDF text extraction completed for attachment '%s' (%d chars extracted)", name, len(extracted_text))
            return ParsedAttachment(
                attachment_id=attachment_id,
                name=name,
                content_type=content_type,
                extracted_text=extracted_text,
                ocr_applied=True,
                source_message_id=source_message_id,
            )
        logger.info("Attachment '%s' has no extractable PDF text; falling back to vision OCR", name)
        client = ollama_client or OllamaClient()
        page_images = pdf_to_page_images(raw_bytes)
        extracted_text = await _vision_ocr_pages(page_images, client)
        logger.info("Vision OCR completed for attachment '%s' (%d page(s), %d chars extracted)", name, len(page_images), len(extracted_text))
        return ParsedAttachment(
            attachment_id=attachment_id,
            name=name,
            content_type=content_type,
            extracted_text=extracted_text,
            ocr_applied=True,
            source_message_id=source_message_id,
        )

    if _is_image(name):
        client = ollama_client or OllamaClient()
        extracted_text = await _vision_ocr_pages([raw_bytes], client)
        logger.info("Vision OCR completed for attachment '%s' (%d chars extracted)", name, len(extracted_text))
        return ParsedAttachment(
            attachment_id=attachment_id,
            name=name,
            content_type=content_type,
            extracted_text=extracted_text,
            ocr_applied=True,
            source_message_id=source_message_id,
        )

    logger.info("Attachment '%s' is not a PDF/image/Excel workbook; skipping extraction", name)
    return ParsedAttachment(attachment_id=attachment_id, name=name, content_type=content_type, source_message_id=source_message_id)
