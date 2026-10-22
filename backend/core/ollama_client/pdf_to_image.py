"""
core/ollama_client/pdf_to_image.py
=====================================
Convert a PDF document (as raw bytes) into a list of PNG page images,
one per page. Used by the Email agent's OCR pipeline to feed individual
page images into `OllamaClient.vision_ocr()`.

Uses `pypdfium2` for PDF rendering and `Pillow` for the bitmap-to-PNG
conversion. Both are already listed in `requirements.txt`.
"""

import io

import pypdfium2 as pdfium
from PIL import Image

from core.logging_config import get_agent_logger

logger = get_agent_logger("pdf_to_image")

# Render at 200 DPI - a good balance between OCR accuracy and memory
# usage for typical scanned business documents.
_DEFAULT_SCALE = 200 / 72  # 72 DPI is the PDF default unit


def pdf_to_page_images(pdf_bytes: bytes, scale: float = _DEFAULT_SCALE) -> list[bytes]:
    """
    Render every page of a PDF into a PNG image (as raw bytes).

    Args:
        pdf_bytes: the full PDF file content as raw bytes.
        scale: rendering scale factor (default ~2.78x for 200 DPI).

    Returns:
        A list of PNG image byte strings, one per page.
    """
    pdf = pdfium.PdfDocument(pdf_bytes)
    page_images: list[bytes] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Convert to RGB if needed (some PDFs produce RGBA or L mode).
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        page_images.append(buf.getvalue())

        logger.debug("Rendered page %d: %dx%d", page_index + 1, pil_image.width, pil_image.height)

    logger.info("Converted PDF (%d bytes) to %d page image(s)", len(pdf_bytes), len(page_images))
    return page_images
