"""docTR OCR adapter.

This module is deliberately separate from the PDF/table pipeline so the OCR
implementation can be changed without changing the JSON schema or table logic.
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Point docTR to locally cached model weights so it never attempts a download.
# Must be set before any `from doctr …` import.
os.environ.setdefault("DOCTR_CACHE_DIR", r"C:\Users\C59249\.cache\doctr")

# Prevent docTR from importing weasyprint (HTML export) which requires
# GTK/GLib native libraries not available on Windows.  We only use the
# OCR recognition and detection models, never the HTML I/O layer.
if "weasyprint" not in sys.modules:
    sys.modules["weasyprint"] = MagicMock()


def _patch_doctr_offline():
    """Monkey-patch docTR's download_from_url so it never hits the network.

    If the exact hash-named file is missing but a file with the same model
    prefix exists in the cache (e.g. crnn_vgg16_bn-0417f351.pt instead of
    crnn_vgg16_bn-9762b0b0.pt), that file is used instead.  This lets a
    corporate machine that cannot reach doctr-static.mindee.com work with
    whichever model weights were manually placed in the cache directory.
    """
    import doctr.utils.data as _du

    _original = _du.download_from_url

    def _offline_download(url, file_name=None, hash_prefix=None,
                          cache_dir=None, cache_subdir=None, **kw):
        if not isinstance(file_name, str):
            file_name = url.rpartition("/")[-1].split("&")[0]
        cache_dir = cache_dir or os.environ.get(
            "DOCTR_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "doctr"),
        )
        folder = Path(cache_dir) if cache_subdir is None else Path(cache_dir, cache_subdir)
        exact = folder / file_name

        # Exact file present and non-empty → use it (skip hash check).
        if exact.is_file() and exact.stat().st_size > 0:
            logging.info("doctr offline: using %s", exact)
            return exact

        # Look for an alternative file with the same model prefix
        # e.g. crnn_vgg16_bn-*.pt when crnn_vgg16_bn-9762b0b0.pt is missing.
        stem = file_name.split("-")[0]  # "crnn_vgg16_bn"
        suffix = Path(file_name).suffix       # ".pt"
        candidates = sorted(
            (p for p in folder.glob(f"{stem}-*{suffix}")
             if p.stat().st_size > 0),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            logging.info("doctr offline: %s not found, using %s", file_name, candidates[0].name)
            return candidates[0]

        raise FileNotFoundError(
            f"doctr offline mode: {file_name} not found in {folder} and no "
            f"alternative {stem}-*{suffix} file is available.  Download the "
            f"model manually from {url} and place it in {folder}."
        )

    _du.download_from_url = _offline_download

    # Also patch the direct reference held by the model-loading module,
    # which imported the function before our patch could run.
    try:
        import doctr.models.utils.pytorch as _mu
        _mu.download_from_url = _offline_download
    except (ImportError, AttributeError):
        pass


_patch_doctr_offline()

import numpy as np
from PIL import Image

_recognizer = None
_page_engine = None

NAME = "doctr"
RECO_MODEL = os.getenv("DOCTR_RECO_MODEL", "crnn_vgg16_bn")
DET_MODEL = os.getenv("DOCTR_DET_MODEL", "fast_base")
RECO_BATCH_SIZE = int(os.getenv("DOCTR_RECO_BATCH_SIZE", "32"))


def _rgb(image: np.ndarray) -> np.ndarray:
    """Convert one grayscale/RGB crop to the uint8 RGB input docTR expects."""
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def recognizer():
    global _recognizer
    if _recognizer is None:
        from doctr.models import recognition_predictor

        _recognizer = recognition_predictor(
            RECO_MODEL,
            pretrained=True,
            batch_size=RECO_BATCH_SIZE,
            symmetric_pad=True,
        )
    return _recognizer


def recognize(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """Recognize already-segmented table cells, preserving the old return type."""
    if not crops:
        return []
    return [(text.strip(), float(score)) for text, score in recognizer()([_rgb(c) for c in crops])]


def page_engine():
    global _page_engine
    if _page_engine is None:
        from doctr.models import ocr_predictor

        _page_engine = ocr_predictor(
            det_arch=DET_MODEL,
            reco_arch=RECO_MODEL,
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=True,
            symmetric_pad=True,
        )
    return _page_engine


def page_text(image_path: str | Path) -> str:
    """Read a scanned page and return its lines in document order."""
    with Image.open(image_path) as image:
        page = np.asarray(image.convert("RGB"))
    result = page_engine()([page])
    return result.pages[0].render().strip() if result.pages else ""
