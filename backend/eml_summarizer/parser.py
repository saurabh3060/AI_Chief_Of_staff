"""
eml_summarizer/parser.py
==========================
Parses `.eml` files from the configured storage directory into a
structured dict containing subject, sender, recipients, date, and
the plain-text body.

Uses Python's built-in `email` module — no extra dependencies needed.
"""

import email
import email.policy
import os
from pathlib import Path

from core.config import BACKEND_ROOT, get_settings
from core.logging_config import get_agent_logger
from core.text_match import html_to_text

logger = get_agent_logger("eml_summarizer")


def _get_eml_dir() -> Path:
    """
    Return the absolute path to the EML storage directory.

    Does NOT create the directory. Graph API is the single source of
    truth for mail; auto-creating this folder would re-materialize an
    empty `temp_mail_data` and risk a second, stale mail source. The
    folder is used only when it already exists, so callers must tolerate
    its absence.
    """
    settings = get_settings()
    return BACKEND_ROOT / settings.eml_storage_dir


def list_eml_files() -> list[str]:
    """
    Return a sorted list of `.eml` filenames (not full paths) present
    in the configured storage directory. Returns an empty list when the
    directory does not exist.
    """
    eml_dir = _get_eml_dir()
    if not eml_dir.is_dir():
        return []
    return sorted(f.name for f in eml_dir.iterdir() if f.suffix.lower() == ".eml")


def parse_eml_file(filename: str) -> dict:
    """
    Parse a single `.eml` file and return a structured dict:

        {
            "filename": "report.eml",
            "subject": "...",
            "from": "...",
            "to": "...",
            "cc": "...",
            "date": "...",
            "body": "...(plain-text body)..."
        }

    Raises ``FileNotFoundError`` if the file does not exist.
    Raises ``ValueError`` if the file cannot be parsed.
    """
    eml_dir_resolved = _get_eml_dir().resolve()

    # Security (CWE-22/CWE-23 path injection): `filename` is
    # user-controlled, so it must never be able to escape the EML
    # storage directory.
    #
    # 1. Strip every directory component to a bare filename - this
    #    neutralizes `../`, absolute paths, and drive/UNC prefixes.
    # 2. Reject a name that isn't a plain `*.eml` leaf (empty, `.`,
    #    `..`, or a name that still somehow contains a separator).
    # 3. Canonicalise with os.path.realpath and assert the candidate is
    #    CONTAINED within the realpath'd storage dir via
    #    os.path.commonpath, which guarantees the file cannot escape the
    #    base directory via path traversal or symlinks.
    safe_name = Path(filename).name
    if (
        not safe_name
        or safe_name in {".", ".."}
        or "/" in safe_name
        or "\\" in safe_name
        or not safe_name.lower().endswith(".eml")
    ):
        raise ValueError(f"Invalid email filename: {filename!r}")

    # Security (CWE-22/CWE-23 path injection): the path passed to open()
    # must NOT be constructed from the user-controlled value at all.
    # Instead, enumerate the real files already present in the storage
    # directory and select the Path object whose name matches. The chosen
    # `filepath` therefore originates entirely from the filesystem listing
    # (a trusted source), and the user input is used only for an equality
    # comparison against each known-good entry - never to build the path.
    # This fully breaks the taint flow from `filename` into exists()/open().
    eml_dir = _get_eml_dir()
    if not eml_dir.is_dir():
        raise FileNotFoundError(
            f"Email file '{safe_name}' not found in the emails folder. "
            f"Available files: []"
        )

    filepath: Path | None = None
    available: list[str] = []
    real_base = os.path.realpath(str(eml_dir_resolved))
    for entry in eml_dir.iterdir():
        if entry.suffix.lower() != ".eml":
            continue
        available.append(entry.name)
        if entry.name == safe_name:
            # Defence in depth: confirm the trusted entry still resolves
            # inside the base directory (guards against symlink escape).
            real_entry = os.path.realpath(str(entry))
            if (
                real_entry == real_base
                or os.path.commonpath([real_base, real_entry]) == real_base
            ):
                filepath = entry

    available.sort()

    if filepath is None:
        raise FileNotFoundError(
            f"Email file '{safe_name}' not found in the emails folder. "
            f"Available files: {available}"
        )

    # Strip CR/LF from the user-controlled filename before logging to
    # prevent log injection / forged log lines (CWE-117).
    sanitized_item_id = safe_name.replace("\r\n", "").replace("\r", "").replace("\n", "")
    logger.info("Parsing EML file: %s", sanitized_item_id)

    with open(filepath, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    body = _extract_body(msg)

    return {
        "filename": safe_name,
        "subject": msg.get("Subject", "(no subject)"),
        "from": msg.get("From", "(unknown sender)"),
        "to": msg.get("To", ""),
        "cc": msg.get("Cc", ""),
        "date": msg.get("Date", ""),
        "body": body,
    }


def _extract_body(msg: email.message.Message) -> str:
    """
    Extract the plain-text body from an email message.

    Walks MIME parts, preferring text/plain. Falls back to text/html
    with tags stripped if no plain-text part is found.
    """
    # If it's not multipart, grab the payload directly.
    if not msg.is_multipart():
        content_type = msg.get_content_type()
        payload = msg.get_content()
        if isinstance(payload, str):
            if content_type == "text/html":
                return html_to_text(payload)
            return payload
        return str(payload) if payload else "(empty body)"

    # Walk multipart — prefer text/plain, fall back to text/html.
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in msg.walk():
        content_type = part.get_content_type()
        # Skip attachments.
        if part.get_content_disposition() == "attachment":
            continue
        try:
            payload = part.get_content()
        except (KeyError, LookupError):
            continue

        if not isinstance(payload, str):
            continue

        if content_type == "text/plain":
            plain_parts.append(payload)
        elif content_type == "text/html":
            html_parts.append(payload)

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        return html_to_text("\n".join(html_parts))
    return "(empty body)"
