"""
chatbot/email_parser.py
==========================
Parse .eml files into structured text for RAG indexing.

Handles MIME multipart messages, base64/quoted-printable decoding,
and HTML-to-text conversion via BeautifulSoup.
"""

import os
import email
import base64
import quopri
from email import policy
from email.header import decode_header

from bs4 import BeautifulSoup

from core.config import get_settings, BACKEND_ROOT
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot.email_parser")


def _get_emails_dir() -> str:
    """Resolve the emails directory from project settings."""
    settings = get_settings()
    eml_dir = settings.eml_storage_dir
    # If relative, resolve against BACKEND_ROOT
    if not os.path.isabs(eml_dir):
        eml_dir = str(BACKEND_ROOT / eml_dir)
    return eml_dir


def decode_payload(part):
    """Decode an email part's payload based on its transfer encoding."""
    payload = part.get_payload(decode=False)
    encoding = part.get("Content-Transfer-Encoding", "").lower()

    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")

    if encoding == "base64":
        try:
            return base64.b64decode(payload).decode("utf-8", errors="replace")
        except Exception:
            return ""
    elif encoding == "quoted-printable":
        try:
            return quopri.decodestring(payload.encode()).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return payload if isinstance(payload, str) else ""


def html_to_text(html_content: str) -> str:
    """Convert HTML email body to clean plain text."""
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    return "\n".join(lines)


def decode_email_header(header_value):
    """Decode encoded email headers (e.g., UTF-8, ISO-8859)."""
    if header_value is None:
        return ""
    decoded_parts = decode_header(header_value)
    result = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(part)
    return " ".join(result)


def extract_body(msg) -> str:
    """Extract plain text body from an email message."""
    plain_text = ""
    html_text = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                text = decode_payload(part)
                if text.strip():
                    plain_text += text + "\n"
            elif content_type == "text/html":
                html = decode_payload(part)
                if html.strip():
                    html_text += html
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            plain_text = decode_payload(msg)
        elif content_type == "text/html":
            html_text = decode_payload(msg)

    if plain_text.strip():
        return plain_text.strip()
    elif html_text.strip():
        return html_to_text(html_text)
    return ""


def parse_eml_file(filepath: str) -> dict:
    """Parse a single .eml file and return structured data."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        msg = email.message_from_file(f, policy=policy.default)

    sender = decode_email_header(msg.get("From", ""))
    to = decode_email_header(msg.get("To", ""))
    cc = decode_email_header(msg.get("Cc", ""))
    subject = decode_email_header(msg.get("Subject", ""))
    date = decode_email_header(msg.get("Date", ""))

    body = extract_body(msg)

    attachments = []
    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append(decode_email_header(filename))

    return {
        "filename": os.path.basename(filepath),
        "from": sender,
        "to": to,
        "cc": cc,
        "subject": subject,
        "date": date,
        "body": body,
        "attachments": attachments,
    }


def parse_all_emails(emails_dir: str | None = None) -> list[dict]:
    """Parse all .eml files from the emails directory."""
    if emails_dir is None:
        emails_dir = _get_emails_dir()

    if not os.path.isdir(emails_dir):
        logger.warning("Emails directory does not exist: %s", emails_dir)
        return []

    emails = []
    for filename in os.listdir(emails_dir):
        if filename.lower().endswith(".eml"):
            filepath = os.path.join(emails_dir, filename)
            try:
                parsed = parse_eml_file(filepath)
                emails.append(parsed)
                logger.info("Parsed: %s (subject: %s)", filename, parsed["subject"][:60])
            except Exception as e:
                logger.error("Error parsing %s: %s", filename, e)

    logger.info("Total emails parsed: %d", len(emails))
    return emails


def email_to_text(email_data: dict) -> str:
    """Convert parsed email dict to a single text block for embedding."""
    parts = [
        f"Subject: {email_data['subject']}",
        f"From: {email_data['from']}",
        f"To: {email_data['to']}",
    ]
    if email_data["cc"]:
        parts.append(f"CC: {email_data['cc']}")
    parts.append(f"Date: {email_data['date']}")
    if email_data["attachments"]:
        parts.append(f"Attachments: {', '.join(email_data['attachments'])}")
    parts.append(f"\n{email_data['body']}")

    return "\n".join(parts)
