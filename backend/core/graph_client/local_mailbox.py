"""
core/graph_client/local_mailbox.py
=====================================
Offline stand-in for Microsoft Graph mail endpoints.

When `GRAPH_AUTH_MODE=local`, `GraphClient` serves mailbox data from
`.eml` files on disk (default: `backend/temp_mail_data`) instead of
calling graph.microsoft.com. Messages and attachments are shaped like
Graph JSON so the rest of the email pipeline (`_parse_message`,
attachment extractors, BU CSV writer) stays unchanged.

Calendar endpoints are served separately from `local_calendar.py`.
"""

from __future__ import annotations

import base64
import email
import email.policy
import hashlib
import re
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger, sanitize_log_value
from core.text_match import html_to_text

logger = get_agent_logger("local_mailbox")

_SUBJECT_PREFIX_RE = re.compile(r"^(re|fw|fwd)\s*:\s*", re.IGNORECASE)


def _normalize_subject(subject: str) -> str:
    normalized = (subject or "").strip()
    while True:
        next_value = _SUBJECT_PREFIX_RE.sub("", normalized).strip()
        if next_value == normalized:
            return normalized.lower()
        normalized = next_value


def _stable_id(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:32]


def _parse_addresses(header_value: str | None) -> list[dict]:
    return [
        {"emailAddress": {"name": name, "address": address}}
        for name, address in getaddresses([header_value or ""])
        if address
    ]


def _parse_received_at(raw_date: str | None) -> datetime:
    if raw_date:
        try:
            parsed = parsedate_to_datetime(raw_date)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (TypeError, ValueError, OverflowError):
            pass
    return datetime.now(timezone.utc)


def _extract_body(msg: email.message.Message) -> tuple[str, str]:
    """Return (content_type, body_text) preferring text/plain."""
    if not msg.is_multipart():
        content_type = msg.get_content_type()
        try:
            payload = msg.get_content()
        except (KeyError, LookupError):
            payload = ""
        text = payload if isinstance(payload, str) else (str(payload) if payload else "")
        if content_type == "text/html":
            return "html", html_to_text(text)
        return "text", text

    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
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
        return "text", "\n".join(plain_parts)
    if html_parts:
        return "html", html_to_text("\n".join(html_parts))
    return "text", ""


def _extract_attachments(msg: email.message.Message, message_id: str) -> list[dict]:
    attachments: list[dict] = []
    index = 0
    for part in msg.walk():
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        content_type = part.get_content_type()
        is_attachment = disposition == "attachment" or (
            bool(filename) and content_type not in {"text/plain", "text/html", "multipart/mixed", "multipart/related", "multipart/alternative"}
        )
        if not is_attachment:
            continue
        try:
            payload = part.get_payload(decode=True) or b""
        except (KeyError, LookupError, TypeError, ValueError):
            payload = b""
        if not payload:
            continue
        index += 1
        name = filename or f"attachment-{index}"
        attachments.append(
            {
                "id": f"{message_id}-att-{index}",
                "name": name,
                "contentType": content_type or "application/octet-stream",
                "size": len(payload),
                "contentBytes": base64.b64encode(payload).decode("ascii"),
            }
        )
    return attachments


def _eml_to_graph_message(filepath: Path) -> tuple[dict, list[dict]]:
    with open(filepath, "rb") as handle:
        msg = email.message_from_binary_file(handle, policy=email.policy.default)

    raw_message_id = (msg.get("Message-ID") or filepath.name).strip()
    message_id = _stable_id(raw_message_id)
    subject = msg.get("Subject") or "(no subject)"
    thread_key = msg.get("Thread-Topic") or subject
    conversation_id = _stable_id(_normalize_subject(thread_key) or filepath.stem)
    received_at = _parse_received_at(msg.get("Date"))
    from_name, from_address = parseaddr(msg.get("From") or "")
    content_type, body_text = _extract_body(msg)
    attachments = _extract_attachments(msg, message_id)
    preview = re.sub(r"\s+", " ", body_text).strip()[:240]

    graph_message = {
        "id": message_id,
        "conversationId": conversation_id,
        "subject": subject,
        "from": {"emailAddress": {"name": from_name, "address": from_address}},
        "toRecipients": _parse_addresses(msg.get("To")),
        "ccRecipients": _parse_addresses(msg.get("Cc")),
        "receivedDateTime": received_at.isoformat().replace("+00:00", "Z"),
        "hasAttachments": bool(attachments),
        "bodyPreview": preview,
        "body": {"contentType": content_type, "content": body_text},
        "isRead": True,
        "_localFilename": filepath.name,
    }
    return graph_message, attachments


class LocalMailbox:
    """In-memory Graph-shaped mailbox built from `.eml` files."""

    def __init__(self, messages: list[dict], attachments: dict[str, list[dict]]) -> None:
        self.messages = messages
        self._attachments = attachments
        self._by_id = {message["id"]: message for message in messages}

    def list_messages(self, top: int = 10, skip: int = 0, received_after: str | None = None) -> dict:
        items = list(self.messages)
        if received_after:
            cutoff = received_after.replace("Z", "+00:00")
            try:
                cutoff_dt = datetime.fromisoformat(cutoff)
            except ValueError:
                cutoff_dt = None
            if cutoff_dt is not None:
                if cutoff_dt.tzinfo is None:
                    cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)
                filtered = []
                for message in items:
                    received = datetime.fromisoformat(message["receivedDateTime"].replace("Z", "+00:00"))
                    if received >= cutoff_dt:
                        filtered.append(message)
                items = filtered
        return {"value": items[skip : skip + top]}

    def get_message(self, message_id: str) -> dict:
        message = self._by_id.get(message_id)
        if message is None:
            raise KeyError(message_id)
        return message

    def list_attachments(self, message_id: str) -> dict:
        return {
            "value": [
                {key: value for key, value in attachment.items() if key != "contentBytes"}
                for attachment in self._attachments.get(message_id, [])
            ]
        }

    def get_attachment(self, message_id: str, attachment_id: str) -> dict:
        for attachment in self._attachments.get(message_id, []):
            if attachment["id"] == attachment_id:
                return attachment
        raise KeyError(attachment_id)


_mailbox_cache: tuple[str, float, LocalMailbox] | None = None


def _dir_signature(eml_dir: Path) -> float:
    if not eml_dir.is_dir():
        return 0.0
    mtimes = [eml_dir.stat().st_mtime]
    mtimes.extend(path.stat().st_mtime for path in eml_dir.glob("*.eml"))
    return max(mtimes) if mtimes else 0.0


def load_local_mailbox(settings: Settings | None = None) -> LocalMailbox:
    """
    Load (and cache) the local `.eml` mailbox. The cache invalidates when
    any file in the folder changes so newly dropped Outlook exports show
    up without a process restart.
    """
    global _mailbox_cache
    active = settings or get_settings()
    eml_dir = active.resolved_eml_storage_dir
    signature = _dir_signature(eml_dir)
    cache_key = str(eml_dir.resolve())
    if _mailbox_cache and _mailbox_cache[0] == cache_key and _mailbox_cache[1] == signature:
        return _mailbox_cache[2]

    messages: list[dict] = []
    attachments: dict[str, list[dict]] = {}
    if eml_dir.is_dir():
        for filepath in sorted(eml_dir.glob("*.eml")):
            try:
                message, message_attachments = _eml_to_graph_message(filepath)
            except Exception as exc:  # noqa: BLE001 - one bad file must not empty the mailbox
                logger.warning(
                    "Skipping unreadable local eml '%s': %s",
                    sanitize_log_value(filepath.name),
                    exc,
                )
                continue
            messages.append(message)
            attachments[message["id"]] = message_attachments

    messages.sort(key=lambda item: item["receivedDateTime"], reverse=True)
    mailbox = LocalMailbox(messages, attachments)
    _mailbox_cache = (cache_key, signature, mailbox)
    logger.info(
        "Loaded local mailbox from %s (%d message(s), %d with attachments)",
        eml_dir,
        len(messages),
        sum(1 for message in messages if message.get("hasAttachments")),
    )
    return mailbox
