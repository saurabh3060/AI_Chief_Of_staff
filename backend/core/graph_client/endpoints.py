"""
core/graph_client/endpoints.py
=================================
Builds Microsoft Graph API URLs, aware of whether the backend is
targeting the signed-in user's own mailbox (`/me/...`) or a configured
shared mailbox (`/users/{MAILBOX_EMAIL}/...`), matching the pattern from
the user's reference code.

Kept separate from `client.py` so URL-construction logic (which is pure
and easy to unit test) doesn't get mixed with the HTTP call logic.

SSRF (CWE-918) note
-------------------
Every URL here is assembled from TWO kinds of input only:

- the single trusted base URL (`settings.graph_base_url`), which is read
  exclusively from application configuration and structurally validated
  at startup (see `core.config.Settings._validate_graph_base_url` /
  `core.graph_client.url_safety.validate_graph_base_url`). It is the sole
  source of the scheme/host/port, and it is NEVER taken from request
  input; and
- application-defined, hardcoded endpoint path templates (`/messages`,
  `/attachments`, ...).

Any DYNAMIC value interpolated into the path (message id, attachment id,
mailbox address) is passed through `safe_path_segment`, which validates
it as a single opaque segment and percent-encodes it, so a user-supplied
id can never change the host, scheme, port, origin, or path structure.
"""

import base64

from core.config import Settings
# All Graph URL-safety logic lives in ONE module: `core.graph_client.url_safety`.
# `graph_url` is a thin backwards-compatible shim over it; `safe_segment` and
# `graph_url()` below are its (shimmed) helpers, retained for the SharePoint/
# drive builders that join a settings-less base. Mail/calendar builders use
# the config-driven `_base_url(settings)` + `safe_path_segment` directly.
from core.graph_client.graph_url import graph_url, safe_segment
from core.graph_client.url_safety import safe_path_segment, validate_graph_base_url

# Backwards-compatible module constant: the default trusted base URL. The
# authoritative value at runtime is `settings.graph_base_url` (read from
# config and validated at startup); this constant is retained only so any
# code/tests that referenced `endpoints.GRAPH_BASE_URL` keep resolving to
# the same default. All builders below use the per-settings base, not this.
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

ME_FIELDS = (
    "id,displayName,givenName,surname,userPrincipalName,mail,jobTitle,"
    "officeLocation,mobilePhone,businessPhones,preferredLanguage,country"
)


def _base_url(settings: Settings) -> str:
    """
    Return the trusted, structurally-validated Graph base URL for this
    settings object (trailing slash stripped).

    `settings.graph_base_url` is already normalised + validated by the
    `Settings` model validator at construction time; re-running the cheap
    structured validation here is defence-in-depth so this builder is safe
    even if ever handed a settings object constructed by some other path.
    """
    return validate_graph_base_url(settings.graph_base_url)


def _mailbox_segment(settings: Settings) -> str:
    """
    Return the '/me' or '/users/{mailbox}' URL segment depending on whether
    a shared mailbox is configured.

    `mailbox_email` comes from trusted config, but it is still routed
    through `safe_path_segment` (defence-in-depth) so a malformed mailbox
    value can never inject path structure into the `/users/{...}` segment.
    """
    if settings.mailbox_email:
        return f"/users/{safe_path_segment(settings.mailbox_email, 'mailbox_email')}"
    return "/me"


def me_url(settings: Settings) -> str:
    """Build the URL for fetching the current user's profile (delegated mode only)."""
    return f"{_base_url(settings)}/me?$select={ME_FIELDS}"


def messages_url(settings: Settings) -> str:
    """Build the URL for listing inbox messages, targeting the shared mailbox if configured."""
    return f"{_base_url(settings)}{_mailbox_segment(settings)}/mailFolders/inbox/messages"


def message_detail_url(settings: Settings, message_id: str) -> str:
    """Build the URL for fetching a single message's full detail."""
    segment = safe_path_segment(message_id, "message_id")
    return f"{_base_url(settings)}{_mailbox_segment(settings)}/messages/{segment}"


def message_attachments_url(settings: Settings, message_id: str) -> str:
    """List a message's attachments."""
    segment = safe_path_segment(message_id, "message_id")
    return f"{_base_url(settings)}{_mailbox_segment(settings)}/messages/{segment}/attachments"


def attachment_detail_url(settings: Settings, message_id: str, attachment_id: str) -> str:
    """Build the URL for fetching (downloading) a single attachment's content."""
    message_segment = safe_path_segment(message_id, "message_id")
    attachment_segment = safe_path_segment(attachment_id, "attachment_id")
    return (
        f"{_base_url(settings)}{_mailbox_segment(settings)}"
        f"/messages/{message_segment}"
        f"/attachments/{attachment_segment}"
    )


def calendar_events_url(settings: Settings) -> str:
    """Build the URL for listing calendar events, targeting the shared mailbox if configured."""
    return f"{_base_url(settings)}{_mailbox_segment(settings)}/events"


def calendar_view_url(settings: Settings) -> str:
    """
    Build the URL for the `calendarView` endpoint, targeting the shared
    mailbox if configured. Unlike `/events` (which has no implicit date
    bound and, when ordered by start ascending, surfaces the oldest
    events in the whole mailbox history rather than anything current),
    `calendarView` takes an explicit `startDateTime`/`endDateTime`
    window and also expands recurring meeting instances - the correct
    way to ask Graph "what's on the calendar between X and Y".
    """
    # Config-driven base (honours a non-default GRAPH_BASE_URL), consistent
    # with the sibling mail/calendar builders above. `_mailbox_segment`
    # already routes the mailbox through `safe_path_segment`.
    return f"{_base_url(settings)}{_mailbox_segment(settings)}/calendarView"


# ── SharePoint / OneDrive sharing URL endpoints ──────────────
#
# These use Graph's "shares" API to resolve a sharing URL (from OneDrive
# or SharePoint) into a driveItem. This approach works with the
# Sites.Selected permission — no Files.Read.All needed — as long as the
# app has been granted access to the underlying site by an admin.
#
# Flow:
#   1. Base64-encode the sharing URL → "u!aHR0cHM6Ly9y..."
#   2. GET /shares/{encoded}/driveItem?$expand=children → driveId + itemId
#   3. GET /drives/{driveId}/items/{itemId}/children   → list subfolder
#   4. GET /drives/{driveId}/items/{itemId}/content     → download file


def encode_sharing_url(url: str) -> str:
    """Encode a SharePoint/OneDrive sharing URL into Graph's sharing token.

    Graph's ``/shares/{shareId}`` API requires the URL to be base64-encoded
    with padding stripped, ``+`` → ``-``, ``/`` → ``_``, prefixed with ``u!``.
    This is Graph's documented encoding for resolving any sharing URL to a
    driveItem.
    """
    b64 = base64.b64encode(url.encode("utf-8")).decode("utf-8")
    b64 = b64.rstrip("=").replace("/", "_").replace("+", "-")
    return f"u!{b64}"


def shares_driveitem_url(sharing_url: str) -> str:
    """Resolve a sharing URL to a driveItem via the shares API.

    Example: GET /shares/u!aHR0cHM6Ly9y.../driveItem
    """
    share_id = encode_sharing_url(sharing_url)
    # share_id is a base64-encoded token (alphanumeric + !_-), not
    # arbitrary user input, so safe_segment is appropriate here.
    sid = safe_segment(share_id, "share_id")
    return graph_url(f"/shares/{sid}/driveItem")


def drive_item_children_url(drive_id: str, item_id: str) -> str:
    """List children of a drive item by driveId + itemId.

    Example: GET /drives/{driveId}/items/{itemId}/children
    """
    did = safe_segment(drive_id, "drive_id")
    iid = safe_segment(item_id, "item_id")
    return graph_url(f"/drives/{did}/items/{iid}/children")


def drive_item_by_id_url(drive_id: str, item_id: str) -> str:
    """Get metadata for a specific drive item by driveId + itemId."""
    did = safe_segment(drive_id, "drive_id")
    iid = safe_segment(item_id, "item_id")
    return graph_url(f"/drives/{did}/items/{iid}")


def drive_item_content_by_id_url(drive_id: str, item_id: str) -> str:
    """Download the content of a drive item by driveId + itemId.

    Graph returns a 302 redirect to a pre-authenticated CDN URL.
    """
    did = safe_segment(drive_id, "drive_id")
    iid = safe_segment(item_id, "item_id")
    return graph_url(f"/drives/{did}/items/{iid}/content")
