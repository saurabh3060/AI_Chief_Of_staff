"""
core/exceptions.py
====================
Shared exception types used across `core/` clients and agents.

Having a small, explicit exception hierarchy lets `core.retry.with_retry`
(and calling code) distinguish "this is a transient network problem,
worth retrying" from "this is a real application error, don't retry"
without inspecting library-specific exception types everywhere.
"""


class BackendError(Exception):
    """Base class for all custom exceptions raised by this backend."""


class GraphApiError(BackendError):
    """
    Raised when a Microsoft Graph API call fails after all retries.

    Carries the HTTP ``status_code`` (when the failure came from an actual
    HTTP response rather than a transport error) so callers can reason
    about the failure *structurally* instead of parsing the message text.
    ``status_code`` is ``None`` for failures that never produced a response
    (connection errors, timeouts) or where no status was supplied.

    IMPORTANT for deletion detection: a missing message is signalled ONLY
    by a verified HTTP 404 (see ``GraphNotFoundError``). Every other
    outcome - 403 (permission), 429/503 (throttle), 5xx, timeouts, and
    exhausted retries - is INCONCLUSIVE and must never be treated as proof
    that a message was deleted.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GraphNotFoundError(GraphApiError):
    """
    Raised specifically for a verified HTTP 404 from Graph API: the
    requested resource (e.g. a single message by id) does not exist at the
    queried mailbox scope.

    This is the ONLY Graph outcome that may be treated as a *deletion
    candidate*. It is kept as its own type - rather than a plain
    ``GraphApiError`` - so a caller can catch it explicitly and distinguish
    "verified absent" from every inconclusive failure. It is deliberately
    NOT in ``GraphClient._get``'s retry set: a 404 is a definitive answer,
    not a transient error, so retrying it would be pointless.

    Note: a 404 proves the resource is absent *from the queried scope*. It
    does not by itself distinguish a deletion from a move/archive out of
    that scope - that disambiguation is the caller's responsibility and is
    out of scope for this type.
    """

    def __init__(self, message: str, status_code: int = 404) -> None:
        super().__init__(message, status_code=status_code)


class GraphAuthError(GraphApiError):
    """
    Raised specifically for a 401 (Unauthorized) response from Graph API,
    i.e. the delegated access token is expired/invalid/revoked.

    Kept as its own type - rather than a plain ``GraphApiError`` - so
    ``GraphClient._get`` can treat it as a one-time, refreshable condition:
    when the client was given a refresh-capable token provider it mints a
    fresh access token from the stored refresh token and retries the call
    ONCE, instead of failing the whole (possibly long-running) pipeline. If
    no provider is configured, or the single refresh+retry still returns
    401, the error propagates like any other ``GraphApiError``.

    It is deliberately NOT handled by ``core.retry.with_retry``'s generic
    backoff loop: a 401 is not a transient network blip, so blind backoff
    would just resend the same dead token. The refresh+retry is an explicit,
    single, token-swapping attempt handled inside ``_get`` itself.
    """

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message, status_code=status_code)


class GraphThrottledError(GraphApiError):
    """
    Raised specifically for a 429 (or 503) throttling response from Graph
    API - e.g. the shared mailbox's own concurrency limit
    ("ApplicationThrottled" / "Application is over its MailboxConcurrency
    limit"), which is separate from and can be stricter than this
    process's own `MAX_CONCURRENT_GRAPH_CALLS` semaphore. Kept as its own
    type (rather than a plain `GraphApiError`) so `GraphClient._get` can
    retry it with the server-provided `Retry-After` backoff instead of
    treating it as a permanent failure.
    """

    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after_seconds = retry_after_seconds


class OllamaError(BackendError):
    """Raised when a call to the local Ollama server fails after all retries."""


class StorageError(BackendError):
    """Raised when a storage backend operation fails."""


class AgentNotImplementedError(BackendError):
    """
    Raised by stub agents (e.g. MIS Insight) whose routes are registered
    but whose business logic has not been built yet.
    """
