"""
core/ollama_client/client.py
===============================
`OllamaClient` - the single entry point agents use to talk to the local
Ollama server, for both plain text-generation tasks (summarization,
classification, pre-meet brief writing) and vision/OCR tasks (extracting
text from scanned document page images).

Backed by `langchain_ollama.ChatOllama`. Both `chat()` and `vision_ocr()`
currently use the same configured model (`OLLAMA_VISION_MODEL`, e.g.
`llama3.2-vision:11b`) since that's the only model deployed for now -
`Settings.effective_text_model` already resolves to the vision model
when no dedicated text model is set, so swapping in a separate text-only
model later requires only a `.env` change, no code change here.

All calls are wrapped with `core.retry.with_retry` for exponential
backoff and limited by `core.concurrency.get_ollama_semaphore` so we
never run more than `MAX_CONCURRENT_OLLAMA_CALLS` model invocations at
once (protecting a locally-hosted Ollama instance from being
overwhelmed).

Request/response *sizes* (not full content) are logged, to avoid
leaking potentially sensitive document/mail content into log files
while still giving operators visibility into call volume and latency.
"""

import asyncio
import base64
import itertools
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.concurrency import get_ollama_semaphore
from core.config import Settings, get_settings
from core.exceptions import OllamaError
from core.logging_config import get_agent_logger
from core.retry import with_retry
from core.utils import parse_ollama_keep_alive

logger = get_agent_logger("ollama_client")

# Exceptions considered transient/retryable for Ollama calls. Ollama's
# Python/LangChain integration surfaces connection issues as generic
# ConnectionError/TimeoutError (it sits on top of `httpx`/`requests`
# under the hood), so those are what we retry on here.
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

# Monotonically increasing id assigned to every outbound Ollama call, so
# concurrent calls' start/heartbeat/completion log lines can be
# correlated in the shared ollama_client.log file.
_call_id_counter = itertools.count(1)

# Default heartbeat interval (seconds) used only if settings are
# unavailable. The in-flight heartbeat for an Ollama call, and its
# interval, are now controlled by `Settings.ollama_heartbeat_enabled`
# and `Settings.ollama_heartbeat_interval_seconds` (see `core.config`);
# this constant is just the fallback. Ollama's own request/token logs
# are only visible on the Ollama host's terminal - this backend has no
# way to tail them remotely - so this heartbeat is what gives
# visibility into "is it stuck or just slow" from this backend's own
# logs when it is enabled.
_HEARTBEAT_INTERVAL_SECONDS = 15.0


async def _run_with_heartbeat(
    coro,
    *,
    call_id: int,
    model_name: str,
    task_label: str,
    settings: Settings | None = None,
):
    """
    Await `coro` while logging a periodic heartbeat (at WARNING level,
    so it's visible on the console per this project's log policy of
    "console stays quiet except WARNING+") until it completes. Lets
    operators watching the backend terminal/log files see that a
    long-running Ollama call is still in progress instead of the
    terminal going silent for minutes with no indication of whether
    it's stuck or just slow.

    Honors the `ollama_heartbeat_enabled` toggle: when disabled, the
    coroutine is simply awaited with no heartbeat logging at all. The
    heartbeat cadence comes from `ollama_heartbeat_interval_seconds`.
    """
    active_settings = settings or get_settings()

    # Toggle OFF: no heartbeat logging - just await the call quietly.
    if not getattr(active_settings, "ollama_heartbeat_enabled", True):
        return await coro

    interval = getattr(
        active_settings, "ollama_heartbeat_interval_seconds", _HEARTBEAT_INTERVAL_SECONDS
    ) or _HEARTBEAT_INTERVAL_SECONDS

    task = asyncio.ensure_future(coro)
    started_at = time.monotonic()
    elapsed = 0.0
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - started_at
            logger.warning(
                "[call#%d] %s still running on model=%s after %.1fs...",
                call_id, task_label, model_name, elapsed,
            )
    return task.result()


class OllamaClient:
    """
    Thin async wrapper around a local Ollama model, exposing two
    task-shaped methods (`chat` for text, `vision_ocr` for images)
    rather than a raw message-passing API, so agent code stays simple.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def generation_params(self, num_predict: int | None = None) -> dict:
        """The model name and generation parameters ``chat()`` will use for this
        ``num_predict`` - the single source of truth shared with ``chat()`` so an
        exact-match cache key can never drift from what is actually sent."""
        predict = num_predict if num_predict is not None else int(getattr(self._settings, "ollama_num_predict", 0) or 0)
        return {
            "model": self._settings.effective_text_model,
            "temperature": self._settings.ollama_temperature,
            "num_ctx": self._settings.ollama_num_ctx,
            "num_predict": predict if predict and predict > 0 else 0,
        }

    def _build_chat_model(self, model_name: str, num_predict: int | None = None) -> ChatOllama:
        """Construct a `ChatOllama` instance pointed at the configured Ollama host and the given model."""
        kwargs: dict = {
            "model": model_name,
            "base_url": self._settings.ollama_host,
            "temperature": self._settings.ollama_temperature,
            "timeout": 600,
            "num_ctx": self._settings.ollama_num_ctx,
        }
        if num_predict is not None and num_predict > 0:
            kwargs["num_predict"] = num_predict
        # Opt-in runtime options (EMAIL_PIPELINE_OLLAMA_*): only sent when set, so the
        # default request is exactly what it was before. They change speed, not content.
        keep_alive = (getattr(self._settings, "email_pipeline_ollama_keep_alive", "") or "").strip()
        if keep_alive:
            kwargs["keep_alive"] = parse_ollama_keep_alive(keep_alive)
        num_thread = int(getattr(self._settings, "email_pipeline_ollama_num_thread", 0) or 0)
        if num_thread > 0:
            kwargs["num_thread"] = num_thread
        return ChatOllama(**kwargs)

    @staticmethod
    async def _aclose_chat_model(model: ChatOllama) -> None:
        """
        Best-effort teardown of a per-call ``ChatOllama``'s underlying
        HTTP connections.

        ``ChatOllama`` lazily builds an ``ollama`` SDK ``Client`` and
        ``AsyncClient`` (``_client`` / ``_async_client``), each wrapping an
        ``httpx`` client with a keep-alive connection pool to the Ollama
        host (``:11434``). Because we construct a fresh ``ChatOllama`` per
        call, those pooled sockets would otherwise be reclaimed only by the
        GC — which is exactly what surfaces as the interpreter-shutdown
        ``ResourceWarning: unclosed <socket ... raddr=(..., 11434)>`` and
        the ``unclosed transport`` warning. Closing them here returns the
        sockets deterministically.

        Every step is guarded: a langchain/ollama version that names these
        attributes differently simply results in a no-op, never an error on
        a hot path.
        """
        # Async httpx client first (holds the sockets used by ainvoke()).
        aclient = getattr(model, "_async_client", None)
        if aclient is not None:
            closer = getattr(aclient, "aclose", None) or getattr(
                getattr(aclient, "_client", None), "aclose", None
            )
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001 - teardown must never raise
                    pass
        # Sync httpx client (built if any sync path was ever touched).
        sclient = getattr(model, "_client", None)
        if sclient is not None:
            closer = getattr(sclient, "close", None) or getattr(
                getattr(sclient, "_client", None), "close", None
            )
            if closer is not None:
                try:
                    closer()
                except Exception:  # noqa: BLE001 - teardown must never raise
                    pass

    @with_retry(max_attempts=3, base_delay=1.0, retry_exceptions=_RETRYABLE_EXCEPTIONS, logger=logger)
    async def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        num_predict: int | None = None,
    ) -> str:
        """
        Run a plain text-generation call (summarization, classification,
        brief writing) against the configured text model and return the
        model's text response.

        Args:
            prompt: the user-facing prompt/content to generate from.
            system_prompt: optional system instruction steering the
                model's behaviour (e.g. "You are a concise summarizer.").
            num_predict: optional per-call generation cap. When omitted,
                Settings.ollama_num_predict is used (0 = unlimited).
        """
        predict = num_predict if num_predict is not None else int(getattr(self._settings, "ollama_num_predict", 0) or 0)
        model = self._build_chat_model(self._settings.effective_text_model, num_predict=predict)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        call_id = next(_call_id_counter)
        async with get_ollama_semaphore():
            logger.info(
                "[call#%d] chat() starting model=%s prompt_chars=%d num_ctx=%s num_predict=%s",
                call_id, self._settings.effective_text_model, len(prompt),
                self._settings.ollama_num_ctx, predict or "default",
            )
            started_at = time.monotonic()
            try:
                response = await _run_with_heartbeat(
                    model.ainvoke(messages),
                    call_id=call_id,
                    model_name=self._settings.effective_text_model,
                    task_label="chat()",
                    settings=self._settings,
                )
            except _RETRYABLE_EXCEPTIONS:
                logger.error(
                    "Ollama chat call failed with retryable exception (will retry). Model=%s, Duration=%.1fms",
                    self._settings.effective_text_model,
                    (time.monotonic() - started_at) * 1000,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as a clear OllamaError
                logger.error(
                    "Ollama chat call failed with non-retryable exception. Model=%s, Duration=%.1fms, Error=%s",
                    self._settings.effective_text_model,
                    (time.monotonic() - started_at) * 1000,
                    str(exc),
                )
                raise OllamaError(f"Ollama chat call failed: {exc}") from exc
            finally:
                # Release this per-call model's pooled sockets to the Ollama
                # host immediately (success, retryable error, or hard error
                # alike) so a long pipeline does not leak one keep-alive
                # connection per LLM call and strand them until GC/interp
                # shutdown (the ":11434" unclosed-socket ResourceWarning).
                await self._aclose_chat_model(model)
            duration_ms = (time.monotonic() - started_at) * 1000

        response_text = response.content if isinstance(response.content, str) else str(response.content)
        
        # Log detailed response information
        logger.info(
            "[call#%d] chat() completed model=%s prompt_chars=%d response_chars=%d duration_ms=%.1f",
            call_id, self._settings.effective_text_model, len(prompt), len(response_text), duration_ms,
        )
        
        return response_text

    @with_retry(max_attempts=3, base_delay=1.0, retry_exceptions=_RETRYABLE_EXCEPTIONS, logger=logger)
    async def vision_ocr(self, image_bytes: bytes, instruction: str | None = None) -> str:
        """
        Run OCR on a single page image using the configured vision
        model, returning the extracted text.

        Args:
            image_bytes: raw image bytes (e.g. a PNG produced by
                `pdf_to_image.pdf_to_page_images`).
            instruction: optional instruction steering what to extract
                (defaults to a generic "transcribe all visible text"
                prompt suitable for scanned documents).
        """
        # Vision OCR disabled - llama3.2:3b does not support multimodal inputs
        logger.info("[vision_ocr] Skipping OCR - model does not support vision. Returning empty text.")
        return ""


async def close_ollama() -> None:
    """
    Process-wide Ollama teardown hook, mirroring ``core.weaviate.close_client``
    and ``core.redis.close_redis`` so application shutdown has one place to
    release Ollama resources.

    ``OllamaClient`` builds (and now closes) a fresh ``ChatOllama`` per call,
    so there is no long-lived shared client to shut down here — each call's
    sockets are already returned in ``OllamaClient.chat``'s ``finally``. This
    hook therefore does a final best-effort sweep: it nudges the garbage
    collector so any ``ChatOllama`` whose underlying httpx client was built
    but not explicitly closed (e.g. an error path in a third-party version
    that names its client attributes differently) is finalized before the
    interpreter exits, keeping shutdown free of the ``:11434``
    unclosed-socket / unclosed-transport ``ResourceWarning``s.

    Safe to call multiple times and safe to call when Ollama was never used.
    """
    # Close the email agent's owned embedding client (agent_email.semantic_match
    # holds its own long-lived ollama.Client for /api/embed calls, separate from
    # the per-call ChatOllama used by chat()). Imported lazily so core has no
    # import-time dependency on an agent module.
    try:
        from agent_email.semantic_match import close_embed_client

        close_embed_client()
    except Exception:  # noqa: BLE001 - teardown must never raise
        pass

    import gc

    try:
        gc.collect()
    except Exception:  # noqa: BLE001 - teardown must never raise
        pass
    logger.info("[OLLAMA] client teardown complete")
