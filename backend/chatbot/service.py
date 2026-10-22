"""
chatbot/service.py
======================
LLM-powered chatbot that answers questions using retrieved email context.

Ported from the standalone chatbot_final project's chatbot.py, adapted to:
  - read tuning knobs (LLM model, max tokens, context window) from
    core.config.get_settings() instead of a module-local config.py
  - use this project's centralised logger
  - expose `answer_question()` / `generate_response_stream()` with the
    signatures expected by chatbot/routes.py (bearer_token kept for
    backward compatibility, wv_client/bm25_index passed through from
    the module-level chatbot state)
"""

import ast
import asyncio
import hashlib
import json
import math
import operator
import queue
import re
import threading
import time
from pathlib import Path
from typing import Generator

import ollama

from chatbot.postgres_store import save_chat_message
from chatbot.redis_history import save_message
from chatbot.redis_qa_cache import get_cached_answer, set_cached_answer
from chatbot.retriever import retrieve_weaviate
from chatbot.schema import ChatResponse, SourceInfo, TimingInfo
from core.config import BACKEND_ROOT, get_settings
from core.db import get_main_loop
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")


def _resolve_prompt_path(raw: str) -> Path:
    """Relative prompt paths are resolved against the backend root."""
    path = Path(raw)
    return path if path.is_absolute() else BACKEND_ROOT / path


_prompt_cache: dict = {"path": None, "mtime": None, "text": ""}
_calc_prompt_cache: dict = {"path": None, "mtime": None, "text": ""}


def _read_prompt_file(raw_path: str, cache: dict, label: str) -> str:
    """Return the text of a prompt file, re-reading it when its mtime changes.

    Raises RuntimeError when the file is missing or empty so a bad deployment
    fails loudly instead of silently answering with a different prompt. Logs
    path, size and a short hash on each (re)load so the prompt a running server
    actually uses can be confirmed.
    """
    path = _resolve_prompt_path(raw_path)
    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        raise RuntimeError(f"{label} file not readable: {path} ({e})") from e

    if cache["path"] == path and cache["mtime"] == mtime:
        return cache["text"]

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{label} file is empty: {path}")

    cache.update(path=path, mtime=mtime, text=text)
    logger.info(
        "%s loaded: path=%s chars=%d sha256=%s",
        label, path, len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
    return text


def get_system_prompt() -> str:
    """Return the chatbot system prompt from CHATBOT_SYSTEM_PROMPT_PATH."""
    return _read_prompt_file(get_settings().chatbot_system_prompt_path, _prompt_cache, "Chatbot system prompt")


def get_calc_prompt() -> str:
    """Return the instruction for the calculation call from CHATBOT_CALC_PROMPT_PATH."""
    return _read_prompt_file(get_settings().chatbot_calc_prompt_path, _calc_prompt_cache, "Chatbot calc prompt")


def _chat_request(
    user_message: str,
    model: str,
    *,
    system_prompt: str | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
) -> dict:
    """Keyword arguments for ``ollama.Client.chat`` - the single place that
    decides messages, options and keep_alive for the streaming, non-streaming
    and calculation calls (``model`` comes from ``_select_model``). Every value
    comes from Settings / the prompt files; the three keyword arguments let the
    calculation call override the system prompt, token cap and temperature."""
    settings = get_settings()
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": get_system_prompt() if system_prompt is None else system_prompt},
            {"role": "user", "content": user_message},
        ],
        "options": {
            "num_predict": settings.chatbot_max_tokens if num_predict is None else num_predict,
            "temperature": settings.chatbot_temperature if temperature is None else temperature,
            "repeat_penalty": settings.chatbot_repeat_penalty,
            "num_ctx": settings.chatbot_llm_num_ctx,
            "num_thread": settings.chatbot_num_thread,
        },
        "keep_alive": settings.chatbot_keep_alive,
    }


GREETING_RESPONSE = (
    "Hi! I'm your Meeting, Email & Document Assistant. I can dig through your meeting summaries, "
    "email discussions, and uploaded documents.\n\n"
    "Ask me anything about your meetings, emails, or documents!"
)

NOT_FOUND_RESPONSE = "Not found in the provided emails."

# Sent to the client in an SSE "error" event; exception details stay in the server log.
STREAM_ERROR_MESSAGE = "An internal error occurred. Please try again."

def _matches_pattern(pattern: str, setting_name: str, query: str) -> bool:
    """Case-insensitive regex match of a question; an invalid pattern logs a
    WARNING and counts as no match."""
    try:
        return re.search(pattern, query, re.IGNORECASE) is not None
    except re.error as e:
        logger.warning("Invalid %s %r: %s", setting_name, pattern, e)
        return False


def _wants_number(query: str) -> bool:
    """True when the question asks for a figure (CHATBOT_NUMERIC_QUESTION_PATTERN)."""
    return _matches_pattern(get_settings().chatbot_numeric_question_pattern, "CHATBOT_NUMERIC_QUESTION_PATTERN", query)


def _wants_calc(query: str) -> bool:
    """True when the question asks for a calculation (CHATBOT_CALC_QUESTION_PATTERN)."""
    return _matches_pattern(get_settings().chatbot_calc_question_pattern, "CHATBOT_CALC_QUESTION_PATTERN", query)


def _select_model(query: str) -> str:
    """Pick the chat model for this question.

    CHATBOT_MODEL_ROUTING off (default): always CHATBOT_LLM_MODEL, as before.
    On: the accurate model for questions that ask for a number or a calculation,
    the fast model for everything else.
    """
    settings = get_settings()
    if not settings.chatbot_model_routing:
        return settings.chatbot_llm_model
    if _wants_number(query) or _wants_calc(query):
        return settings.chatbot_llm_model_accurate
    return settings.chatbot_llm_model_fast


def _build_user_message(context: str, query: str) -> str:
    """Assemble the user turn sent to the LLM from context + question."""
    return (
        f"Context from meeting summaries and emails:\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer only from the context above."
    )


# ── Arithmetic in code (CHATBOT_CALC_IN_CODE) ─────────────────
# The small model is unreliable at arithmetic and unit conversion. When the
# switch is on, a short extra call asks the accurate model for ONE arithmetic
# expression, Python evaluates it safely, and the result is appended to the
# prompt. Spec: temperature 0 for that call.
_CALC_TEMPERATURE = 0.0
_CALC_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_CALC_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression without eval/exec.

    Only numbers, + - * / **, a leading sign and parentheses are allowed; any
    other syntax raises ValueError. Every number is handled as a float, so a
    huge power overflows (OverflowError) instead of building an enormous
    integer. The result must be a finite real number.
    """
    def ev(node) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINARY_OPS:
            return _CALC_BINARY_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARY_OPS:
            return _CALC_UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"not allowed in a calculation: {type(node).__name__}")

    value = ev(ast.parse(expression.strip(), mode="eval"))
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("result is not a finite real number")
    return value


def _clean_expression(raw: str) -> str:
    """First line of the model reply, without code fences or thousands
    separators (a comma between two digits)."""
    text = (raw or "").strip().strip("`").strip()
    first_line = text.splitlines()[0] if text else ""
    return re.sub(r"(?<=\d),(?=\d)", "", first_line).strip()


def _format_number(value: float) -> str:
    """Display form: at most 2 decimals, no trailing zeros."""
    return f"{round(value, 2):.2f}".rstrip("0").rstrip(".")


def _compute_in_code(context: str, query: str) -> dict:
    """One short extra call for an arithmetic expression, then a safe evaluation.

    Never raises: every failure (call error, timeout, no/invalid expression)
    comes back in ``error`` with ``used`` False. The call uses the accurate
    model, temperature 0, CHATBOT_CALC_MAX_TOKENS and CHATBOT_CALC_TIMEOUT_S.
    """
    settings = get_settings()
    started = time.perf_counter()
    result = {"attempted": True, "used": False, "expression": None, "value": None, "error": None, "seconds": 0.0}
    try:
        client = ollama.Client(host=settings.ollama_host, timeout=settings.chatbot_calc_timeout_s)
        request = _chat_request(
            _build_user_message(context, query),
            settings.chatbot_llm_model_accurate,
            system_prompt=get_calc_prompt(),
            num_predict=settings.chatbot_calc_max_tokens,
            temperature=_CALC_TEMPERATURE,
        )
        expression = _clean_expression(client.chat(**request)["message"]["content"])
        if not expression or expression.upper() == "NONE":
            raise ValueError("the model returned no expression")
        result.update(used=True, expression=expression, value=_safe_eval(expression))
    except Exception as e:  # any failure must leave the normal path untouched
        result["error"] = f"{type(e).__name__}: {e}"
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def _apply_code_calculation(context: str, query: str, user_message: str, stats: dict) -> str:
    """Append the code-computed result to the user message when CHATBOT_CALC_IN_CODE
    is on and the question matches CHATBOT_CALC_QUESTION_PATTERN; otherwise return
    it unchanged. Called from _build_prompt_with_guard, so the streaming and
    non-streaming paths share it. ``stats["calc"]`` records what happened for the
    debug dump."""
    settings = get_settings()
    calc = {
        "enabled": settings.chatbot_calc_in_code, "attempted": False, "used": False,
        "expression": None, "value": None, "error": None, "seconds": 0.0,
    }
    stats["calc"] = calc
    if not settings.chatbot_calc_in_code or not _wants_calc(query):
        return user_message
    if stats.get("threads_used", 0) == 0:
        calc["error"] = "no context to calculate from"
        return user_message

    calc.update(_compute_in_code(context, query))
    if calc["used"]:
        shown = _format_number(calc["value"])
        logger.info(
            "Calc: expression=%r value=%s extra_seconds=%.2f", calc["expression"], shown, calc["seconds"],
        )
        return f"{user_message}\n\nComputed result: {calc['expression']} = {shown}. Use this value; do not recompute."
    logger.warning(
        "Calc skipped (%s) after %.2fs - continuing with the normal path", calc["error"], calc["seconds"],
    )
    return user_message


def _build_prompt_with_guard(retrieved: list[dict], query: str) -> tuple[str, str, dict]:
    """Return (context, user_message, stats), rebuilding once with a smaller
    budget if the assembled prompt exceeds CHATBOT_PROMPT_CHAR_CEILING.

    Never depends on Ollama truncation: if the first build is too large we
    rebuild the context with a reduced ``budget_chars`` a single time. With
    CHATBOT_CALC_IN_CODE on, a calculation question also gets its code-computed
    result appended (see _apply_code_calculation).
    ``stats`` carries build_context's diagnostics (mode, threads and blocks
    kept/total, characters before/after cleaning and sent, cuts) and the
    calculation outcome for the per-request logs and the debug dump.

    When CHATBOT_DEBUG_PROMPT is on, logs which threads were retrieved
    (subject + rank) and how many blocks build_context kept per thread.
    """
    from chatbot.retriever import ContextConfig, build_context

    settings = get_settings()
    cfg = ContextConfig.from_settings(settings)
    stats: dict = {}
    context = build_context(retrieved, query, cfg=cfg, stats=stats)
    user_message = _build_user_message(context, query)

    if len(user_message) > settings.chatbot_prompt_char_ceiling:
        shrunk = max(1, int(settings.chatbot_context_max_chars * settings.chatbot_prompt_shrink_factor))
        logger.warning(
            "Built prompt %d chars > ceiling %d - rebuilding context once with budget_chars=%d",
            len(user_message), settings.chatbot_prompt_char_ceiling, shrunk,
        )
        stats = {}
        context = build_context(retrieved, query, cfg=cfg, budget_chars=shrunk, stats=stats)
        user_message = _build_user_message(context, query)

    stats["context_chars"] = len(context)
    logger.info(
        "Context: mode=%s threads=%d/%d blocks=%d/%d chars raw=%d cleaned=%d sent=%d "
        "dedup_removed=%d embeddings=%s cuts=%d",
        "everything fit" if stats["mode"] == "everything_fit" else "selection used",
        stats["threads_used"], stats["threads_in"], stats["blocks_kept"], stats["blocks_total"],
        stats["chars_raw"], stats["chars_cleaned"], stats["chars_sent"],
        stats["dedup_removed"], stats["embeddings"], len(stats["cuts"]),
    )
    user_message = _apply_code_calculation(context, query, user_message, stats)
    _debug_log_retrieval(retrieved, stats)

    return context, user_message, stats


def _debug_enabled() -> bool:
    """True when CHATBOT_DEBUG_PROMPT is set."""
    return get_settings().chatbot_debug_prompt


def _debug_log_retrieval(retrieved: list[dict], stats: dict) -> None:
    """Log retrieved threads (subject + rank) and per-thread kept counts.

    No-op unless CHATBOT_DEBUG_PROMPT is on.
    """
    if not _debug_enabled():
        return

    for rank, chunk in enumerate(retrieved, 1):
        subject = chunk.get("subject") or chunk.get("thread_subject") or "(no subject)"
        score = chunk.get("combined_score", chunk.get("score", 0))
        logger.info("[CHATBOT-DEBUG] retrieved rank=%d score=%.3f subject=%r", rank, score, subject)

    for rank, subject, kept, total in stats.get("kept_per_thread", []):
        logger.info(
            "[CHATBOT-DEBUG] build_context kept rank=%d blocks=%d/%d subject=%r",
            rank, kept, total, subject,
        )


def _format_selection_report(stats: dict) -> str:
    """Human-readable build_context summary for the debug dump: mode, threads,
    blocks kept/total per thread, characters before cleaning / after cleaning /
    sent, and every block that was chosen."""
    if not stats:
        return "(no context statistics)\n"
    lines = [
        "mode: " + ("everything fit (nothing dropped)" if stats["mode"] == "everything_fit"
                    else "selection used (ranked blocks)"),
        f"threads: {stats['threads_used']} sent of {stats['threads_in']} retrieved",
        f"blocks kept/total: {stats['blocks_kept']}/{stats['blocks_total']}",
        f"characters: before cleaning={stats['chars_raw']} after cleaning={stats['chars_cleaned']} "
        f"sent={stats['chars_sent']}",
        f"repeated blocks removed: {stats['dedup_removed']}  embeddings: {stats['embeddings']}",
    ]
    for cut in stats["cuts"]:
        lines.append(
            f"CUT: thread {cut['subject']!r} block {cut['block']} ({cut['kind']}) kept "
            f"{cut['lines_kept']}/{cut['lines_total']} lines, {cut['chars_cut']} chars cut"
        )
    for thread in stats["chosen"]:
        kept = len(thread["blocks"])
        total = next((t for r, _s, _k, t in stats["kept_per_thread"] if r == thread["rank"]), kept)
        lines.append(f"[rank {thread['rank']}] {thread['subject']!r}: blocks kept {kept}/{total}")
        for b in thread["blocks"]:
            lines.append(
                f"    block {b['pos']} {b['kind']} chars={b['chars']} tier={b['tier']} "
                f"score={b['score']}{' PARTIAL' if b['partial'] else ''}"
            )
    return "\n".join(lines) + "\n"


def _format_calc_report(calc: dict | None) -> str:
    """Debug-dump summary of the code calculation: whether it was used, the
    expression and the value."""
    if not calc or not calc.get("enabled"):
        return "code calculation: off (CHATBOT_CALC_IN_CODE)\n"
    if not calc["attempted"]:
        reason = calc["error"] or "question does not match CHATBOT_CALC_QUESTION_PATTERN"
        return f"code calculation: not used ({reason})\n"
    if calc["used"]:
        return (
            "code calculation: USED\n"
            f"expression: {calc['expression']}\n"
            f"value: {_format_number(calc['value'])}  (raw {calc['value']!r})\n"
            f"extra seconds: {calc['seconds']}\n"
        )
    return (
        f"code calculation: attempted but FAILED ({calc['error']}) after {calc['seconds']}s - "
        "normal path used\n"
    )


def _debug_dump_prompt(user_message: str, raw_answer: str, stats: dict | None = None) -> None:
    """Write the final prompt (system + context + question), the context
    selection report, the code-calculation report and the RAW LLM answer
    (before the hallucination regex / NOT_FOUND replacement) to the debug file,
    overwriting it each request.

    No-op unless CHATBOT_DEBUG_PROMPT is on. Never raises: a debug-dump
    failure must not break the chat response.
    """
    if not _debug_enabled():
        return

    settings = get_settings()
    path = settings.chatbot_debug_prompt_path
    try:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        contents = (
            "===== SYSTEM PROMPT =====\n"
            f"{get_system_prompt()}\n\n"
            "===== CONTEXT SELECTION =====\n"
            f"{_format_selection_report(stats or {})}\n"
            "===== CODE CALCULATION =====\n"
            f"{_format_calc_report((stats or {}).get('calc'))}\n"
            "===== USER MESSAGE (context + question) =====\n"
            f"{user_message}\n\n"
            "===== RAW LLM ANSWER (pre hallucination/NOT_FOUND) =====\n"
            f"{raw_answer}\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(contents)
        logger.info("[CHATBOT-DEBUG] wrote prompt + raw answer to %s", path)
    except Exception as e:
        logger.warning("[CHATBOT-DEBUG] could not write debug prompt file %s: %s", path, e)


def _ns_to_s(value) -> float:
    """Ollama reports *_duration in nanoseconds; convert to seconds."""
    try:
        return round(float(value) / 1e9, 2)
    except (TypeError, ValueError):
        return 0.0


def _log_llm_metrics(
    response_obj,
    *,
    model: str,
    prompt_chars: int,
    retrieval_ms: float,
    stats: dict,
    first_token_s: float | None = None,
) -> None:
    """Log one metrics line per request from the Ollama response object.

    Logs model, thread count (used/retrieved), kept context characters,
    prompt tokens (Ollama's prompt_eval_count), time to first token, eval
    count/duration, tokens/s and retrieval time. ``first_token_s`` is the
    measured time to the first streamed token; when it is None (non-streaming
    call) it is estimated as load_duration + prompt_eval_duration. Emits a
    WARNING if prompt_eval_count exceeds CHATBOT_PROMPT_WARN_TOKENS.
    """
    def _get(key, default=0):
        if isinstance(response_obj, dict):
            return response_obj.get(key, default)
        return getattr(response_obj, key, default)

    prompt_eval_count = _get("prompt_eval_count", 0) or 0
    eval_count = _get("eval_count", 0) or 0
    prompt_eval_s = _ns_to_s(_get("prompt_eval_duration", 0))
    eval_s = _ns_to_s(_get("eval_duration", 0))
    tokens_per_sec = round(eval_count / eval_s, 1) if eval_s > 0 else 0.0
    if first_token_s is None:
        ttft_s, ttft_src = round(_ns_to_s(_get("load_duration", 0)) + prompt_eval_s, 2), "est"
    else:
        ttft_s, ttft_src = round(first_token_s, 2), "measured"

    logger.info(
        "LLM metrics: model=%s threads=%d/%d context_chars=%d prompt_chars=%d "
        "prompt_tokens=%d first_token=%.2fs(%s) prompt_eval_duration=%.2fs "
        "eval_count=%d eval_duration=%.2fs tokens_per_sec=%.1f retrieval=%.1fms",
        model, stats.get("threads_used", 0), stats.get("threads_in", 0),
        stats.get("context_chars", 0), prompt_chars,
        prompt_eval_count, ttft_s, ttft_src, prompt_eval_s,
        eval_count, eval_s, tokens_per_sec, retrieval_ms,
    )
    warn_tokens = get_settings().chatbot_prompt_warn_tokens
    if prompt_eval_count > warn_tokens:
        logger.warning(
            "prompt_eval_count=%d exceeds %d - prompt is still too large; "
            "reduce CHATBOT_CONTEXT_MAX_CHARS or CHATBOT_CONTEXT_TOP_THREADS",
            prompt_eval_count, warn_tokens,
        )


# ── Pre-LLM relevance gate ────────────────────────────────────


def _is_context_relevant(retrieved: list[dict]) -> bool:
    """Return True if the retrieved chunks likely contain useful context.

    Checks whether at least one chunk had a BM25 keyword match. If BM25
    found zero results, the retrieved emails have no keyword overlap with
    the query at all — semantic similarity alone is unreliable (Weaviate
    normalises the top score to 1.0 regardless of actual relevance).

    When BM25 returns nothing, we short-circuit with 'Not found' instead
    of letting the LLM hallucinate from irrelevant context.
    """
    if not retrieved:
        return False

    # Check if ANY chunk had a BM25 keyword match
    has_bm25_hit = any(
        chunk.get("bm25_score", 0) > 0 or chunk.get("bm25_score_norm", 0) > 0
        for chunk in retrieved
    )

    if not has_bm25_hit:
        logger.info(
            "Pre-LLM relevance check FAILED: no BM25 keyword hits across %d chunks — skipping LLM",
            len(retrieved),
        )
        return False

    return True


# ── Post-LLM hallucination detection ─────────────────────────
# Phrases that indicate the LLM is describing its capabilities instead
# of answering from email context. All matched case-insensitively.
_HALLUCINATION_PATTERNS = [
    "i can provide insights",
    "i can help you",
    "i'm here to help",
    "i am here to help",
    "what would you like",
    "how can i assist",
    "i can assist you",
    "executive dashboard",
    "performance metrics and key",
    "today's meetings and preparation",
    "priority levels",
]


def _is_hallucinated_response(text: str) -> bool:
    """Return True if the LLM response looks like a hallucinated capability
    description rather than an answer grounded in email context.

    Checks the response against known hallucination phrases
    (case-insensitive).
    """
    lower = text.lower()
    return any(pattern in lower for pattern in _HALLUCINATION_PATTERNS)


def _persist_message(query: str, answer: str) -> None:
    """Save a Q&A pair to both Redis (recent-window cache) and the
    permanent Postgres `chatbot_history` table.

    `generate_response_stream` is a sync generator (iterated by
    FastAPI's StreamingResponse in a worker thread - no running event
    loop in that thread), so the Postgres write can't simply be
    `await`-ed. It also can't use `asyncio.run()`: that spins up a
    *brand new* event loop, but the shared asyncpg pool's connections
    are bound to the main loop created in `core.db.init_db()` - using
    them from a different loop raises
    "cannot perform operation: another operation is in progress".
    Instead, schedule the coroutine onto that same main loop with
    `run_coroutine_threadsafe` and wait for it here. Never raises: a
    Postgres hiccup must not break the chat response, same contract as
    `save_message`.
    """
    save_message(query, answer)
    try:
        loop = get_main_loop()
        if loop is None:
            logger.warning("Could not persist chat message to Postgres: no main event loop available")
            return
        future = asyncio.run_coroutine_threadsafe(save_chat_message(query, answer), loop)
        future.result(timeout=10)
    except Exception as e:
        logger.warning("Could not persist chat message to Postgres: %s", e)


GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "sup", "yo",
    "hi there", "hello there", "hey there",
    "hii", "hiii", "hiiii", "helloo", "hellooo", "heyyy", "heyy",
    "namaste", "namaskar", "good day",
    "how are you", "how r u", "wassup", "wazzup",
}


def is_greeting(query: str) -> bool:
    """Check if the user query is a simple greeting.

    Handles variations like 'hii', 'hiii', 'helloo', 'heyyy', 'hiiii there', etc.
    """
    cleaned = query.strip().lower().rstrip("!.,?;:")

    if cleaned in GREETINGS:
        return True

    # Normalize repeated characters: "hiii" -> "hi", "heyyy" -> "hey", "helloo" -> "hello"
    normalized = re.sub(r"(.)\1{1,}", r"\1", cleaned)
    if normalized in GREETINGS:
        return True

    # Handle "hii", "hiii" specifically (collapse only trailing repeats)
    normalized2 = re.sub(r"(.)\1+$", r"\1", cleaned)
    if normalized2 in GREETINGS:
        return True

    # Handle greetings with extra words: "hi there!", "hello sir", "hey buddy"
    first_word = cleaned.split()[0] if cleaned.split() else ""
    first_normalized = re.sub(r"(.)\1{1,}", r"\1", first_word)
    greeting_words = {"hi", "hello", "hey", "howdy", "greetings", "yo", "sup"}
    if first_normalized in greeting_words and len(cleaned.split()) <= 3:
        return True

    return False


def _source_key(chunk: dict) -> str:
    """Match source records by durable conversation (thread) identity."""
    conversation_id = str(chunk.get("conversation_id") or "").strip()
    if conversation_id:
        return f"conversation:{conversation_id}"
    meeting_id = str(chunk.get("meeting_id") or "").strip()
    if meeting_id:
        return f"meeting:{meeting_id}"
    return f"legacy:{chunk.get('email_id', '')}:{chunk.get('subject', '')}"


def _extract_matched_message_content(chunk: dict) -> str:
    """Find the matched message's rendered content from the thread's messages_json."""
    matched_id = chunk.get("matched_message_id", "")
    if not matched_id:
        return ""

    raw_json = chunk.get("messages_json", "[]")
    try:
        messages = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except (json.JSONDecodeError, TypeError):
        return ""

    for msg in messages:
        if str(msg.get("message_id", "")) == str(matched_id):
            parts = []
            if msg.get("from_address"):
                parts.append(f"From:    {msg['from_address']}")
            if msg.get("received_at"):
                parts.append(f"Date:    {msg['received_at']}")
            if msg.get("subject"):
                parts.append(f"Subject: {msg['subject']}")
            parts.append("")
            body = (msg.get("body") or "").strip()
            attachment = (msg.get("attachment_data") or "").strip()
            if body:
                parts.append(body)
            if attachment:
                if body:
                    parts.append("")
                parts.append(f"[Attachment]\n{attachment}")
            return "\n".join(parts)

    return ""


def _get_sources(retrieved: list[dict]) -> list[dict]:
    """Extract unique source info from retrieved chunks, including the content sent to the LLM."""
    sources = []
    seen = set()
    for i, chunk in enumerate(retrieved):
        source_type = chunk.get("source_type", "email")
        score = chunk.get("combined_score", chunk.get("score", 0))

        if source_type == "email_conversation":
            participants = chunk.get("participants", [])
            if isinstance(participants, str):
                participants = [participants] if participants else []
            chunk_context = "\n".join([
                f"--- Conversation {i + 1}: {chunk.get('subject', 'No Subject')} (relevance: {score:.3f}) ---",
                f"Conversation ID: {chunk.get('conversation_id', 'N/A')}",
                f"Messages: {chunk.get('message_count', 0)}",
                f"Date range: {chunk.get('first_received_at', 'N/A')} to {chunk.get('last_received_at', 'N/A')}",
                f"Participants: {', '.join(participants) if participants else 'N/A'}",
                "",
                chunk.get("raw_content", ""),
            ])
        elif source_type == "email":
            chunk_context = "\n".join([
                f"--- Email {i + 1}: {chunk.get('subject', 'No Subject')} (relevance: {score:.3f}) ---",
                f"From:     {chunk.get('from_address', 'N/A')}",
                f"Received: {chunk.get('received_at', 'N/A')}",
                "",
                chunk.get("raw_content", ""),
            ])
        elif source_type == "meeting":
            chunk_context = (
                f"--- Meeting {i + 1} (relevance: {score:.3f}) ---\n"
                f"Subject: {chunk.get('subject', 'N/A')}"
            )
        else:
            chunk_context = chunk.get("raw_content", "")

        if source_type in ("email", "email_conversation"):
            key = _source_key(chunk)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "conversation_id": chunk.get("conversation_id", ""),
                    "message_count": chunk.get("message_count", 1),
                    "first_received_at": chunk.get("first_received_at", chunk.get("received_at", "")),
                    "last_received_at": chunk.get("last_received_at", chunk.get("received_at", "")),
                    "participants": chunk.get("participants", []),
                    "subject": chunk.get("subject", "Unknown"),
                    "filename": chunk.get("from_address", "email"),
                    "score": score,
                    "chunk_index": i + 1,
                    "source_type": source_type,
                    "context_sent": chunk_context,
                    "matched_message_id": chunk.get("matched_message_id", ""),
                    "matched_message_subject": chunk.get("matched_message_subject", ""),
                    "matched_message_received_at": chunk.get("matched_message_received_at", ""),
                    "matched_message_content": _extract_matched_message_content(chunk),
                })

        elif source_type == "meeting":
            key = chunk.get("meeting_id", chunk.get("subject", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "subject": chunk.get("subject", "Unknown Meeting"),
                    "filename": "meeting",
                    "score": score,
                    "chunk_index": i + 1,
                    "source_type": source_type,
                    "context_sent": chunk_context,
                })
    return sources


def _retrieve_for_query(query: str, bm25_index, wv_client) -> tuple[list[dict], str]:
    """Run Weaviate retrieval and return (retrieved_chunks, retriever_label)."""
    return retrieve_weaviate(query, wv_client, bm25_index=bm25_index), "weaviate"


async def answer_question(
    message: str,
    bearer_token: str | None = None,
    bm25_index=None,
    wv_client=None,
) -> ChatResponse:
    """Generate a chatbot response (non-streaming). Used by POST /chatbot/query.

    Returns a full ChatResponse with sources, timings, follow-up questions,
    and is_greeting — matching the standalone chatbot's /chat contract.

    `bearer_token` is accepted for backward compatibility with the
    existing request contract but is not used - the Graph token (if ever
    needed here) is resolved server-side from the session, not from the
    client, mirroring the rest of this backend's auth model.
    """
    settings = get_settings()
    query = message.strip()

    timings: dict = {}
    t_total = time.perf_counter()

    if is_greeting(query):
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total"] = round(total_ms, 1)
        logger.info("Greeting detected: '%s' | %.1fms (no LLM call)", sanitize_log_input(query), timings["total"])
        save_message(query, GREETING_RESPONSE)
        await save_chat_message(query, GREETING_RESPONSE)
        return ChatResponse(
            answer=GREETING_RESPONSE,
            sources=[],
            is_greeting=True,
            latency_ms=round(total_ms, 1),
            timings=TimingInfo(total=round(total_ms, 1)),
            follow_up_questions=[],
            tool_used="weaviate",
        )

    # ── Cache-first: same question answered before → return from Redis ──
    cached = get_cached_answer(query, mode="email")
    if cached:
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info("QA cache HIT (email): '%s' | %.1fms", sanitize_log_input(query), total_ms)
        save_message(query, cached["answer"])
        await save_chat_message(query, cached["answer"])
        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceInfo(**s) for s in cached.get("sources", [])],
            is_greeting=False,
            latency_ms=round(total_ms, 1),
            timings=TimingInfo(total=round(total_ms, 1)),
            follow_up_questions=[],
            tool_used=cached.get("tool_used", "weaviate"),
        )

    logger.info("Processing query: '%s'", sanitize_log_input(query))

    t_retrieve = time.perf_counter()
    retrieved, retriever_label = _retrieve_for_query(query, bm25_index, wv_client)
    timings["retrieval"] = round((time.perf_counter() - t_retrieve) * 1000, 1)

    # ── Pre-LLM relevance gate ────────────────────────────────
    # If retrieved context is clearly irrelevant, skip the LLM entirely
    # and return "Not found" — saves ~80s of LLM time and prevents hallucination.
    if not _is_context_relevant(retrieved):
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total"] = round(total_ms, 1)
        logger.info(
            "Query short-circuited (irrelevant context): retriever=%s retrieval=%.0fms total=%.0fms",
            retriever_label, timings["retrieval"], timings["total"],
        )
        save_message(query, NOT_FOUND_RESPONSE)
        await save_chat_message(query, NOT_FOUND_RESPONSE)
        sources = _get_sources(retrieved)
        return ChatResponse(
            answer=NOT_FOUND_RESPONSE,
            sources=[SourceInfo(**s) for s in sources],
            is_greeting=False,
            latency_ms=round(total_ms, 1),
            timings=TimingInfo(
                retrieval=timings.get("retrieval"),
                context_format=timings.get("context_format"),
                total=timings["total"],
            ),
            follow_up_questions=[],
            tool_used=retriever_label,
        )

    # Rebuild the prompt through the size guard (shrinks + rebuilds once if
    # it exceeds the char ceiling) so we never depend on Ollama truncation.
    t_context = time.perf_counter()
    context, user_message, ctx_stats = _build_prompt_with_guard(retrieved, query)
    timings["context_format"] = round((time.perf_counter() - t_context) * 1000, 1)

    model = _select_model(query)
    client = ollama.Client(host=settings.ollama_host)
    t_llm = time.perf_counter()
    response = client.chat(**_chat_request(user_message, model))
    timings["llm_generation"] = round((time.perf_counter() - t_llm) * 1000, 1)

    _log_llm_metrics(
        response,
        model=model,
        prompt_chars=len(user_message),
        retrieval_ms=timings.get("retrieval", 0.0),
        stats=ctx_stats,
    )

    answer_text = response["message"]["content"]

    # Debug dump of the final prompt + RAW answer (before any post-processing).
    _debug_dump_prompt(user_message, answer_text, ctx_stats)

    # ── Post-LLM hallucination check ─────────────────────────
    # If the LLM ignored the system prompt and hallucinated a capability
    # description, replace it with the safe "Not found" response.
    if _is_hallucinated_response(answer_text):
        answer_text = NOT_FOUND_RESPONSE

    total_ms = (time.perf_counter() - t_total) * 1000
    timings["total"] = round(total_ms, 1)

    logger.info(
        "Query answered: retriever=%s retrieval=%.0fms llm=%.0fms total=%.0fms",
        retriever_label, timings["retrieval"], timings["llm_generation"], timings["total"],
    )

    save_message(query, answer_text)
    await save_chat_message(query, answer_text)

    sources = _get_sources(retrieved)

    set_cached_answer(query, {
        "answer": answer_text,
        "sources": sources,
        "follow_up_questions": [],
        "tool_used": retriever_label,
    }, mode="email")

    return ChatResponse(
        answer=answer_text,
        sources=[SourceInfo(**s) for s in sources],
        is_greeting=False,
        latency_ms=round(total_ms, 1),
        timings=TimingInfo(
            retrieval=timings.get("retrieval"),
            context_format=timings.get("context_format"),
            llm_generation=timings.get("llm_generation"),
            total=timings["total"],
        ),
        follow_up_questions=[],
        tool_used=retriever_label,
    )


def generate_response_stream(
    question: str,
    bm25_index=None,
    wv_client=None,
) -> Generator[str, None, None]:
    """Stream chatbot response as Server-Sent Events (SSE).

    Event types: step, sources, token, greeting, done, error — matching
    the existing frontend contract (see chatbot/routes.py docstring).
    """
    settings = get_settings()
    query = question.strip()

    timings: dict = {}
    t_total = time.perf_counter()

    # ── Handle greetings (instant, no LLM) ───────────────────
    if is_greeting(query):
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info("Greeting detected: '%s' | %.1fms (no LLM call)", sanitize_log_input(query), total_ms)
        _persist_message(query, GREETING_RESPONSE)
        yield f"data: {json.dumps({'type': 'greeting', 'content': GREETING_RESPONSE})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timings': {'total': round(total_ms, 1)}, 'follow_up_questions': []})}\n\n"
        return

    # ── Cache-first: same question answered before → stream from Redis ──
    cached = get_cached_answer(query, mode="email")
    if cached:
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info("QA cache HIT (email, stream): '%s' | %.1fms", sanitize_log_input(query), total_ms)
        _persist_message(query, cached["answer"])
        yield f"data: {json.dumps({'type': 'step', 'step': 'cache', 'message': 'Returning cached answer...'})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': cached.get('sources', []), 'all_candidates': [], 'retriever': cached.get('tool_used', 'weaviate'), 'timings': {}})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'content': cached['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timings': {'total': round(total_ms, 1)}, 'follow_up_questions': []})}\n\n"
        return

    logger.info("Processing query (streaming): '%s'", sanitize_log_input(query))

    yield f"data: {json.dumps({'type': 'step', 'step': 'analyzing', 'message': 'Analyzing your query...'})}\n\n"

    # ── Step 1: Retrieve ──────────────────────────────────────
    t_retrieve = time.perf_counter()
    retrieved, retriever_label = _retrieve_for_query(query, bm25_index, wv_client)
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000
    timings["retrieval"] = round(retrieve_ms, 1)

    yield f"data: {json.dumps({'type': 'step', 'step': 'searching', 'message': f'Searching relevant email conversations... ({retrieve_ms:.0f}ms)'})}\n\n"

    # ── Send sources + retrieval timings immediately ─────────
    sources = _get_sources(retrieved)

    all_candidates = []
    if retrieved and "_all_candidates" in retrieved[0]:
        for i, c in enumerate(retrieved[0]["_all_candidates"]):
            sent_to_llm = any(_source_key(c) == _source_key(r) for r in retrieved)
            all_candidates.append({
                "rank": i + 1,
                "conversation_id": c.get("conversation_id", ""),
                "message_id": c.get("message_id", ""),
                "subject": c.get("thread_subject", c.get("subject", "Unknown")),
                "matched_message_subject": c.get("subject", ""),
                "message_count": c.get("message_count", 0),
                "first_received_at": c.get("first_received_at", ""),
                "last_received_at": c.get("last_received_at", ""),
                "matched_message_received_at": c.get("received_at", ""),
                "participants": c.get("participants", []),
                "score": round(c.get("combined_score", c.get("score", 0)), 4),
                "semantic_score": round(c.get("semantic_score_norm", c.get("score", 0)), 4),
                "bm25_score": round(c.get("bm25_score_norm", c.get("bm25_score", 0)), 4),
                "sent_to_llm": sent_to_llm,
                "source_type": c.get("source_type", "email_message"),
            })

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'all_candidates': all_candidates, 'retriever': retriever_label, 'timings': timings})}\n\n"

    # ── Pre-LLM relevance gate ────────────────────────────────
    # If retrieved context is clearly irrelevant, skip the LLM entirely
    # and return "Not found" — saves ~80s of LLM time and prevents hallucination.
    if not _is_context_relevant(retrieved):
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total"] = round(total_ms, 1)
        logger.info(
            "Streaming query short-circuited (irrelevant context): retrieval=%.0fms total=%.0fms",
            timings["retrieval"], timings["total"],
        )
        _persist_message(query, NOT_FOUND_RESPONSE)
        yield f"data: {json.dumps({'type': 'token', 'content': NOT_FOUND_RESPONSE})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timings': timings, 'follow_up_questions': []})}\n\n"
        return

    # ── Step 3: Stream LLM generation ────────────────────────
    model = _select_model(query)
    yield f"data: {json.dumps({'type': 'step', 'step': 'generating', 'message': f'Sending to LLM ({model})...'})}\n\n"

    # Build the prompt through the size guard (shrinks + rebuilds once if it
    # exceeds the char ceiling) so we never depend on Ollama truncation.
    t_context = time.perf_counter()
    context, user_message, ctx_stats = _build_prompt_with_guard(retrieved, query)
    timings["context_format"] = round((time.perf_counter() - t_context) * 1000, 1)

    prompt_tokens = len(user_message) // 4
    logger.info(
        "LLM streaming start: model=%s, ~%d prompt tokens, max_output=%d",
        model, prompt_tokens, settings.chatbot_max_tokens,
    )

    request = _chat_request(user_message, model)
    client = ollama.Client(host=settings.ollama_host)
    t_llm = time.perf_counter()
    t_first_token = None
    token_count = 0
    full_response = ""

    # ── Heartbeat-aware LLM streaming ────────────────────────
    # Ollama can take 60-120s+ to produce the first token (especially
    # on CPU). During that silence, reverse proxies / load balancers
    # kill the idle SSE connection. We run the Ollama stream in a
    # background thread and yield heartbeat events every 10s while
    # waiting, so the connection always has data flowing through it.
    HEARTBEAT_INTERVAL = 10  # seconds between keepalive events
    _SENTINEL = object()     # marks end of the Ollama stream

    token_queue: queue.Queue = queue.Queue()
    llm_error: list = []     # captured exception from the thread
    llm_final: list = []     # captured final chunk (carries eval counts)

    def _ollama_worker():
        """Run blocking Ollama streaming in a background thread,
        pushing each token (or error/sentinel) into token_queue."""
        try:
            stream = client.chat(**request, stream=True)
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    token_queue.put(token)
                # The terminal chunk carries prompt_eval_count/eval_count/
                # *_duration once Ollama finishes; capture it for metrics.
                is_done = chunk.get("done") if isinstance(chunk, dict) else getattr(chunk, "done", False)
                if is_done:
                    llm_final.append(chunk)
        except Exception as e:
            llm_error.append(e)
        finally:
            token_queue.put(_SENTINEL)

    worker = threading.Thread(target=_ollama_worker, daemon=True)
    worker.start()

    try:
        while True:
            try:
                item = token_queue.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                # No token yet — send a heartbeat to keep the SSE
                # connection alive through proxies / load balancers.
                elapsed = time.perf_counter() - t_llm
                yield f"data: {json.dumps({'type': 'heartbeat', 'elapsed_s': round(elapsed, 1)})}\n\n"
                continue

            if item is _SENTINEL:
                break

            # Real token from Ollama
            if t_first_token is None:
                t_first_token = time.perf_counter()
                first_token_ms = (t_first_token - t_llm) * 1000
                logger.info("First token arrived: %.0fms", first_token_ms)
                yield f"data: {json.dumps({'type': 'step', 'step': 'writing', 'message': f'Generating response... (first token in {first_token_ms:.0f}ms)'})}\n\n"

            token_count += 1
            full_response += item
            yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"

    except Exception:
        # Full exception server-side only; the client gets a generic message (CWE-209).
        logger.exception("LLM streaming error")
        yield f"data: {json.dumps({'type': 'error', 'content': STREAM_ERROR_MESSAGE})}\n\n"
        return

    # Surface any error from the background thread.
    if llm_error:
        logger.error("LLM streaming error", exc_info=llm_error[0])
        yield f"data: {json.dumps({'type': 'error', 'content': STREAM_ERROR_MESSAGE})}\n\n"
        return

    # ── Final timing summary ─────────────────────────────────
    llm_ms = (time.perf_counter() - t_llm) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000
    tokens_per_sec = (token_count / (llm_ms / 1000)) if llm_ms > 0 else 0

    timings["llm_generation"] = round(llm_ms, 1)
    timings["first_token"] = round((t_first_token - t_llm) * 1000, 1) if t_first_token else 0
    timings["total"] = round(total_ms, 1)
    timings["tokens"] = token_count
    timings["tokens_per_sec"] = round(tokens_per_sec, 1)

    # Debug dump of the final prompt + RAW streamed answer (before the
    # hallucination regex / NOT_FOUND replacement below).
    _debug_dump_prompt(user_message, full_response, ctx_stats)

    # ── Post-LLM hallucination check ─────────────────────────
    # If the LLM streamed a hallucinated response, send a correction
    # event so the frontend can replace the displayed text.
    if _is_hallucinated_response(full_response):
        full_response = NOT_FOUND_RESPONSE
        yield f"data: {json.dumps({'type': 'replace', 'content': NOT_FOUND_RESPONSE})}\n\n"

    logger.info(
        "Streaming query timing: retrieval=%.1fms context=%.1fms first_token=%.1fms "
        "llm_total=%.1fms (%d tokens, %.0f tok/s) total=%.1fms sources=%s",
        timings["retrieval"], timings["context_format"], timings["first_token"],
        timings["llm_generation"], token_count, tokens_per_sec, timings["total"],
        ", ".join(s["subject"][:30] for s in sources),
    )

    # ── Per-request LLM metrics (from Ollama's terminal chunk) ──
    first_token_s = (t_first_token - t_llm) if t_first_token else None
    if llm_final:
        _log_llm_metrics(
            llm_final[0],
            model=model,
            prompt_chars=len(user_message),
            retrieval_ms=timings.get("retrieval", 0.0),
            stats=ctx_stats,
            first_token_s=first_token_s,
        )
    else:
        # No terminal chunk captured — fall back to our own token count so the
        # metrics line is still emitted (prompt tokens estimated as chars/4).
        logger.info(
            "LLM metrics: model=%s threads=%d/%d context_chars=%d prompt_chars=%d "
            "prompt_tokens=%d (est) first_token=%.2fs eval_count=%d "
            "eval_duration=%.2fs tokens_per_sec=%.1f retrieval=%.1fms",
            model, ctx_stats.get("threads_used", 0), ctx_stats.get("threads_in", 0),
            ctx_stats.get("context_chars", 0), len(user_message),
            len(user_message) // 4, first_token_s or 0.0, token_count,
            llm_ms / 1000, tokens_per_sec, timings.get("retrieval", 0.0),
        )
        if (len(user_message) // 4) > settings.chatbot_prompt_warn_tokens:
            logger.warning(
                "estimated prompt tokens %d exceed %d - reduce CHATBOT_CONTEXT_MAX_CHARS",
                len(user_message) // 4, settings.chatbot_prompt_warn_tokens,
            )

    _persist_message(query, full_response)

    set_cached_answer(query, {
        "answer": full_response,
        "sources": sources,
        "follow_up_questions": [],
        "tool_used": retriever_label,
    }, mode="email")

    yield f"data: {json.dumps({'type': 'done', 'timings': timings, 'follow_up_questions': []})}\n\n"
