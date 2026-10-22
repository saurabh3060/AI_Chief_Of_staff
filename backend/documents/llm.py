"""Client for the local Ollama server.

Chat calls use Ollama's native ``/api/chat`` endpoint because granite 4.2's
thinking mode is ON by default and can only be disabled via the native
``think`` parameter (not available on the OpenAI-compat ``/v1`` endpoint).
Embeddings use ``/v1/embeddings`` which works fine for all models.
"""
import base64
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from . import config
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")

# Loading a model from disk or a long generation can take minutes; only connecting is quick.
_TIMEOUT = httpx.Timeout(None, connect=5)

# Parameters the llama.cpp server understood that Ollama does not.
_UNSUPPORTED_PARAMS = ("cache_prompt", "chat_template_kwargs", "response_format")

# Context window — Ollama defaults to 2048 which is too small for the 10K
# system prompt the documents Q&A bot sends.
_NUM_CTX = int(os.getenv("DOC_LLM_NUM_CTX", "16384"))


def _clean(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in _UNSUPPORTED_PARAMS}


# ---------------------------------------------------------------------------
# Ollama native /api/chat  (used for all chat calls)
# ---------------------------------------------------------------------------

def _convert_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages (with image_url parts) to Ollama native format."""
    out = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts, images = [], []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        images.append(url.split(",", 1)[1])
            converted = {"role": msg["role"], "content": "\n".join(text_parts)}
            if images:
                converted["images"] = images
            out.append(converted)
        else:
            out.append({"role": msg["role"], "content": content or ""})
    return out


def _ollama_chat(model: str, messages: list[dict], stream: bool = False, **params) -> dict:
    """Call Ollama's native /api/chat endpoint.

    Key difference from /v1: supports ``think: false`` to disable granite's
    reasoning mode, which otherwise eats the entire token budget on large prompts.
    """
    body = {
        "model": model,
        "messages": _convert_messages(messages),
        "stream": stream,
        "think": False,         # granite 4.2 has thinking ON by default — must disable
        "options": {
            "num_ctx": _NUM_CTX,
        },
    }
    if "temperature" in params:
        body["options"]["temperature"] = params["temperature"]
    if "max_tokens" in params:
        body["options"]["num_predict"] = params["max_tokens"]
    if "tools" in params and params["tools"]:
        body["tools"] = params["tools"]
    if "tool_choice" in params and params["tool_choice"]:
        body["tool_choice"] = params["tool_choice"]

    logger.info("[DOC-LLM] POST /api/chat  model=%s  host=%s  tools=%s  think=false  num_ctx=%d  num_predict=%s",
                model, config.OLLAMA_HOST, bool(params.get("tools")),
                _NUM_CTX, params.get("max_tokens", "default"))
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=_TIMEOUT)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[DOC-LLM] POST /api/chat  status=%s  %.0fms", r.status_code, elapsed)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-LLM] POST /api/chat  CONNECT FAILED after %.0fms: %s", elapsed, e)
        raise
    except httpx.HTTPStatusError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-LLM] POST /api/chat  HTTP %s after %.0fms: %s",
                     e.response.status_code, elapsed, e)
        raise
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-LLM] POST /api/chat  FAILED after %.0fms: %s", elapsed, e)
        raise


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/embeddings  (kept for embeddings only)
# ---------------------------------------------------------------------------

def _post_v1(path: str, body: dict) -> dict:
    model = body.get("model", "?")
    logger.info("[DOC-LLM] POST /v1%s  model=%s  host=%s", sanitize_log_input(path), sanitize_log_input(str(model)), config.OLLAMA_HOST)
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{config.OLLAMA_HOST}/v1{path}", json=body, timeout=_TIMEOUT)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[DOC-LLM] POST /v1%s  status=%s  %.0fms", path, r.status_code, elapsed)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-LLM] POST /v1%s  CONNECT FAILED after %.0fms: %s", path, elapsed, e)
        raise
    except httpx.HTTPStatusError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-LLM] POST /v1%s  HTTP %s after %.0fms: %s",
                     path, e.response.status_code, elapsed, e)
        raise
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-LLM] POST /v1%s  FAILED after %.0fms: %s", path, elapsed, e)
        raise


# ---------------------------------------------------------------------------
# Public API (same signatures as before — callers don't change)
# ---------------------------------------------------------------------------

def models() -> dict[str, str]:
    """Model name -> "loaded" for every model Ollama has pulled."""
    logger.info("[DOC-LLM] Checking available models at %s/api/tags", config.OLLAMA_HOST)
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        result = {m["name"]: "loaded" for m in r.json()["models"]}
        logger.info("[DOC-LLM] Available models: %s (%.0fms)",
                    list(result.keys()), (time.perf_counter() - t0) * 1000)
        return result
    except Exception as e:
        logger.error("[DOC-LLM] Failed to list models: %s (%.0fms)",
                     e, (time.perf_counter() - t0) * 1000)
        raise


def unload(model: str) -> None:
    """Ask Ollama to drop the model from memory now."""
    httpx.post(f"{config.OLLAMA_HOST}/api/generate",
               json={"model": model, "keep_alive": 0}, timeout=30)


def embed(texts: list[str]) -> list[list[float]]:
    logger.info("[DOC-LLM] Embedding %d text(s), model=%s, total_chars=%d",
                len(texts), config.EMBED_MODEL, sum(len(t) for t in texts))
    t0 = time.perf_counter()
    data = _post_v1("/embeddings", {"model": config.EMBED_MODEL, "input": texts})["data"]
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-LLM] Embedding done: %d vectors, dim=%d, %.0fms",
                len(data), len(data[0]["embedding"]) if data else 0, elapsed)
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def image_part(path: str) -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def chat(model: str, messages: list[dict], **params) -> str:
    logger.info("[DOC-LLM] chat() model=%s, messages=%d, params=%s",
                model, len(messages), {k: v for k, v in params.items() if k != "tools"})
    return chat_full(model, messages, **params)[0]


def chat_full(model: str, messages: list[dict], **params) -> tuple[str, bool]:
    """Returns (text, truncated): truncated is True when the reply hit max_tokens."""
    logger.info("[DOC-LLM] chat_full() model=%s, messages=%d", model, len(messages))
    t0 = time.perf_counter()
    resp = _ollama_chat(model, messages, **_clean(params))
    content = resp.get("message", {}).get("content", "") or ""
    truncated = resp.get("done_reason") == "length"
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-LLM] chat_full() done: %d chars, truncated=%s, done_reason=%s, %.0fms",
                len(content), truncated, resp.get("done_reason"), elapsed)
    return content, truncated


def chat_message(model: str, messages: list[dict], **params) -> dict:
    """The whole reply message, including tool_calls when ``tools`` are passed.

    Returns a dict shaped like OpenAI's message object so embedqa.py doesn't change:
    ``{"content": str, "tool_calls": list | None}``
    """
    has_tools = "tools" in params
    logger.info("[DOC-LLM] chat_message() model=%s, messages=%d, has_tools=%s",
                model, len(messages), has_tools)
    t0 = time.perf_counter()
    resp = _ollama_chat(model, messages, **_clean(params))
    msg = resp.get("message", {})
    content = msg.get("content", "") or ""
    # Ollama native: arguments is a dict. OpenAI: arguments is a JSON string.
    # Normalize to JSON string so embedqa.py's json.loads() works.
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        if isinstance(fn.get("arguments"), dict):
            fn["arguments"] = json.dumps(fn["arguments"])
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-LLM] chat_message() done: content=%d chars, tool_calls=%d, %.0fms",
                len(content), len(tool_calls), elapsed)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            logger.info("[DOC-LLM]   tool_call[%d]: %s(%s)",
                        i, fn.get("name"), str(fn.get("arguments", ""))[:200])
    return {"content": content, "tool_calls": tool_calls or None}


def rerank(query: str, documents: list[str]) -> list[float]:
    """Not available on Ollama, which has no cross-encoder endpoint."""
    logger.error("[DOC-LLM] rerank() called but Ollama has no rerank endpoint")
    raise NotImplementedError("Ollama has no rerank endpoint; leave config.RERANK_MODEL empty.")


def chat_stream(model: str, messages: list[dict], **params) -> Iterator[tuple[str, str]]:
    """Yields (kind, text) with kind "thinking" or "answer"."""
    logger.info("[DOC-LLM] chat_stream() model=%s, messages=%d", model, len(messages))
    t0 = time.perf_counter()
    token_count = 0
    body = {
        "model": model,
        "messages": _convert_messages(messages),
        "stream": True,
        "think": False,
        "options": {"num_ctx": _NUM_CTX},
    }
    if "temperature" in params:
        body["options"]["temperature"] = params["temperature"]
    if "max_tokens" in params:
        body["options"]["num_predict"] = params["max_tokens"]
    if "tools" in params and params["tools"]:
        body["tools"] = params["tools"]

    with httpx.stream("POST", f"{config.OLLAMA_HOST}/api/chat",
                      json=body, timeout=_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = data.get("message", {})
            text = msg.get("content", "")
            if text:
                token_count += 1
                yield "answer", text
            if data.get("done"):
                break
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-LLM] chat_stream() done: %d tokens, %.0fms", token_count, elapsed)
