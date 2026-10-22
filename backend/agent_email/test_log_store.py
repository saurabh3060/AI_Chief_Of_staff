"""Append-only JSONL storage for explicit email summarizer test runs."""

import asyncio
import json
from pathlib import Path

from agent_email.schema import EmailSummarizerTestResult
from core.config import Settings, get_settings


_write_lock = asyncio.Lock()


def get_test_log_path(settings: Settings | None = None) -> Path:
    active_settings = settings or get_settings()
    filename = Path(active_settings.email_summarizer_test_log_filename).name
    return active_settings.resolved_bu_email_data_dir / filename


async def append_test_result(
    result: EmailSummarizerTestResult,
    settings: Settings | None = None,
) -> Path:
    """Append one complete test result as one UTF-8 JSON line."""
    active_settings = settings or get_settings()
    if not active_settings.email_summarizer_test_log_enabled:
        raise RuntimeError("Email summarizer test logging is disabled.")

    path = get_test_log_path(active_settings)
    line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    async with _write_lock:
        with path.open("a", encoding="utf-8", newline="") as file:
            file.write(line + "\n")
    return path
