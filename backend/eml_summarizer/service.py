"""
eml_summarizer/service.py
============================
High-level service that combines EML parsing with Ollama summarisation.

Called by the chatbot service when the ``eml_summary`` tool is selected.
"""

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.ollama_client.client import OllamaClient

from eml_summarizer.parser import parse_eml_file
from eml_summarizer.prompts import (
    EML_SUMMARY_SYSTEM_PROMPT,
    build_eml_summary_prompt,
)

logger = get_agent_logger("eml_summarizer")


async def summarise_eml(
    filename: str,
    ollama_client: OllamaClient | None = None,
    settings: Settings | None = None,
) -> dict:
    """
    Parse a `.eml` file and generate a summary via Ollama.

    Returns a dict with:
        - ``filename``: the original filename
        - ``subject``: email subject line
        - ``from``: sender
        - ``summary``: the LLM-generated summary text

    Raises ``FileNotFoundError`` if the file doesn't exist.
    """
    active_settings = settings or get_settings()
    client = ollama_client or OllamaClient(settings=active_settings)

    # 1. Parse the .eml file.
    parsed = parse_eml_file(filename)
    # Strip CR/LF from user-controlled values before logging to prevent
    # log injection / forged log lines (CWE-117).
    sanitized_item_id = filename.replace("\r\n", "").replace("\r", "").replace("\n", "")
    sanitized_item_subject = str(parsed["subject"]).replace("\r\n", "").replace("\r", "").replace("\n", "")
    logger.info(
        "Parsed EML '%s': subject='%s', body_chars=%d",
        sanitized_item_id, sanitized_item_subject, len(parsed["body"]),
    )

    # 2. Build the summarisation prompt and send to Ollama.
    prompt = build_eml_summary_prompt(parsed)
    summary_text = await client.chat(prompt, system_prompt=EML_SUMMARY_SYSTEM_PROMPT)

    logger.info("Generated summary for '%s': %d chars", sanitized_item_id, len(summary_text))

    return {
        "filename": parsed["filename"],
        "subject": parsed["subject"],
        "from": parsed["from"],
        "summary": summary_text,
    }
