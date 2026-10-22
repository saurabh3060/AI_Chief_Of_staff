"""
eml_summarizer/routes.py
===========================
FastAPI routes for the EML summariser — utility endpoints for listing
available `.eml` files and summarising a specific one by name.

These are convenience/debug endpoints; the primary integration is
through the chatbot (``POST /chatbot/query``).
"""

from fastapi import APIRouter, HTTPException

from core.config import get_settings
from core.logging_config import get_agent_logger
from core.ollama_client.client import OllamaClient
from eml_summarizer.parser import list_eml_files
from eml_summarizer.service import summarise_eml

logger = get_agent_logger("eml_summarizer")

router = APIRouter(prefix="/eml", tags=["eml_summarizer"])


@router.get("/files")
async def get_eml_files() -> dict:
    """List all `.eml` files available in the storage folder."""
    files = list_eml_files()
    return {"files": files, "count": len(files)}


@router.get("/summarize/{filename}")
async def summarize_eml_file(filename: str) -> dict:
    """
    Parse and summarise a specific `.eml` file by name.

    Returns the email metadata plus the LLM-generated summary.
    """
    try:
        settings = get_settings()
        client = OllamaClient(settings=settings)
        result = await summarise_eml(filename, ollama_client=client, settings=settings)
        return result
    except FileNotFoundError:
        # The exception text carries a server file path; keep it in the log only.
        logger.exception("EML file not found")
        raise HTTPException(status_code=404, detail="EML file not found.")
    except Exception:
        logger.exception("EML summarisation failed")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
