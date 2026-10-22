"""
chatbot/postgres_store.py
============================
Fetches raw email message rows for the chatbot's RAG pipeline from the
shared PostgreSQL database (same connection pool as the rest of the
backend - see core/db.py).

Mirrors the standalone chatbot_final project's postgres_client.py, but
uses this project's shared asyncpg pool (core.db.get_pool()) instead of
opening a dedicated psycopg2 connection, so the chatbot does not add a
second database driver or a second set of connection credentials.

The table is expected to carry one row per raw email message with at
least these columns (see chatbot_final's bu_emails table layout):
    conversation_id, message_id, subject, raw_content,
    attachment_data, received_at, from_address
"""

from core.config import get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")


async def fetch_email_rows() -> list[dict]:
    """Fetch all messages required to build conversation-level chunks.

    Every source row must have both a stable ``message_id`` and the
    ``conversation_id`` used to group replies and forwards into one
    thread. Individual empty messages are retained because their
    metadata or attachment may still belong to a conversation with
    usable content elsewhere.

    Raises RuntimeError if the shared PostgreSQL pool is not configured
    (POSTGRES_DSN empty) — the chatbot has no CSV/offline fallback for
    its data source, unlike the tracker stores.
    """
    pool = get_pool()
    if pool is None:
        raise RuntimeError(
            "Chatbot requires PostgreSQL (POSTGRES_DSN) to be configured; "
            "the connection pool is not initialised."
        )

    settings = get_settings()
    table = settings.chatbot_pg_table

    sql = f"""
        SELECT
            conversation_id,
            message_id,
            subject,
            raw_content,
            attachment_data,
            received_at,
            from_address
        FROM {table}
        WHERE NULLIF(BTRIM(conversation_id), '') IS NOT NULL
          AND NULLIF(BTRIM(message_id), '') IS NOT NULL
          AND is_active = 1
        ORDER BY conversation_id ASC,
                 received_at ASC NULLS LAST,
                 message_id ASC;
    """

    logger.info("Fetching conversation messages from Postgres table '%s'...", table)
    async with pool.acquire() as conn:
        records = await conn.fetch(sql)

    rows: list[dict] = []
    for r in records:
        rows.append({
            "conversation_id": (r["conversation_id"] or "").strip(),
            "message_id":      (r["message_id"] or "").strip(),
            "subject":         (r["subject"] or "").strip(),
            "raw_content":     (r["raw_content"] or "").strip(),
            "attachment_data": (r["attachment_data"] or "").strip(),
            "received_at":     str(r["received_at"] or "").strip(),
            "from_address":    (r["from_address"] or "").strip(),
        })

    logger.info("Fetched %d conversation messages from Postgres.", len(rows))
    return rows


async def save_chat_message(question: str, answer: str) -> None:
    """Persist one Q&A pair to the durable `chatbot_history` table.

    This is the permanent record (unlike Redis, which only keeps a
    bounded recent window). Never raises - logs a warning and returns
    silently if PostgreSQL is unavailable, so a DB hiccup never breaks
    the chat response.
    """
    pool = get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chatbot_history (question, answer) VALUES ($1, $2);",
                question, answer,
            )
    except Exception as e:
        logger.warning("Could not save chat message to Postgres: %s", e)


async def fetch_chat_history(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` Q&A pairs, oldest first.

    Returns an empty list if PostgreSQL is unavailable or history is
    empty, mirroring chatbot/redis_history.get_history's contract.
    """
    pool = get_pool()
    if pool is None:
        return []
    try:
        async with pool.acquire() as conn:
            records = await conn.fetch(
                """
                SELECT question, answer, created_at
                FROM chatbot_history
                ORDER BY created_at DESC, id DESC
                LIMIT $1;
                """,
                limit,
            )
    except Exception as e:
        logger.warning("Could not fetch chat history from Postgres: %s", e)
        return []

    # Query orders newest-first (so LIMIT keeps the most recent rows);
    # reverse here to return oldest-first, matching the API contract.
    return [
        {
            "question": r["question"],
            "answer": r["answer"],
            "timestamp": r["created_at"].timestamp(),
        }
        for r in reversed(records)
    ]


async def health_check() -> bool:
    """Return True if the configured chatbot table exists in Postgres."""
    pool = get_pool()
    if pool is None:
        return False

    settings = get_settings()
    table = settings.chatbot_pg_table
    try:
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1;",
                table,
            )
        if not exists:
            logger.error("Postgres table '%s' not found.", table)
        return bool(exists)
    except Exception as e:
        logger.error("Postgres health check failed: %s", e)
        return False
