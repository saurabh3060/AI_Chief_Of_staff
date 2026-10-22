"""
agent_email/db_schema.py
========================
Single source of truth for the email agent's PostgreSQL schema.
 
The FastAPI startup path (``core.db.init_db``) imports the DDL from here,
so the table/index definitions live in exactly one place.
 
Storage model
-------------
PostgreSQL is the system of record: these tables store the **row** data
AND the embedding itself (``embedding`` JSONB + ``embedding_model`` +
``embedding_dim`` + ``content_hash``). Each row that has an embedding also
carries a ``weaviate_id`` (TEXT, a deterministic UUID string — see
``agent_email/storage_ids.py``) that is the object id of the mirrored
vector in the corresponding Weaviate collection (a rebuildable copy, see
``agent_email/weaviate_vector_store.py``). Rows without an embedding leave
``embedding``/``weaviate_id`` NULL.
 
Table structure
---------------
``email_clusters``            one synthesized topic cluster per row
    cluster_id      TEXT PRIMARY KEY   -- e.g. "cluster_<key>"
    department      TEXT NOT NULL
    topic           TEXT
    summary         TEXT               -- "|"-joined bullet points
    top_questions   TEXT               -- "|"-joined
    action_items    TEXT               -- "|"-joined "<owner>: <task> (by <deadline>)"
    email_refs      JSONB              -- list of ClusterEmailRef dicts
    email_count     INTEGER NOT NULL DEFAULT 0
    weaviate_id     TEXT               -- id into the "clusters" Weaviate collection (NULL = no embedding)
    meeting_id      TEXT               -- past meeting occurrence (empty for semantic-only)
    meeting_start   TEXT
    is_active       SMALLINT NOT NULL DEFAULT 1  -- 1 = current/live cluster for its meeting, 0 = superseded/historical
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
 
Active-vs-historical model
--------------------------
PostgreSQL keeps the FULL history: every cluster ever synthesized is a
row and nothing is deleted. ``is_active`` marks which row is the live one
for a given meeting. The invariant is: **for any non-empty ``meeting_id``
at most one row has ``is_active = 1``** (enforced by a partial unique
index). When a new cluster is formed for a meeting that already has an
active cluster, the pipeline flips the old row(s) to ``is_active = 0`` and
removes their vectors from Weaviate — so Weaviate and the Redis cache hold
exactly the active (one-per-meeting) set, while PostgreSQL retains the
superseded rows for history.
 
``email_topics``              per-mail LLM topic tag (no embedding)
    message_id      TEXT
    subject         TEXT
    from_address    TEXT
    received_at     TEXT
    topic_key       TEXT
    run_id          TEXT
    id              BIGSERIAL PRIMARY KEY  -- surrogate; message_id is not unique across runs
 
``meeting_embeddings``        cached per-meeting embedding pointer
    meeting_id      TEXT PRIMARY KEY
    subject         TEXT
    weaviate_id     TEXT               -- id into the "meetings" Weaviate collection
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
 
``meeting_history``           past calendar occurrences used as anchors
    meeting_id      TEXT PRIMARY KEY
    subject         TEXT
    start_datetime  TEXT
    weaviate_id     TEXT               -- id into the "meetings" Weaviate collection
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
 
``bu_emails``                 one row per raw fetched email message
    (See ``create_bu_emails_schema`` below. Unlike the four tables above
    this one is NOT part of the cluster synthesis storage model and is
    created regardless of ``EMAIL_STORAGE_BACKEND`` — it is the chatbot's
    RAG source table, written by the email pipeline and read by
    ``chatbot/postgres_store.fetch_email_rows``.)
 
Indexes
-------
* ``email_clusters(department)``  — get_email_cluster_records filters by dept
* ``email_clusters(meeting_id)``  — exact meeting_id match in match_meeting_to_cluster
* ``email_topics(run_id)``        — topics are grouped/replaced per run
* ``email_topics(message_id, run_id)`` unique — upsert key for write_topic_records
* unique weaviate_id per namespace is guaranteed by the app (ids are
  derived deterministically), but a partial unique index is added
  defensively.
"""
 
import re
 
from core.logging_config import get_agent_logger
 
_logger = get_agent_logger("agent_email")
 
# Ordered list of (name, DDL) pairs. Executed sequentially; every statement
# is IF NOT EXISTS so running this repeatedly is safe (idempotent).
EMAIL_TABLE_DDL: list[tuple[str, str]] = [
    (
        "email_clusters",
        """
        CREATE TABLE IF NOT EXISTS email_clusters (
            cluster_id    TEXT PRIMARY KEY,
            department    TEXT NOT NULL,
            topic         TEXT DEFAULT '',
            summary       TEXT DEFAULT '',
            top_questions TEXT DEFAULT '',
            action_items  TEXT DEFAULT '',
            email_refs    JSONB DEFAULT '[]'::jsonb,
            email_count   INTEGER NOT NULL DEFAULT 0,
            weaviate_id   TEXT,
            embedding       JSONB,
            embedding_model TEXT DEFAULT '',
            embedding_dim   INTEGER DEFAULT 0,
            content_hash    TEXT DEFAULT '',
            meeting_id    TEXT DEFAULT '',
            anchored_meeting_id TEXT DEFAULT '',
            meeting_start TEXT DEFAULT '',
            is_active     SMALLINT NOT NULL DEFAULT 1,
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
    (
        "email_topics",
        """
        CREATE TABLE IF NOT EXISTS email_topics (
            id                 BIGSERIAL PRIMARY KEY,
            message_id         TEXT DEFAULT '',
            subject            TEXT DEFAULT '',
            from_address       TEXT DEFAULT '',
            received_at        TEXT DEFAULT '',
            topic_key          TEXT DEFAULT '',
            run_id             TEXT DEFAULT '',
            business_objective TEXT  DEFAULT '',
            topics             JSONB DEFAULT '[]'::jsonb,
            entities           JSONB DEFAULT '[]'::jsonb,
            actions            JSONB DEFAULT '[]'::jsonb,
            subtopics          JSONB DEFAULT '[]'::jsonb,
            decisions          JSONB DEFAULT '[]'::jsonb,
            context            TEXT  DEFAULT ''
        );
        """,
    ),
    (
        "meeting_embeddings",
        """
        CREATE TABLE IF NOT EXISTS meeting_embeddings (
            meeting_id  TEXT PRIMARY KEY,
            subject     TEXT DEFAULT '',
            department     TEXT DEFAULT '',
            weaviate_id TEXT,
            embedding       JSONB,
            embedding_model TEXT DEFAULT '',
            embedding_dim   INTEGER DEFAULT 0,
            content_hash    TEXT DEFAULT '',
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
    (
        "meeting_history",
        """
        CREATE TABLE IF NOT EXISTS meeting_history (
            meeting_id      TEXT PRIMARY KEY,
            subject         TEXT DEFAULT '',
            start_datetime  TEXT DEFAULT '',
            business_context TEXT DEFAULT '',
            department     TEXT DEFAULT '',
            weaviate_id     TEXT,
            embedding       JSONB,
            embedding_model TEXT DEFAULT '',
            embedding_dim   INTEGER DEFAULT 0,
            content_hash    TEXT DEFAULT '',
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
    (
        "meeting_notes",
        """
        CREATE TABLE IF NOT EXISTS meeting_notes (
            id BIGSERIAL PRIMARY KEY,
            meeting_id TEXT NOT NULL,
            note_text TEXT NOT NULL,
            action_items JSONB DEFAULT '[]'::jsonb,
            tags JSONB DEFAULT '[]'::jsonb,
            source TEXT DEFAULT 'ceo_note',
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
        """,
    ),
    (
        "email_summaries",
        """
        CREATE TABLE IF NOT EXISTS email_summaries (
            thread_id    TEXT PRIMARY KEY,
            summary_json JSONB NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
]
 
# Secondary indexes (all IF NOT EXISTS).
EMAIL_INDEX_DDL: list[tuple[str, str]] = [
    (
        "idx_email_clusters_department",
        "CREATE INDEX IF NOT EXISTS idx_email_clusters_department "
        "ON email_clusters (department);",
    ),
    (
        "idx_email_clusters_meeting_id",
        "CREATE INDEX IF NOT EXISTS idx_email_clusters_meeting_id "
        "ON email_clusters (meeting_id);",
    ),
    (
        # Supports the meeting-match / pre-meeting-brief lookup that
        # resolves a live meeting's raw Graph id back to its cluster(s).
        "idx_email_clusters_anchored_meeting_id",
        "CREATE INDEX IF NOT EXISTS idx_email_clusters_anchored_meeting_id "
        "ON email_clusters (anchored_meeting_id);",
    ),
    (
        "idx_email_clusters_is_active",
        "CREATE INDEX IF NOT EXISTS idx_email_clusters_is_active "
        "ON email_clusters (is_active);",
    ),
    (
        # Enforce the core invariant: at most one ACTIVE cluster per
        # (non-empty) meeting. Semantic-only clusters (meeting_id = '')
        # are excluded so many of them can coexist as active.
        "uq_email_clusters_active_meeting",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_email_clusters_active_meeting "
        "ON email_clusters (meeting_id) "
        "WHERE is_active = 1 AND meeting_id <> '';",
    ),
    (
        "idx_email_clusters_weaviate_id",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_email_clusters_weaviate_id "
        "ON email_clusters (weaviate_id) WHERE weaviate_id IS NOT NULL;",
    ),
    (
        "idx_email_topics_run_id",
        "CREATE INDEX IF NOT EXISTS idx_email_topics_run_id "
        "ON email_topics (run_id);",
    ),
    (
        "idx_email_topics_message_id",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_email_topics_message_id "
        "ON email_topics (message_id);",
    ),
    (
        "idx_meeting_embeddings_weaviate_id",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_meeting_embeddings_weaviate_id "
        "ON meeting_embeddings (weaviate_id) WHERE weaviate_id IS NOT NULL;",
    ),
    (
        "idx_meeting_history_weaviate_id",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_meeting_history_weaviate_id "
        "ON meeting_history (weaviate_id) WHERE weaviate_id IS NOT NULL;",
    ),
    (
        # Supports department-scoped lookups when reducing a meeting's
        # candidate cluster pool to its own department.
        "idx_meeting_history_department",
        "CREATE INDEX IF NOT EXISTS idx_meeting_history_department "
        "ON meeting_history (department);",
    ),
    (
        "idx_meeting_notes_meeting_id",
        "CREATE INDEX IF NOT EXISTS idx_meeting_notes_meeting_id "
        "ON meeting_notes (meeting_id);",
    ),
    (
        "idx_meeting_notes_source",
        "CREATE INDEX IF NOT EXISTS idx_meeting_notes_source "
        "ON meeting_notes (source);",
    ),
    (
        "idx_email_summaries_thread_id",
        "CREATE INDEX IF NOT EXISTS idx_email_summaries_thread_id "
        "ON email_summaries (thread_id);",
    ),
    (
        "idx_email_clusters_content_hash",
        "CREATE INDEX IF NOT EXISTS idx_email_clusters_content_hash "
        "ON email_clusters (content_hash);",
    ),
    (
        "idx_meeting_embeddings_content_hash",
        "CREATE INDEX IF NOT EXISTS idx_meeting_embeddings_content_hash "
        "ON meeting_embeddings (content_hash);",
    ),
    (
        "idx_meeting_history_content_hash",
        "CREATE INDEX IF NOT EXISTS idx_meeting_history_content_hash "
        "ON meeting_history (content_hash);",
    ),
]
 
 
# Tables (and their old FAISS-era unique index names) that need the
# legacy `faiss_id BIGINT` column migrated to `weaviate_id TEXT` on
# deployments created before the Weaviate migration. New deployments never
# have the old column, so every statement here is a defensive no-op guard.
_FAISS_TO_WEAVIATE_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("email_clusters", "idx_email_clusters_faiss_id"),
    ("meeting_embeddings", "idx_meeting_embeddings_faiss_id"),
    ("meeting_history", "idx_meeting_history_faiss_id"),
]
 
 
async def _migrate_legacy_faiss_id_columns(conn) -> None:
    """
    One-time, idempotent migration for deployments that still have the old
    FAISS-era `faiss_id BIGINT` column: drop the old unique index, rename
    the column to `weaviate_id`, and widen its type to TEXT (BIGINT values
    cannot be cast to the new UUID-string ids, so any pre-existing FAISS
    ids are discarded — the pipeline re-embeds and rewrites vectors on its
    next run, which already self-heals missing embeddings).
    """
    for table, old_index in _FAISS_TO_WEAVIATE_COLUMN_MIGRATIONS:
        has_old_column = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = $1 AND column_name = 'faiss_id'",
            table,
        )
        if not has_old_column:
            continue
 
        has_new_column = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = $1 AND column_name = 'weaviate_id'",
            table,
        )
        await conn.execute(f"DROP INDEX IF EXISTS {old_index}")
        if has_new_column:
            # Both columns exist (e.g. `weaviate_id` was already added by
            # normal schema DDL on a deployment that never cleaned up the
            # legacy `faiss_id` column). There is nothing to rename —
            # just drop the stray legacy column instead of renaming into
            # a name that's already taken.
            await conn.execute(f"ALTER TABLE {table} DROP COLUMN faiss_id")
            continue
        await conn.execute(
            f"ALTER TABLE {table} RENAME COLUMN faiss_id TO weaviate_id"
        )
        await conn.execute(
            f"ALTER TABLE {table} ALTER COLUMN weaviate_id TYPE TEXT "
            f"USING NULL"
        )
 
 
async def create_email_schema(conn) -> None:
    """
    Create all email agent tables + indexes on an asyncpg connection.
 
    Idempotent (every statement is IF NOT EXISTS). Wrapped in a single
    transaction by the caller when atomicity across all DDL is required.
 
    Also runs column-level migrations so tables that existed before a new
    column was introduced get the missing column added automatically.
    """
    for _name, ddl in EMAIL_TABLE_DDL:
        await conn.execute(ddl)
    # Backfill columns introduced after the original table was deployed.
    await conn.execute(
        "ALTER TABLE meeting_history "
        "ADD COLUMN IF NOT EXISTS business_context TEXT DEFAULT ''"
    )
    await conn.execute(
        "ALTER TABLE meeting_history "
        "ADD COLUMN IF NOT EXISTS end_datetime TEXT DEFAULT ''"
    )
    # Embedding-identified department for the occurrence. Added additively so
    # pre-existing deployments gain the column; existing rows re-label on the
    # next history pull (department is part of the content hash, so the pull
    # no longer skips them as unchanged).
    await conn.execute(
        "ALTER TABLE meeting_history "
        "ADD COLUMN IF NOT EXISTS department TEXT DEFAULT ''"
    )
    # Same embedding-identified department on the per-meeting query-vector
    # table. This is the one the pre-meeting-brief match resolves its vector
    # from, so the department tagged here at today's-meetings capture time is
    # what drives the candidate-pool reduction. Added additively; the row's
    # content hash now folds in department so an unlabeled -> labeled change
    # is actually persisted instead of skipped as unchanged.
    await conn.execute(
        "ALTER TABLE meeting_embeddings "
        "ADD COLUMN IF NOT EXISTS department TEXT DEFAULT ''"
    )
    # Full structured semantics extracted ONCE at the relevance stage and
    # reused by clustering (replaces the former second topic-tag LLM call).
    # Added additively so pre-existing deployments gain the columns; the
    # per-run upsert (write_topic_records) repopulates them on the next run.
    # JSONB list columns default to an empty array, text columns to ''.
    for _column, _definition in (
        ("business_objective", "TEXT  DEFAULT ''"),
        ("topics", "JSONB DEFAULT '[]'::jsonb"),
        ("entities", "JSONB DEFAULT '[]'::jsonb"),
        ("actions", "JSONB DEFAULT '[]'::jsonb"),
        ("subtopics", "JSONB DEFAULT '[]'::jsonb"),
        ("decisions", "JSONB DEFAULT '[]'::jsonb"),
        ("context", "TEXT  DEFAULT ''"),
    ):
        await conn.execute(
            f"ALTER TABLE email_topics ADD COLUMN IF NOT EXISTS {_column} {_definition}"
        )
    # Structured department/team tags a CEO attached to a meeting note
    # (the "@Risk" / "@BB&RL" mentions typed in the notes box). Stored as a
    # JSONB string array so cluster synthesis can read which teams the note
    # was directed at, rather than losing that routing inside free text.
    # Added additively so deployments created before this gain the column
    # (existing rows default to an empty array).
    await conn.execute(
        "ALTER TABLE meeting_notes "
        "ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'::jsonb"
    )
    # Three-layer storage: PostgreSQL is the system of record and now holds
    # the embedding itself (JSONB) plus its provenance (model, dim) and a
    # content_hash used to skip unchanged writes. Added additively here so
    # tables deployed before this change gain the columns without any drop
    # or data loss. embedding is NULLABLE: a row with no embedding keeps it
    # NULL, exactly as it kept weaviate_id NULL before.
    # Active-vs-historical flag: added additively so pre-existing
    # deployments gain the column (defaulting every current row to active).
    await conn.execute(
        "ALTER TABLE email_clusters "
        "ADD COLUMN IF NOT EXISTS is_active SMALLINT NOT NULL DEFAULT 1"
    )
    # Raw Microsoft Graph meeting occurrence id this cluster anchored to,
    # kept distinct from meeting_id (which may hold a per-occurrence series
    # tag). Added additively so pre-existing deployments gain the column.
    # Lets a live meeting arriving with its Graph id resolve back to its
    # cluster(s) during pre-meeting brief / meeting-match even when
    # meeting_id stores a series tag.
    await conn.execute(
        "ALTER TABLE email_clusters "
        "ADD COLUMN IF NOT EXISTS anchored_meeting_id TEXT DEFAULT ''"
    )
    # One-time reconciliation BEFORE the partial unique index is created:
    # pre-existing deployments have many rows per meeting that all defaulted
    # to is_active = 1 when the column was added. The invariant allows at
    # most one active cluster per (non-empty) meeting_id, so keep only the
    # NEWEST active cluster per meeting (by updated_at, cluster_id as a
    # stable tiebreak) and flip the older duplicates to inactive. Idempotent
    # and a no-op once the data already conforms, so it is safe to run on
    # every startup. Must precede the index creation below, otherwise the
    # CREATE UNIQUE INDEX fails on the pre-existing duplicates.
    await conn.execute(
        """
        UPDATE email_clusters AS ec
           SET is_active = 0,
               updated_at = now()
          FROM (
                SELECT cluster_id,
                       row_number() OVER (
                           PARTITION BY meeting_id
                           ORDER BY updated_at DESC NULLS LAST, cluster_id DESC
                       ) AS rn
                  FROM email_clusters
                 WHERE is_active = 1
                   AND meeting_id <> ''
               ) ranked
         WHERE ec.cluster_id = ranked.cluster_id
           AND ranked.rn > 1
        """
    )
    for _table in ("email_clusters", "meeting_embeddings", "meeting_history"):
        await conn.execute(
            f"ALTER TABLE {_table} ADD COLUMN IF NOT EXISTS embedding JSONB"
        )
        await conn.execute(
            f"ALTER TABLE {_table} "
            f"ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT ''"
        )
        await conn.execute(
            f"ALTER TABLE {_table} "
            f"ADD COLUMN IF NOT EXISTS embedding_dim INTEGER DEFAULT 0"
        )
        await conn.execute(
            f"ALTER TABLE {_table} "
            f"ADD COLUMN IF NOT EXISTS content_hash TEXT DEFAULT ''"
        )
    # Migrate any pre-existing FAISS-era faiss_id columns to weaviate_id
    # BEFORE creating the new indexes below (which reference weaviate_id).
    await _migrate_legacy_faiss_id_columns(conn)
    for _name, ddl in EMAIL_INDEX_DDL:
        await conn.execute(ddl)
 
 
# ---------------------------------------------------------------------------
# bu_emails — the chatbot's raw-message RAG source table
# ---------------------------------------------------------------------------
# This table sits outside the cluster-synthesis storage model above. It holds
# one row per individual email message fetched from Microsoft Graph, written
# by the email pipeline (``agent_email/postgres_store.save_emails_to_bu_emails``)
# and read by the chatbot (``chatbot/postgres_store.fetch_email_rows``).
#
# It is therefore created unconditionally at startup — NOT gated behind
# ``EMAIL_STORAGE_BACKEND=postgres``, which governs only the cluster
# pipeline's own persistence choice. The chatbot needs this table whenever
# PostgreSQL is configured at all.
#
# Column contract (must stay in sync with the chatbot's SELECT):
#   conversation_id  Graph ``conversationId`` — groups replies/forwards into
#                    one thread. The chatbot filters out blank values.
#   message_id       Graph message ``id`` — globally stable, and the dedup
#                    key. A UNIQUE index on it makes every write an upsert,
#                    so re-running the pipeline over an unchanged mailbox
#                    refreshes rows in place instead of duplicating them.
#   received_at      TIMESTAMPTZ (not TEXT) so the chatbot's
#                    ``ORDER BY received_at ASC NULLS LAST`` sorts
#                    chronologically rather than lexically. The read side
#                    stringifies it, so this needs no change there.
#   attachment_data  " || "-joined "<filename>: <ocr_text>" for the
#                    attachments belonging to THIS message only.
#   to_address       ", "-joined Graph ``toRecipients`` addresses for the
#                    message (the "to" the pipeline fetched via Graph).
 
# PostgreSQL identifiers cannot be passed as bound parameters, so the
# configurable table name is interpolated into the DDL. Validate it against
# a strict identifier pattern first so a malformed/hostile CHATBOT_PG_TABLE
# can never inject SQL.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
 
# Default table name. The real value comes from Settings.chatbot_pg_table.
BU_EMAILS_DEFAULT_TABLE = "bu_emails"
 
 
def validate_table_name(table: str) -> str:
    """Return `table` if it is a safe bare SQL identifier, else raise.
 
    Guards every code path that interpolates the configurable chatbot table
    name into a SQL string.
    """
    name = (table or "").strip()
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe SQL table name {table!r}: must match "
            f"[A-Za-z_][A-Za-z0-9_]* (check CHATBOT_PG_TABLE)."
        )
    return name
 
 
def bu_emails_table_ddl(table: str = BU_EMAILS_DEFAULT_TABLE) -> str:
    """CREATE TABLE IF NOT EXISTS statement for the chatbot's source table."""
    name = validate_table_name(table)
    return f"""
        CREATE TABLE IF NOT EXISTS {name} (
            id              BIGSERIAL PRIMARY KEY,
            conversation_id TEXT        NOT NULL,
            message_id      TEXT        NOT NULL,
            subject         TEXT        DEFAULT '',
            raw_content     TEXT        DEFAULT '',
            attachment_data TEXT        DEFAULT '',
            received_at     TIMESTAMPTZ,
            from_address    TEXT        DEFAULT '',
            to_address      TEXT        DEFAULT '',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """
 
 
def bu_emails_index_ddl(table: str = BU_EMAILS_DEFAULT_TABLE) -> list[tuple[str, str]]:
    """(index_name, DDL) pairs for the chatbot's source table.
 
    The UNIQUE index on ``message_id`` is load-bearing: it is the conflict
    target for the pipeline's ``ON CONFLICT (message_id) DO UPDATE`` upsert.
    Without it, repeat pipeline runs would append duplicate rows.
    """
    name = validate_table_name(table)
    return [
        (
            # Deliberately prefixed "uq_" rather than "idx_": an older
            # externally-created table may already have a NON-UNIQUE
            # "idx_<table>_message_id". CREATE INDEX IF NOT EXISTS matches
            # on the index NAME, not its definition, so reusing that name
            # would silently skip creating the unique index and leave the
            # upsert without a conflict target.
            f"uq_{name}_message_id",
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{name}_message_id "
            f"ON {name} (message_id);",
        ),
        (
            f"idx_{name}_conversation_id",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_conversation_id "
            f"ON {name} (conversation_id);",
        ),
        (
            f"idx_{name}_received_at",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_received_at "
            f"ON {name} (received_at);",
        ),
    ]
 
 
async def create_bu_emails_schema(
    conn, table: str = BU_EMAILS_DEFAULT_TABLE
) -> None:
    """Create the chatbot's raw-message source table + indexes.

    Idempotent (every statement is IF NOT EXISTS), so it is safe to run on
    every application startup.

    Tolerates a table that pre-exists from an external loader: missing
    bookkeeping columns are backfilled, and the UNIQUE index on
    ``message_id`` is attempted in its own savepoint so that legacy
    duplicate rows (which make it impossible to create) degrade into a
    warning instead of failing the whole schema setup. Callers detect that
    case with ``bu_emails_has_unique_message_id`` and fall back to a
    non-upsert write path.
    """
    name = validate_table_name(table)
    await conn.execute(bu_emails_table_ddl(name))
    # Backfill columns for tables that pre-exist from an external loader
    # with only the seven columns the chatbot reads.
    for column, definition in (
        ("to_address", "TEXT DEFAULT ''"),
        ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ("updated_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ("is_active", "SMALLINT NOT NULL DEFAULT 1"),
    ):
        await conn.execute(
            f"ALTER TABLE {name} ADD COLUMN IF NOT EXISTS {column} {definition}"
        )
    for index_name, ddl in bu_emails_index_ddl(name):
        # Each index gets its own savepoint (asyncpg maps a nested
        # transaction to SAVEPOINT). Without this, a single failing
        # CREATE UNIQUE INDEX - which happens when a pre-existing table
        # already holds duplicate message_ids - would abort the caller's
        # surrounding transaction and roll back the table and the other
        # indexes along with it.
        try:
            async with conn.transaction():
                await conn.execute(ddl)
        except Exception as e:
            _logger.warning(
                "Could not create index %s on '%s': %s", index_name, name, e
            )


# ---------------------------------------------------------------------------
# chatbot_embeddings — cached per-chunk embedding vectors for the chatbot
# ---------------------------------------------------------------------------

CHATBOT_EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS chatbot_embeddings (
    message_id      TEXT PRIMARY KEY,
    embedding       JSONB,
    embedding_model TEXT    DEFAULT '',
    embedding_dim   INTEGER DEFAULT 0,
    content_hash    TEXT    DEFAULT '',
    weaviate_id     TEXT    DEFAULT '',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

CHATBOT_EMBEDDINGS_INDEX_DDL = [
    (
        "idx_chatbot_embeddings_content_hash",
        "CREATE INDEX IF NOT EXISTS idx_chatbot_embeddings_content_hash "
        "ON chatbot_embeddings (content_hash);",
    ),
]


async def create_chatbot_embeddings_schema(conn) -> None:
    """Create the chatbot_embeddings table + indexes. Idempotent."""
    await conn.execute(CHATBOT_EMBEDDINGS_DDL)
    for index_name, ddl in CHATBOT_EMBEDDINGS_INDEX_DDL:
        try:
            async with conn.transaction():
                await conn.execute(ddl)
        except Exception as e:
            _logger.warning(
                "Could not create index %s on chatbot_embeddings: %s",
                index_name, e,
            )


async def has_unique_single_column_index(conn, table: str, column: str) -> bool:
    """True if a unique index/constraint covers exactly ``column``.

    ``ON CONFLICT (<column>)`` raises unless such an index exists, so the
    writer uses this to pick a valid conflict target (or to fall back to a
    plain INSERT when there is none).
    """
    return bool(
        await conn.fetchval(
            """
            SELECT 1
            FROM pg_index idx
            JOIN pg_class  tbl ON tbl.oid = idx.indrelid
            JOIN pg_attribute att
                 ON att.attrelid = tbl.oid
                 AND att.attnum = idx.indkey[0]
            WHERE tbl.relname = $1
              AND idx.indisunique
              AND idx.indnatts = 1
              AND att.attname = $2
            LIMIT 1;
            """,
            table,
            column,
        )
    )


async def bu_emails_has_unique_message_id(conn, table: str) -> bool:
    """Backwards-compatible shim for the message_id-specific check."""
    return await has_unique_single_column_index(conn, table, "message_id")


async def bu_emails_existing_columns(conn, table: str) -> set[str]:
    """Column names actually present on the live table.

    The chatbot's source table may predate this feature. An externally
    created one (e.g. the standalone chatbot project's 25-column layout)
    has extra NOT NULL columns such as a ``unique_id`` primary key, and
    uses different names for the body/subject fields. The writer
    introspects rather than assuming, so it can populate whatever the real
    table requires instead of failing on a column it never knew about.
    """
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = $1",
        table,
    )
    return {r["column_name"] for r in rows}
