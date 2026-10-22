"""
documents/db_schema.py
=======================
PostgreSQL schema for the documents bot (ex-SQLite ``library.db``).

Tables (one row = one logical thing):

- ``documents``          one row per ingested PDF
- ``pages``              one row per page (heading, text, image, raw record)
- ``blocks``             one row per narrative/table/chart region (passage unit)
- ``series``             one row per metric series (metric over periods)
- ``facts``              one row per value in a series x period
- ``issues``             one row per validation flag found during ingestion
- ``corrections``        one row per reviewer-fixed value (survives re-ingestion)
- ``metric_rules``       one row per business metric rule (from workbooks)
- ``calculated_metrics`` one row per computed/derived metric value

Full-text search
----------------
SQLite's FTS5 virtual tables (``blocks_fts``, ``series_fts``) are replaced by
PostgreSQL ``tsvector`` columns on ``blocks`` (over ``text``) and ``series``
(over ``label``), with GIN indexes and auto-populating triggers.

Vector search
-------------
SQLite-vec virtual tables (``blocks_vec``, ``series_vec``) are gone.  Block and
series embeddings live in Weaviate (see ``documents/weaviate_store.py``);
PostgreSQL stores only the row data.
"""

from __future__ import annotations

import asyncpg


async def create_documents_schema(conn: asyncpg.Connection) -> None:
    """Create every documents-bot table, index, trigger if it doesn't exist.

    Called from ``core.db.init_db()`` at startup inside a transaction, so
    all DDL applies atomically.  Safe to run every startup — every object
    uses ``IF NOT EXISTS`` (or an equivalent catalog check for triggers).
    """
    # --- documents --------------------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id       TEXT PRIMARY KEY,
            file_name    TEXT NOT NULL DEFAULT '',
            title        TEXT NOT NULL DEFAULT '',
            doc_type     TEXT NOT NULL DEFAULT '',
            entity       TEXT NOT NULL DEFAULT '',
            as_of_key    TEXT NOT NULL DEFAULT '',
            as_of_label  TEXT NOT NULL DEFAULT '',
            pages        INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL DEFAULT '',
            progress     TEXT NOT NULL DEFAULT '',
            ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # --- pages ------------------------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            doc_id   TEXT NOT NULL,
            page     INTEGER NOT NULL,
            heading  TEXT NOT NULL DEFAULT '',
            text     TEXT NOT NULL DEFAULT '',
            image    TEXT NOT NULL DEFAULT '',
            record   TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (doc_id, page)
        );
    """)

    # --- blocks (with tsvector for FTS) -----------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            block_id         BIGSERIAL PRIMARY KEY,
            doc_id           TEXT NOT NULL,
            page             INTEGER NOT NULL,
            kind             TEXT NOT NULL DEFAULT '',
            section          TEXT NOT NULL DEFAULT '',
            title            TEXT NOT NULL DEFAULT '',
            text             TEXT NOT NULL DEFAULT '',
            content          TEXT,
            bbox             TEXT,
            source           TEXT NOT NULL DEFAULT '',
            confidence       REAL NOT NULL DEFAULT 1.0,
            section_path     TEXT,
            anchor_block_id  BIGINT,
            tsv              TSVECTOR
        );
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS blocks_doc ON blocks(doc_id, page);")
    await conn.execute("CREATE INDEX IF NOT EXISTS blocks_tsv ON blocks USING GIN (tsv);")
    # Trigger to auto-maintain tsv on insert/update.
    await conn.execute("""
        CREATE OR REPLACE FUNCTION blocks_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', COALESCE(NEW.text, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
    await conn.execute("""
        DROP TRIGGER IF EXISTS blocks_tsv_trigger ON blocks;
    """)
    await conn.execute("""
        CREATE TRIGGER blocks_tsv_trigger
        BEFORE INSERT OR UPDATE OF text ON blocks
        FOR EACH ROW EXECUTE FUNCTION blocks_tsv_update();
    """)

    # --- series (with tsvector for FTS on label) --------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS series (
            series_id   BIGSERIAL PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            page        INTEGER NOT NULL,
            block_id    BIGINT,
            label       TEXT NOT NULL DEFAULT '',
            name        TEXT NOT NULL DEFAULT '',
            metric_id   TEXT NOT NULL DEFAULT '',
            kind        TEXT NOT NULL DEFAULT '',
            unit        TEXT NOT NULL DEFAULT '',
            entity      TEXT NOT NULL DEFAULT '',
            source      TEXT NOT NULL DEFAULT '',
            confidence  REAL NOT NULL DEFAULT 1.0,
            path        TEXT,
            scale       REAL,
            tsv         TSVECTOR
        );
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS series_doc ON series(doc_id);")
    await conn.execute("CREATE INDEX IF NOT EXISTS series_metric ON series(metric_id);")
    await conn.execute("CREATE INDEX IF NOT EXISTS series_block ON series(block_id);")
    await conn.execute("CREATE INDEX IF NOT EXISTS series_tsv ON series USING GIN (tsv);")
    await conn.execute("""
        CREATE OR REPLACE FUNCTION series_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', COALESCE(NEW.label, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
    await conn.execute("DROP TRIGGER IF EXISTS series_tsv_trigger ON series;")
    await conn.execute("""
        CREATE TRIGGER series_tsv_trigger
        BEFORE INSERT OR UPDATE OF label ON series
        FOR EACH ROW EXECUTE FUNCTION series_tsv_update();
    """)

    # --- facts ------------------------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id       BIGSERIAL PRIMARY KEY,
            series_id     BIGINT NOT NULL,
            doc_id        TEXT NOT NULL,
            page          INTEGER NOT NULL,
            period_key    TEXT NOT NULL DEFAULT '',
            period_kind   TEXT NOT NULL DEFAULT '',
            period_start  TEXT NOT NULL DEFAULT '',
            period_end    TEXT NOT NULL DEFAULT '',
            period_label  TEXT NOT NULL DEFAULT '',
            partial       INTEGER NOT NULL DEFAULT 0,
            value         DOUBLE PRECISION,
            as_printed    TEXT NOT NULL DEFAULT '',
            validated     INTEGER NOT NULL DEFAULT 0,
            confidence    REAL NOT NULL DEFAULT 1.0,
            note          TEXT NOT NULL DEFAULT ''
        );
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS facts_series ON facts(series_id, period_start);")
    await conn.execute("CREATE INDEX IF NOT EXISTS facts_period ON facts(period_key);")
    await conn.execute("CREATE INDEX IF NOT EXISTS facts_doc ON facts(doc_id, page);")

    # --- issues -----------------------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            issue_id    BIGSERIAL PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            page        INTEGER NOT NULL,
            kind        TEXT NOT NULL DEFAULT '',
            ref         TEXT NOT NULL DEFAULT '',
            message     TEXT NOT NULL DEFAULT '',
            suggestion  TEXT NOT NULL DEFAULT ''
        );
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS issues_doc ON issues(doc_id, page);")

    # --- corrections ------------------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            doc_id         TEXT NOT NULL,
            page           INTEGER NOT NULL,
            series_label   TEXT NOT NULL,
            period_key     TEXT NOT NULL,
            value          DOUBLE PRECISION,
            note           TEXT NOT NULL DEFAULT '',
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (doc_id, page, series_label, period_key)
        );
    """)

    # --- metric_rules -----------------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS metric_rules (
            rule_id              TEXT PRIMARY KEY,
            source_name          TEXT NOT NULL DEFAULT '',
            source_page          INTEGER,
            source_heading       TEXT NOT NULL DEFAULT '',
            document_pattern     TEXT NOT NULL DEFAULT '',
            metric_family        TEXT NOT NULL DEFAULT '',
            metric_name          TEXT NOT NULL DEFAULT '',
            definition           TEXT NOT NULL DEFAULT '',
            applies_to           TEXT NOT NULL DEFAULT '',
            example              TEXT NOT NULL DEFAULT '',
            operation            TEXT NOT NULL DEFAULT '',
            compiled_spec        TEXT NOT NULL DEFAULT '{}',
            aliases              TEXT NOT NULL DEFAULT '[]',
            routing_spec         TEXT NOT NULL DEFAULT '{}',
            interpretation_spec  TEXT NOT NULL DEFAULT '{}',
            output_type          TEXT NOT NULL DEFAULT '',
            enabled              INTEGER NOT NULL DEFAULT 1,
            status               TEXT NOT NULL DEFAULT 'draft',
            review_reason        TEXT NOT NULL DEFAULT '',
            version              INTEGER NOT NULL DEFAULT 1,
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS metric_rules_page "
        "ON metric_rules(source_name, source_page, enabled);"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS metric_rules_name "
        "ON metric_rules(metric_name, enabled);"
    )

    # --- calculated_metrics ----------------------------------------------
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS calculated_metrics (
            result_id                  BIGSERIAL PRIMARY KEY,
            doc_id                     TEXT NOT NULL,
            source_page                INTEGER NOT NULL,
            metric_id                  TEXT NOT NULL,
            metric_name                TEXT NOT NULL,
            metric_family              TEXT NOT NULL,
            entity                     TEXT NOT NULL DEFAULT '',
            period_key                 TEXT NOT NULL DEFAULT '',
            period_label               TEXT NOT NULL DEFAULT '',
            comparison_period_key      TEXT NOT NULL DEFAULT '',
            comparison_period_label    TEXT NOT NULL DEFAULT '',
            value                      DOUBLE PRECISION,
            formatted_value            TEXT NOT NULL DEFAULT '',
            unit                       TEXT NOT NULL DEFAULT '',
            formula                    TEXT NOT NULL DEFAULT '',
            inputs_json                TEXT NOT NULL DEFAULT '',
            source_fact_ids            TEXT NOT NULL DEFAULT '',
            answer                     TEXT NOT NULL DEFAULT '',
            search_text                TEXT NOT NULL DEFAULT '',
            calculation_status         TEXT NOT NULL DEFAULT 'validated',
            calculation_version        INTEGER NOT NULL DEFAULT 1,
            calculated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (doc_id, source_page, metric_id, entity, period_key,
                    comparison_period_key, calculation_version)
        );
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS calculated_metrics_lookup "
        "ON calculated_metrics(doc_id, source_page, metric_id, entity, "
        "period_key, calculation_status);"
    )
