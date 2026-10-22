"""
migrations/migrate_email_to_postgres.py
========================================
Creates the email agent's PostgreSQL tables + indexes and (optionally)
copies existing CSV data into PostgreSQL + Weaviate.

Run from the backend/ directory:

    python migrations/migrate_email_to_postgres.py            # schema only
    python migrations/migrate_email_to_postgres.py --copy-csv # schema + data

What it does
------------
1. Reads POSTGRES_DSN (and the WEAVIATE_* settings / BU_EMAIL_DATA_DIR)
   from backend/.env via Settings.
2. Connects to PostgreSQL and creates the four email tables + their
   indexes (IF NOT EXISTS — safe to run repeatedly).
3. With --copy-csv: reads the existing CSVs (email_clusters.csv,
   email_topics.csv, meeting_embeddings.csv, meeting_history.csv), inserts
   the ROW data into PostgreSQL inside a single transaction, and writes the
   inline JSON EMBEDDINGS into Weaviate (clusters + meetings namespaces),
   keyed by a deterministic weaviate_id per row.

The weaviate_id is a deterministic UUID (version 5) derived from the
row's natural key (cluster_id / meeting_id), so re-running the copy
overwrites the same Weaviate object instead of duplicating.

This migration does not use FAISS or any other vector store as a
fallback — Weaviate must be reachable for --copy-csv to succeed.

Prerequisites
-------------
  - POSTGRES_DSN set in backend/.env
  - WEAVIATE_* connection settings set in backend/.env
  - pip install asyncpg weaviate-client numpy  (already in requirements.txt)
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import asyncpg  # noqa: E402
import numpy as np  # noqa: E402

from core.config import get_settings  # noqa: E402
from agent_email.db_schema import create_email_schema  # noqa: E402
from agent_email import weaviate_vector_store as wvs  # noqa: E402
from agent_email.weaviate_vector_store import (  # noqa: E402
    NAMESPACE_CLUSTERS,
    NAMESPACE_MEETINGS,
)
from agent_email.storage_ids import stable_object_id  # noqa: E402


def _decode_embedding(raw: str) -> np.ndarray | None:
    if not raw:
        return None
    try:
        values = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(values, list) or not values:
        return None
    return np.array(values, dtype=np.float32)


async def _copy_clusters(conn, settings) -> tuple[int, int]:
    path = settings.resolved_bu_email_data_dir / "email_clusters.csv"
    if not path.exists():
        print(f"  {path.name} not found — skipping")
        return 0, 0
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    vectors: list[tuple[str, np.ndarray]] = []
    inserted = 0
    async with conn.transaction():
        for row in rows:
            cluster_id = (row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            weaviate_id = None
            vec = _decode_embedding(row.get("cluster_embedding") or "")
            if vec is not None:
                weaviate_id = stable_object_id(cluster_id)
                vectors.append((weaviate_id, vec))
            await conn.execute(
                """
                INSERT INTO email_clusters
                    (cluster_id, department, topic, summary, top_questions,
                     action_items, email_refs, email_count, weaviate_id,
                     meeting_id, meeting_start, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11, now())
                ON CONFLICT (cluster_id) DO UPDATE SET
                    department=EXCLUDED.department, topic=EXCLUDED.topic,
                    summary=EXCLUDED.summary, top_questions=EXCLUDED.top_questions,
                    action_items=EXCLUDED.action_items, email_refs=EXCLUDED.email_refs,
                    email_count=EXCLUDED.email_count, weaviate_id=EXCLUDED.weaviate_id,
                    meeting_id=EXCLUDED.meeting_id, meeting_start=EXCLUDED.meeting_start,
                    updated_at=now()
                """,
                cluster_id,
                row.get("department") or "",
                row.get("topic") or "",
                row.get("summary") or "",
                row.get("top_questions") or "",
                row.get("action_items") or "",
                row.get("email_refs") or "[]",
                int(row.get("email_count") or 0),
                weaviate_id,
                row.get("meeting_id") or "",
                row.get("meeting_start") or "",
            )
            inserted += 1
    # Weaviate write happens only after the DB transaction commits.
    written = wvs.upsert_vectors(NAMESPACE_CLUSTERS, vectors, settings=settings)
    return inserted, written


async def _copy_meeting_embeddings(conn, settings) -> tuple[int, int]:
    path = settings.resolved_bu_email_data_dir / "meeting_embeddings.csv"
    if not path.exists():
        print(f"  {path.name} not found — skipping")
        return 0, 0
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    vectors: list[tuple[str, np.ndarray]] = []
    inserted = 0
    async with conn.transaction():
        for row in rows:
            meeting_id = (row.get("meeting_id") or "").strip()
            if not meeting_id:
                continue
            weaviate_id = None
            vec = _decode_embedding(row.get("embedding") or "")
            if vec is not None:
                weaviate_id = stable_object_id(f"meeting:{meeting_id}")
                vectors.append((weaviate_id, vec))
            await conn.execute(
                """
                INSERT INTO meeting_embeddings (meeting_id, subject, weaviate_id, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (meeting_id) DO UPDATE SET
                    subject=EXCLUDED.subject, weaviate_id=EXCLUDED.weaviate_id, updated_at=now()
                """,
                meeting_id, row.get("subject") or "", weaviate_id,
            )
            inserted += 1
    written = wvs.upsert_vectors(NAMESPACE_MEETINGS, vectors, settings=settings)
    return inserted, written


async def _copy_meeting_history(conn, settings) -> tuple[int, int]:
    path = settings.resolved_bu_email_data_dir / "meeting_history.csv"
    if not path.exists():
        print(f"  {path.name} not found — skipping")
        return 0, 0
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    vectors: list[tuple[str, np.ndarray]] = []
    inserted = 0
    async with conn.transaction():
        for row in rows:
            meeting_id = (row.get("meeting_id") or "").strip()
            if not meeting_id:
                continue
            weaviate_id = None
            vec = _decode_embedding(row.get("embedding") or "")
            if vec is not None:
                # meeting_history shares the "meetings" namespace but uses a
                # distinct id prefix so it never collides with meeting_embeddings.
                weaviate_id = stable_object_id(f"history:{meeting_id}")
                vectors.append((weaviate_id, vec))
            await conn.execute(
                """
                INSERT INTO meeting_history
                    (meeting_id, subject, start_datetime, weaviate_id, updated_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (meeting_id) DO UPDATE SET
                    subject=EXCLUDED.subject, start_datetime=EXCLUDED.start_datetime,
                    weaviate_id=EXCLUDED.weaviate_id, updated_at=now()
                """,
                meeting_id, row.get("subject") or "",
                row.get("start_datetime") or "", weaviate_id,
            )
            inserted += 1
    written = wvs.upsert_vectors(NAMESPACE_MEETINGS, vectors, settings=settings)
    return inserted, written


async def _copy_topics(conn, settings) -> int:
    path = settings.resolved_bu_email_data_dir / "email_topics.csv"
    if not path.exists():
        print(f"  {path.name} not found — skipping")
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    inserted = 0
    async with conn.transaction():
        # Topics are a full-replace snapshot per run; clear then reload.
        await conn.execute("TRUNCATE email_topics")
        for row in rows:
            await conn.execute(
                """
                INSERT INTO email_topics
                    (message_id, subject, from_address, received_at, topic_key, run_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                row.get("message_id") or "", row.get("subject") or "",
                row.get("from_address") or "", row.get("received_at") or "",
                row.get("topic_key") or "", row.get("run_id") or "",
            )
            inserted += 1
    return inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate email agent data to PostgreSQL.")
    parser.add_argument(
        "--copy-csv",
        action="store_true",
        help="Also copy existing CSV row data + embeddings into PostgreSQL/Weaviate.",
    )
    args = parser.parse_args()

    settings = get_settings()
    dsn = (settings.postgres_dsn or "").strip()
    if not dsn:
        print("ERROR: POSTGRES_DSN is empty in backend/.env")
        print("Set it like: postgresql://user:PASSWORD@host:5432/dbname")
        sys.exit(1)

    print("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(dsn=dsn)
    print("Connected.\n")

    try:
        print("Creating email tables + indexes (IF NOT EXISTS)...")
        async with conn.transaction():
            await create_email_schema(conn)
        print("  email_clusters, email_topics, meeting_embeddings, meeting_history  OK\n")

        if args.copy_csv:
            print(f"Copying CSV data from {settings.resolved_bu_email_data_dir} ...")
            wvs.clear_cache()
            c_rows, c_vecs = await _copy_clusters(conn, settings)
            print(f"  email_clusters:      {c_rows} rows, {c_vecs} vectors -> Weaviate")
            t_rows = await _copy_topics(conn, settings)
            print(f"  email_topics:        {t_rows} rows")
            me_rows, me_vecs = await _copy_meeting_embeddings(conn, settings)
            print(f"  meeting_embeddings:  {me_rows} rows, {me_vecs} vectors -> Weaviate")
            mh_rows, mh_vecs = await _copy_meeting_history(conn, settings)
            print(f"  meeting_history:     {mh_rows} rows, {mh_vecs} vectors -> Weaviate")
            print(
                f"\n  Weaviate: http={settings.weaviate_http_host}:{settings.weaviate_http_port} "
                f"grpc={settings.weaviate_grpc_host}:{settings.weaviate_grpc_port}"
            )
    finally:
        await conn.close()

    print("\n" + "=" * 52)
    print("EMAIL POSTGRES MIGRATION COMPLETE")
    print("Set EMAIL_STORAGE_BACKEND=postgres in backend/.env to use it.")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())
