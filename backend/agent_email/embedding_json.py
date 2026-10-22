"""
agent_email/embedding_json.py
=============================
Embedding <-> JSON-string helpers shared by the PostgreSQL store and the
Weaviate sync/restore layer.

The whole stack passes embeddings around as a JSON float-array **string**
(the shape the business logic expects: ``EmailClusterRecord.cluster_embedding``,
``read_meeting_embeddings -> (subject, embedding_json)``). PostgreSQL stores
that same JSON in a JSONB column; these helpers convert between that string
form and a NumPy vector. Kept in their own tiny module so both
``postgres_store`` and ``weaviate_sync`` can import them without a circular
dependency.
"""

from __future__ import annotations

import json

import numpy as np


def vec_to_json(vec: np.ndarray | None) -> str:
    """Serialize a vector to the JSON float-array string, or '' when None."""
    if vec is None:
        return ""
    return json.dumps([float(x) for x in np.asarray(vec).ravel()])


def json_to_vec(raw: str) -> np.ndarray | None:
    """Parse a JSON float-array string into a float32 vector, or None."""
    if not raw:
        return None
    try:
        values = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(values, list) or not values:
        return None
    return np.array(values, dtype=np.float32)


def jsonb_to_embedding_str(raw) -> str:
    """
    Normalize an ``embedding`` JSONB column value to the JSON float-array
    **string** the callers expect.

    asyncpg returns a JSONB column as a ``str`` unless a codec is set, but
    be defensive: a registered codec (or a list built in Python) would hand
    us a ``list`` instead. Return ``""`` for NULL / empty so a row with no
    embedding looks exactly like it did before (no embedding).
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw if raw.strip() else ""
    if isinstance(raw, (list, tuple)):
        if not raw:
            return ""
        return json.dumps([float(x) for x in raw])
    return ""
