"""
agent_email/storage_ids.py
==========================
Deterministic Weaviate object id derivation for the email agent's
PostgreSQL + Weaviate storage backend.

PostgreSQL rows store a ``weaviate_id`` (TEXT, a UUID string) that keys
into the Weaviate collection where the embedding vector lives (see
``agent_email/weaviate_vector_store.py``). That id is derived
deterministically from the row's natural key (cluster_id / meeting_id)
so that:

* re-running the pipeline for the same key overwrites the same Weaviate
  object instead of leaking duplicate vectors, and
* a row and its vector can be re-linked without a round-trip (the id is a
  pure function of the key).

We use UUID version 5 (name-based, SHA-1), which is the standard way to
derive a stable, collision-resistant UUID from an arbitrary string — the
same string always produces the same UUID, and Weaviate accepts any valid
UUID as an object id.
"""

from __future__ import annotations

import uuid

# Fixed namespace UUID for this application's email-agent vector ids.
# Generated once and frozen — changing this would change every derived id.
_NAMESPACE = uuid.UUID("6f1b1a4a-9c2e-4e3b-8f0a-6b2c6a1e2d7f")


def stable_object_id(key: str) -> str:
    """
    Return a deterministic UUID string (version 5) for `key`.

    The same string always maps to the same id within and across runs.
    Different namespaces should namespace their keys (e.g. ``"meeting:<id>"``
    vs ``"history:<id>"``) before calling this so they never collide inside
    a shared Weaviate collection.
    """
    return str(uuid.uuid5(_NAMESPACE, key))
