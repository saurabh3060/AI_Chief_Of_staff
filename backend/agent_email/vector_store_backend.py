"""
agent_email/vector_store_backend.py
===================================
TEMPORARY dispatch seam that selects the email agent's vector store at
import time from ``Settings.vector_store_backend``:

    "weaviate" (default) -> agent_email.weaviate_vector_store
    "faiss"              -> core.faiss.vector_store

Both modules expose the IDENTICAL public function surface
(``upsert_vectors`` / ``get_vector`` / ``get_vectors`` / ``search_vectors`` /
``remove_vectors`` / ``clear_namespace_objects`` / ``ensure_schema`` /
``ensure_schema_ready`` / ``object_count`` / ``count`` /
``collection_exists`` / ``cluster_node_count`` / ``startup_report`` /
``clear_cache``) plus the constants ``NAMESPACE_CLUSTERS`` /
``NAMESPACE_MEETINGS`` / ``OWNED_NAMESPACES`` / ``COUNT_MISSING`` /
``COUNT_UNRESOLVABLE``, so the agent_email runtime can simply do::

    from agent_email import vector_store_backend as wvs

and keep calling ``wvs.<name>(...)`` unchanged, whichever backend is active.

Removal (when the FAISS experiment is over): delete ``core/faiss/`` and
this file, then revert the ``vector_store_backend as wvs`` imports in the
agent_email runtime files back to ``weaviate_vector_store as wvs``. Nothing
else references this module.
"""

from __future__ import annotations

from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

# Select the backend ONCE at import time. The process reads one Settings
# instance (get_settings is lru_cached), so the backend is fixed for the
# life of the process — exactly how a deployment-level toggle should behave.
_backend = (get_settings().vector_store_backend or "weaviate").strip().lower()

if _backend == "faiss":
    from core.faiss import vector_store as _impl

    logger.warning(
        "[VECTOR-STORE] email agent using the TEMPORARY local FAISS backend "
        "(VECTOR_STORE_BACKEND=faiss); the chatbot is unaffected"
    )
else:
    from agent_email import weaviate_vector_store as _impl

    logger.info("[VECTOR-STORE] email agent using the Weaviate backend")


# ---- Constants (same names/values in both backends) ----
NAMESPACE_CLUSTERS = _impl.NAMESPACE_CLUSTERS
NAMESPACE_MEETINGS = _impl.NAMESPACE_MEETINGS
OWNED_NAMESPACES = _impl.OWNED_NAMESPACES
COUNT_MISSING = _impl.COUNT_MISSING
COUNT_UNRESOLVABLE = _impl.COUNT_UNRESOLVABLE


# ---- Runtime functions (shared contract) ----
upsert_vectors = _impl.upsert_vectors
get_vector = _impl.get_vector
get_vectors = _impl.get_vectors
search_vectors = _impl.search_vectors
remove_vectors = _impl.remove_vectors
clear_namespace_objects = _impl.clear_namespace_objects
ensure_schema = _impl.ensure_schema
ensure_schema_ready = _impl.ensure_schema_ready
object_count = _impl.object_count
count = _impl.count
collection_exists = _impl.collection_exists
cluster_node_count = _impl.cluster_node_count
startup_report = _impl.startup_report
clear_cache = _impl.clear_cache


def active_backend() -> str:
    """Return the selected backend name ("weaviate" or "faiss")."""
    return _backend
