"""
core/faiss/
===========
TEMPORARY local FAISS vector store for the email cluster pipeline.

This whole package exists only so the email agent (``agent_email``) can run
its cluster/meeting vector storage + search against a local on-disk FAISS
index instead of the bank's shared Weaviate deployment, selected by
``Settings.vector_store_backend == "faiss"``. The chatbot is NOT affected.

It is intentionally self-contained so it can be removed cleanly later:

  * Delete this ``core/faiss/`` folder.
  * Delete ``agent_email/vector_store_backend.py`` and revert the
    ``vector_store_backend as wvs`` imports back to
    ``weaviate_vector_store as wvs`` in the agent_email runtime files.
  * Remove ``VECTOR_STORE_BACKEND`` / ``FAISS_INDEX_DIR`` from config +
    the ``.env`` files and ``faiss-cpu`` from requirements.txt.

Nothing in the Weaviate path imports this package.
"""
