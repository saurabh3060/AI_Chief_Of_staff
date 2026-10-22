"""
chatbot/_vec_backend.py
========================
Returns the active chatbot vector store module (``faiss_store`` or
``weaviate_store``) based on the ``CHATBOT_VECTOR_DB`` setting.

Usage::

    from chatbot._vec_backend import vec_store
    store = vec_store()
    store.ensure_collection(client)
"""

from __future__ import annotations


def vec_store():
    """Return the active chatbot vector-store module."""
    from core.config import get_settings
    if get_settings().use_faiss:
        from chatbot import faiss_store
        return faiss_store
    from chatbot import weaviate_store
    return weaviate_store
