"""
tests/agent_email/test_pipeline_runtime_options.py
=====================================================
EMAIL_PIPELINE_OLLAMA_KEEP_ALIVE / EMAIL_PIPELINE_OLLAMA_NUM_THREAD are opt-in:
unset, the request sent to Ollama is exactly what it was before; set, they are
added to the chat model and to the embedding request.
"""

from types import SimpleNamespace

import pytest

from core.utils import parse_ollama_keep_alive


def test_keep_alive_parsing():
    assert parse_ollama_keep_alive("-1") == -1
    assert parse_ollama_keep_alive("300") == 300
    assert parse_ollama_keep_alive("30m") == "30m"


def _chat_settings(**over):
    values = dict(
        ollama_host="http://ollama.invalid", ollama_temperature=0.2, ollama_num_ctx=16384,
        email_pipeline_ollama_keep_alive="", email_pipeline_ollama_num_thread=0,
    )
    values.update(over)
    return SimpleNamespace(**values)


@pytest.fixture
def chat_model_kwargs(monkeypatch):
    client_module = pytest.importorskip("core.ollama_client.client")
    seen = {}

    def fake_chat_ollama(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(client_module, "ChatOllama", fake_chat_ollama)

    def build(**settings_over):
        client_module.OllamaClient(settings=_chat_settings(**settings_over))._build_chat_model("m", num_predict=64)
        return dict(seen)

    return build


def test_chat_model_request_is_unchanged_by_default(chat_model_kwargs):
    assert chat_model_kwargs() == {
        "model": "m", "base_url": "http://ollama.invalid", "temperature": 0.2,
        "timeout": 600, "num_ctx": 16384, "num_predict": 64,
    }


def test_chat_model_gets_keep_alive_and_threads_only_when_set(chat_model_kwargs):
    kwargs = chat_model_kwargs(email_pipeline_ollama_keep_alive="-1", email_pipeline_ollama_num_thread=8)
    assert kwargs["keep_alive"] == -1 and kwargs["num_thread"] == 8
    assert chat_model_kwargs(email_pipeline_ollama_keep_alive="30m")["keep_alive"] == "30m"


@pytest.fixture
def embed_request(monkeypatch):
    from agent_email import semantic_match

    sent = {}

    class FakeEmbedClient:
        def embed(self, **kwargs):
            sent.clear()
            sent.update(kwargs)
            return {"embeddings": [[0.1, 0.2]]}

    monkeypatch.setattr(semantic_match, "_get_embed_client", lambda: FakeEmbedClient())

    def run(**settings_over):
        monkeypatch.setattr(semantic_match, "get_settings", lambda: _chat_settings(**settings_over))
        semantic_match._embed_sync("text", 8192, "nomic-embed-text")
        return dict(sent)

    return run


def test_embedding_request_is_unchanged_by_default(embed_request):
    assert embed_request() == {"model": "nomic-embed-text", "input": "text", "options": {"num_ctx": 8192}}


def test_embedding_request_gets_keep_alive_and_threads_only_when_set(embed_request):
    sent = embed_request(email_pipeline_ollama_keep_alive="-1", email_pipeline_ollama_num_thread=4)
    assert sent["options"] == {"num_ctx": 8192, "num_thread": 4} and sent["keep_alive"] == -1
