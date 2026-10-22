"""
tests/agent_email/test_pipeline_cache.py
==========================================
Exact-match LLM cache used by the email pipeline. A fake in-memory Redis stands
in for the real one; it deliberately has NO delete/unlink/flush methods, so any
attempt by the cache to remove a key would fail the test.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent_email import pipeline_cache


class FakeRedis:
    """get/set only. Records the TTL of every SET."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttls[key] = ex


class FakeClient:
    """Stands in for OllamaClient: records calls, exposes generation_params."""

    def __init__(self, replies, **params):
        self.replies = list(replies)
        self.calls = 0
        self.params = {"model": "m1", "temperature": 0.2, "num_ctx": 16384, **params}

    def generation_params(self, num_predict=None):
        return {**self.params, "num_predict": num_predict or 0}

    async def chat(self, prompt, system_prompt=None, num_predict=None):
        self.calls += 1
        return self.replies.pop(0)


def _settings(**over):
    values = dict(
        email_pipeline_cache_enabled=True,
        email_pipeline_cache_prefix="emailagent:",
        email_pipeline_llm_cache_ttl_seconds=600,
        redis_key_prefix="rbl_email:",
    )
    values.update(over)
    return SimpleNamespace(**values)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()

    async def fake_get_redis(settings=None):
        return fake

    monkeypatch.setattr(pipeline_cache, "get_redis", fake_get_redis)
    return fake


def _chat(client, settings, prompt="P", system="S", num_predict=100, use_cache=True):
    return asyncio.run(pipeline_cache.chat_with_cache(
        client, settings, task="t", prompt=prompt, system_prompt=system,
        num_predict=num_predict, use_cache=use_cache,
    ))


# ── keys ──────────────────────────────────────────────────────

def test_key_depends_on_every_input():
    s = _settings()
    base = dict(task="t", system_prompt="S", prompt="P",
                params={"model": "m", "temperature": 0.2, "num_ctx": 1, "num_predict": 5})
    key = pipeline_cache.llm_cache_key(s, **base)
    assert key == pipeline_cache.llm_cache_key(s, **base)           # stable
    assert key.startswith("emailagent:llmcache:t:")
    changed = [
        {**base, "prompt": "P2"},
        {**base, "system_prompt": "S2"},
        {**base, "params": {**base["params"], "model": "m2"}},
        {**base, "params": {**base["params"], "temperature": 0.0}},
        {**base, "params": {**base["params"], "num_ctx": 2}},
        {**base, "params": {**base["params"], "num_predict": 6}},
    ]
    keys = {pipeline_cache.llm_cache_key(s, **c) for c in changed}
    assert key not in keys and len(keys) == len(changed)


@pytest.mark.parametrize("prefix", ["chatbot:qa_cache:", "chatbot:", "rbl_email:", "rbl_email:llm:"])
def test_prefix_inside_the_chatbot_or_app_key_space_disables_the_cache(prefix):
    s = _settings(email_pipeline_cache_prefix=prefix)
    assert pipeline_cache.llm_cache_key(s, task="t", system_prompt="", prompt="P", params={}) is None


def test_cache_can_be_switched_off():
    s = _settings(email_pipeline_cache_enabled=False)
    assert pipeline_cache.llm_cache_key(s, task="t", system_prompt="", prompt="P", params={}) is None


# ── hit / miss ────────────────────────────────────────────────

def test_second_identical_call_is_served_from_the_cache(redis):
    s, client = _settings(), FakeClient(["reply-1", "reply-2"])
    first, ticket = _chat(client, s)
    assert first == "reply-1" and not ticket.hit and client.calls == 1
    asyncio.run(ticket.commit(s, first))

    second, ticket2 = _chat(client, s)
    assert second == "reply-1" and ticket2.hit and client.calls == 1   # no second model call
    assert list(redis.ttls.values()) == [600]                          # stored with the TTL


def test_a_reply_that_is_not_committed_is_never_cached(redis):
    """Callers retry an invalid reply with the same prompt; it must reach the model again."""
    s, client = _settings(), FakeClient(["bad", "good"])
    _chat(client, s)                                    # caller rejects 'bad' -> no commit
    reply, ticket = _chat(client, s)
    assert reply == "good" and client.calls == 2 and not ticket.hit


def test_different_prompt_or_parameters_do_not_share_an_entry(redis):
    s = _settings()
    client = FakeClient(["a", "b", "c"])
    for kwargs in ({}, {"prompt": "P2"}, {"num_predict": 7}):
        reply, ticket = _chat(client, s, **kwargs)
        asyncio.run(ticket.commit(s, reply))
    assert client.calls == 3 and len(redis.data) == 3


def test_use_cache_false_bypasses_a_stored_reply_and_overwrites_it(redis):
    s, client = _settings(), FakeClient(["old", "new"])
    old, ticket = _chat(client, s)
    asyncio.run(ticket.commit(s, old))

    fresh, ticket2 = _chat(client, s, use_cache=False)   # the cached reply failed validation
    assert fresh == "new" and not ticket2.hit and client.calls == 2
    asyncio.run(ticket2.commit(s, fresh))                # plain SET replaces it
    assert _chat(client, s)[0] == "new"


def test_a_cache_hit_is_not_written_again(redis):
    s, client = _settings(), FakeClient(["r"])
    reply, ticket = _chat(client, s)
    asyncio.run(ticket.commit(s, reply))
    redis.ttls.clear()
    _, hit_ticket = _chat(client, s)
    asyncio.run(hit_ticket.commit(s, "r"))
    assert redis.ttls == {}


# ── failure handling ──────────────────────────────────────────

def test_redis_absent_falls_back_to_the_normal_call(monkeypatch):
    async def no_redis(settings=None):
        return None

    monkeypatch.setattr(pipeline_cache, "get_redis", no_redis)
    s, client = _settings(), FakeClient(["r1", "r2"])
    reply, ticket = _chat(client, s)
    asyncio.run(ticket.commit(s, reply))
    assert reply == "r1" and _chat(client, s)[0] == "r2" and client.calls == 2


def test_redis_errors_never_break_the_call(monkeypatch):
    class Broken:
        async def get(self, key):
            raise ConnectionError("down")

        async def set(self, key, value, ex=None):
            raise ConnectionError("down")

    async def broken(settings=None):
        return Broken()

    monkeypatch.setattr(pipeline_cache, "get_redis", broken)
    s, client = _settings(), FakeClient(["r"])
    reply, ticket = _chat(client, s)
    asyncio.run(ticket.commit(s, reply))     # must not raise
    assert reply == "r"


# ── through a real call site: relevance_context and its retry loop ─────────

_GOOD_CONTEXT = (
    '{"business_objective": "Review the plan", "topics": ["plan"], "subtopics": [], '
    '"entities": [], "actions": [], "decisions": [], "context": ""}'
)


def _extract(monkeypatch, client, settings):
    """Run relevance._extract_email_context against a fake model client."""
    from datetime import datetime, timezone

    from agent_email import relevance
    from agent_email.schema import EmailMessage

    monkeypatch.setattr(relevance, "OllamaClient", lambda settings=None: client)
    message = EmailMessage(
        id="m1", subject="Quarterly plan", received_at=datetime.now(timezone.utc),
        body_text="Please review the plan before the meeting.",
    )
    return asyncio.run(relevance._extract_email_context(message, settings, run_id="r"))


def test_relevance_context_second_run_is_a_cache_hit_with_identical_output(redis, monkeypatch):
    s, client = _settings(), FakeClient([_GOOD_CONTEXT])
    first = _extract(monkeypatch, client, s)
    second = _extract(monkeypatch, client, s)
    assert client.calls == 1 and first == second and first["business_objective"] == "Review the plan"


def test_relevance_context_empty_reply_is_retried_and_never_cached(redis, monkeypatch):
    s, client = _settings(), FakeClient(["", _GOOD_CONTEXT])
    first = _extract(monkeypatch, client, s)       # attempt 1 empty -> attempt 2 asks the model again
    assert client.calls == 2 and first["business_objective"] == "Review the plan"
    again = _extract(monkeypatch, client, s)       # only the validated reply was stored
    assert client.calls == 2 and again == first


def test_relevance_context_stale_cached_reply_is_bypassed_and_replaced(redis, monkeypatch):
    from agent_email.prompts_semantics import RELEVANCE_CONTEXT_SYSTEM_PROMPT, build_relevance_context_prompt

    s, client = _settings(), FakeClient([_GOOD_CONTEXT])
    prompt = build_relevance_context_prompt("Quarterly plan", "Please review the plan before the meeting.", "")
    key = pipeline_cache.llm_cache_key(
        s, task="relevance_context", system_prompt=RELEVANCE_CONTEXT_SYSTEM_PROMPT,
        prompt=prompt, params=client.generation_params(1024),
    )
    redis.data[key] = '{"text": "this cached reply no longer parses"}'
    result = _extract(monkeypatch, client, s)
    assert client.calls == 1 and result["business_objective"] == "Review the plan"
    assert _extract(monkeypatch, client, s) == result and client.calls == 1   # overwritten with the good reply


def test_garbage_in_the_cache_is_a_miss(redis):
    s, client = _settings(), FakeClient(["real"])
    key = pipeline_cache.llm_cache_key(
        s, task="t", system_prompt="S", prompt="P", params=client.generation_params(100),
    )
    redis.data[key] = "not json"
    reply, ticket = _chat(client, s)
    assert reply == "real" and not ticket.hit


# ── embeddings ────────────────────────────────────────────────

import numpy as np  # noqa: E402


def _emb_settings(**over):
    return _settings(
        bu_email_pipeline_enabled=True, embedding_model="nomic-embed-text", ollama_num_ctx=8192,
        email_pipeline_embedding_cache_ttl_seconds=900, **over,
    )


def test_embedding_key_depends_on_model_context_and_exact_text():
    s = _emb_settings()
    key = pipeline_cache.embedding_cache_key(s, model="m", num_ctx=1, text="hello")
    assert key == pipeline_cache.embedding_cache_key(s, model="m", num_ctx=1, text="hello")
    others = [
        pipeline_cache.embedding_cache_key(s, model="m2", num_ctx=1, text="hello"),
        pipeline_cache.embedding_cache_key(s, model="m", num_ctx=2, text="hello"),
        pipeline_cache.embedding_cache_key(s, model="m", num_ctx=1, text="hello "),
    ]
    assert key not in others and len(set(others)) == 3 and key.startswith("emailagent:embcache:")
    off = _emb_settings(email_pipeline_cache_enabled=False)
    assert pipeline_cache.embedding_cache_key(off, model="m", num_ctx=1, text="hello") is None


def test_cached_vector_is_bit_identical_to_the_computed_one(redis):
    s = _emb_settings()
    vec = np.random.default_rng(7).standard_normal(768).astype(np.float32)
    key = pipeline_cache.embedding_cache_key(s, model="m", num_ctx=1, text="t")
    asyncio.run(pipeline_cache.embedding_set(s, key, vec))
    got = asyncio.run(pipeline_cache.embedding_get(s, key))
    assert got.dtype == np.float32 and got.tobytes() == vec.tobytes()
    assert redis.ttls[key] == 900


def test_embedding_cache_misses_on_corrupt_or_wrong_size_values(redis):
    s = _emb_settings()
    key = pipeline_cache.embedding_cache_key(s, model="m", num_ctx=1, text="t")
    for bad in ("nope", '{"d": 3, "v": "AAAA"}', '{"d": 0, "v": ""}'):
        redis.data[key] = bad
        assert asyncio.run(pipeline_cache.embedding_get(s, key)) is None


@pytest.fixture
def embed_calls(monkeypatch):
    """Replace the real Ollama embedding call with a deterministic counting fake."""
    from agent_email import semantic_match

    calls = []

    def fake_embed_sync(text, num_ctx, model):
        calls.append(text)
        return np.array([len(text), sum(map(ord, text)) % 97, 1.0], dtype=np.float32)

    monkeypatch.setattr(semantic_match, "_embed_sync", fake_embed_sync)
    return calls


def test_embed_text_reuses_a_stored_vector(redis, embed_calls):
    from agent_email import semantic_match

    s = _emb_settings()
    first = asyncio.run(semantic_match.embed_text("quarterly plan", s))
    second = asyncio.run(semantic_match.embed_text("quarterly plan", s))
    other = asyncio.run(semantic_match.embed_text("another text", s))
    assert embed_calls == ["quarterly plan", "another text"]          # the repeat made no model call
    assert first.tobytes() == second.tobytes() and not np.array_equal(first, other)


def test_embed_text_without_the_cache_calls_the_model_every_time(redis, embed_calls):
    from agent_email import semantic_match

    s = _emb_settings(email_pipeline_cache_enabled=False)
    asyncio.run(semantic_match.embed_text("x", s))
    asyncio.run(semantic_match.embed_text("x", s))
    assert embed_calls == ["x", "x"] and redis.data == {}


def test_department_index_is_rebuilt_from_cached_seed_vectors_after_a_restart(redis, embed_calls):
    """A new process (empty in-memory index) re-embeds nothing and gets identical vectors."""
    from agent_email import dept_centroids

    depts = [SimpleNamespace(name="Alpha", summary="alpha summary"), SimpleNamespace(name="Beta", summary="")]
    s = _emb_settings(bu_departments=depts, department_tagging_example_phrases={"Alpha": ["a1", "a2"]})

    dept_centroids._index_cache.clear()
    first = asyncio.run(dept_centroids.build_department_index(s))
    seeds_embedded = len(embed_calls)
    assert seeds_embedded == 4                                       # alpha summary + a1 + a2, then Beta's name

    dept_centroids._index_cache.clear()                              # simulate a fresh process
    second = asyncio.run(dept_centroids.build_department_index(s))
    assert len(embed_calls) == seeds_embedded                        # nothing re-embedded
    for name in ("Alpha", "Beta"):
        assert first["centroids"][name].tobytes() == second["centroids"][name].tobytes()
        assert first["prototypes"][name].tobytes() == second["prototypes"][name].tobytes()
    dept_centroids._index_cache.clear()
