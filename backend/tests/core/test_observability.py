"""
tests/core/test_observability.py
===================================
Verifies the Langfuse observability toggle: disabled by default (no
callback handler, empty callbacks list), constructs a real handler when
enabled with valid-looking config (mocked, no real network call to
Langfuse), and degrades gracefully (falls back to None/[]) if handler
construction raises.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_observability(required_env):
    """Ensure each test gets a fresh cached callback handler and settings."""
    from core.config import get_settings
    from core.observability import reset_observability_for_tests

    reset_observability_for_tests()
    yield
    reset_observability_for_tests()
    get_settings.cache_clear()


def test_callback_handler_is_none_when_disabled(monkeypatch):
    """With LANGFUSE_ENABLED=false (the default), get_callback_handler() should return None."""
    from core.config import get_settings
    from core.observability import get_callback_handler

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()

    assert get_callback_handler() is None


def test_get_langfuse_callbacks_is_empty_list_when_disabled(monkeypatch):
    """With Langfuse disabled, get_langfuse_callbacks() should return an empty list, safe to splat into callbacks=."""
    from core.config import get_settings
    from core.observability import get_langfuse_callbacks

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()

    assert get_langfuse_callbacks() == []


def test_callback_handler_constructed_when_enabled(monkeypatch):
    """With LANGFUSE_ENABLED=true and a mocked CallbackHandler class, get_callback_handler() should return an instance."""
    from core.config import get_settings

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    get_settings.cache_clear()

    import langfuse.callback as langfuse_callback_module

    class FakeCallbackHandler:
        def __init__(self, public_key, secret_key, host):
            self.public_key = public_key
            self.secret_key = secret_key
            self.host = host

    monkeypatch.setattr(langfuse_callback_module, "CallbackHandler", FakeCallbackHandler)

    from core.observability import get_callback_handler

    handler = get_callback_handler()
    assert handler is not None
    assert handler.public_key == "pk-test"


def test_get_langfuse_callbacks_returns_handler_in_list_when_enabled(monkeypatch):
    """With Langfuse enabled and construction succeeding, get_langfuse_callbacks() should return [handler]."""
    from core.config import get_settings

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    get_settings.cache_clear()

    import langfuse.callback as langfuse_callback_module

    class FakeCallbackHandler:
        def __init__(self, public_key, secret_key, host):
            pass

    monkeypatch.setattr(langfuse_callback_module, "CallbackHandler", FakeCallbackHandler)

    from core.observability import get_langfuse_callbacks

    callbacks = get_langfuse_callbacks()
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], FakeCallbackHandler)


def test_callback_handler_degrades_gracefully_on_init_failure(monkeypatch):
    """If Langfuse's CallbackHandler constructor raises (e.g. bad keys), get_callback_handler() should return None, not raise."""
    from core.config import get_settings

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "invalid")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "invalid")
    get_settings.cache_clear()

    import langfuse.callback as langfuse_callback_module

    def raising_constructor(*args, **kwargs):
        raise ValueError("simulated invalid Langfuse credentials")

    monkeypatch.setattr(langfuse_callback_module, "CallbackHandler", raising_constructor)

    from core.observability import get_callback_handler, get_langfuse_callbacks

    assert get_callback_handler() is None
    assert get_langfuse_callbacks() == []
