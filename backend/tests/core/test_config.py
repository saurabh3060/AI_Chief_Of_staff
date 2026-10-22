"""
tests/core/test_config.py
===========================
Verifies that `core.config.get_settings()` correctly loads configuration
from environment variables and that missing required secrets fail loudly
instead of silently defaulting to empty strings.
"""

import pytest
from pydantic import ValidationError


def test_get_settings_loads_expected_fields(required_env):
    """Settings should reflect the environment variables set by the fixture."""
    from core.config import get_settings

    settings = get_settings()

    assert settings.tenant_id == "test-tenant-id"
    assert settings.client_id == "test-client-id"
    assert settings.client_secret == "test-client-secret"
    # Non-secret fields should retain their documented defaults.
    assert settings.storage_backend == "memory"
    assert settings.graph_auth_mode == "delegated"
    assert settings.ollama_vision_model == "llama3.2-vision:11b"


def test_get_settings_is_cached(required_env):
    """Repeated calls without clearing the cache return the same instance."""
    from core.config import get_settings

    first = get_settings()
    second = get_settings()
    assert first is second


def test_missing_required_secret_raises_clear_error(monkeypatch):
    """
    If TENANT_ID/CLIENT_ID/CLIENT_SECRET are absent from both the
    environment and any accessible `.env` file, constructing `Settings`
    directly should raise a pydantic ValidationError rather than
    proceeding with an empty/None credential.
    """
    from core.config import Settings

    # Ensure none of the required secrets leak in from the real
    # environment or the developer's actual backend/.env file for this
    # specific assertion, by pointing env_file at a nonexistent path and
    # clearing any matching env vars.
    for key in ("TENANT_ID", "CLIENT_ID", "CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, tenant_id="", client_id="", client_secret="", graph_auth_mode="delegated")


def test_local_mode_allows_empty_graph_secrets(monkeypatch):
    """GRAPH_AUTH_MODE=local must start without Azure / Graph credentials."""
    from core.config import Settings

    for key in ("TENANT_ID", "CLIENT_ID", "CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(
        _env_file=None,
        tenant_id="",
        client_id="",
        client_secret="",
        graph_auth_mode="local",
        eml_storage_dir="temp_mail_data",
    )
    assert settings.is_local_mail_mode
    assert settings.resolved_eml_storage_dir.name == "temp_mail_data"
    assert settings.resolved_local_calendar_path.name == "local_calendar.json"


def test_effective_text_model_falls_back_to_vision_model(required_env):
    """When no dedicated text model is set, effective_text_model should equal the vision model."""
    from core.config import get_settings

    settings = get_settings()
    assert settings.ollama_text_model == ""
    assert settings.effective_text_model == settings.ollama_vision_model
