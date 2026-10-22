"""
core package
============
Shared, agent-agnostic infrastructure used across the entire backend:
configuration loading, logging, retry/backoff, concurrency limiting,
the Microsoft Graph API client, the Ollama client, the storage
abstraction, observability (Langfuse) and the LangGraph orchestrator.

No agent-specific business logic should ever live in this package -
agents import from `core`, `core` never imports from an agent package.
"""
