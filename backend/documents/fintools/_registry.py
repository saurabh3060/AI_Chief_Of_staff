"""Tool registry: each tool is a plain Python function (sync or async) plus the JSON schema
a chat model sees.

Decorating a function with @tool registers it; fintools/__init__.py imports every tool module so
the registry is complete on `import fintools`.
"""
import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    fn: Callable
    spec: dict          # OpenAI-style {"type": "function", "function": {...}} for tool calling
    needs_con: bool     # the function takes a `con` keyword the model never sees


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict, required: tuple[str, ...] = (),
         needs_con: bool = False):
    def register(fn: Callable) -> Callable:
        spec = {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)}}}
        TOOLS[name] = Tool(name, fn, spec, needs_con)
        return fn
    return register


async def dispatch(name: str, args: dict | None, con=None) -> dict:
    """Run one tool call.  Awaits async tools and runs sync tools inline.

    Never raises: a bad call comes back as ``{"error": ...}`` so a tool-calling loop can hand
    it to the model to retry.  Arguments the function doesn't take are dropped (small models
    invent extra keys), and ``con`` is only passed to tools that read the store.
    """
    t = TOOLS.get(name)
    if t is None:
        return {"error": f"no such tool: {name}. Available: {', '.join(TOOLS)}"}
    params = inspect.signature(t.fn).parameters
    kwargs = {k: v for k, v in (args or {}).items() if k in params and k != "con"}
    if t.needs_con:
        kwargs["con"] = con
    try:
        if inspect.iscoroutinefunction(t.fn):
            return await t.fn(**kwargs)
        return t.fn(**kwargs)
    except (TypeError, ValueError, KeyError, ZeroDivisionError) as e:
        return {"error": f"{name}: {e}"}
