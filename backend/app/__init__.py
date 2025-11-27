"""MedRAG backend package."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RedisClient": (".core.redis_client", "RedisClient"),
    "redis_client": (".core.redis_client", "redis_client"),
    "SessionManager": (".core.session_manager", "SessionManager"),
    "session_manager": (".core.session_manager", "session_manager"),
    "ShortTermMemoryStore": (".core.conversation_memory", "ShortTermMemoryStore"),
    "short_term_memory": (".core.conversation_memory", "short_term_memory"),
    "MemoryMessage": (".core.conversation_memory", "MemoryMessage"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - default AttributeError path
        raise AttributeError(f"module 'app' has no attribute '{name}'") from exc
    module = import_module(module_path, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value