"""
evaluation/base_evaluator.py
-----------------------------
Abstract base class for all MedRAG evaluators.

Each eval_*.py module inherits from BaseEvaluator and implements run_eval().
The provider injection mechanism decouples evaluation from any specific backend:
  - Pass provider_fn=None  → use the default HTTP backend (BACKEND_URL from config)
  - Pass provider_fn=<fn>  → use any callable that accepts the same args, e.g. a
                              locally hosted model, a different API, or a test stub.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from .config import BACKEND_URL, is_mock_mode


class BaseEvaluator(ABC):
    """Abstract base for all evaluators.

    Subclasses must implement:
      - component (class attribute): str  — name used in results dict
      - _mock_run()  → dict              — runs evaluation in mock mode
      - _real_run(provider_fn)  → dict   — runs evaluation against a real provider
    """

    component: str = "base"

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_eval(
        self,
        mock: bool = False,
        provider_fn: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """Run evaluation and return a results dict.

        Args:
            mock:        If True (or EVAL_MOCK env var is set) use synthetic data.
            provider_fn: Optional callable to replace the default HTTP backend.
                         Signature varies per evaluator (see subclass docstring).
        """
        mock = mock or is_mock_mode()
        start = time.monotonic()

        try:
            if mock:
                result = self._mock_run()
            else:
                result = self._real_run(provider_fn=provider_fn)
        except Exception as exc:
            result = {
                "component": self.component,
                "error": str(exc),
                "score": 0.0,
            }

        result.setdefault("component", self.component)
        result["evaluation_time_s"] = round(time.monotonic() - start, 3)
        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_provider(
        self,
        provider_fn: Optional[Callable[..., Any]],
        mock: bool,
    ) -> Callable[..., Any]:
        """Return the provider callable to use.

        Priority:
          1. provider_fn (explicit injection)
          2. default HTTP backend wrapper built from BACKEND_URL
        If mock=True, callers should never reach this — they call _mock_run directly.
        """
        if provider_fn is not None:
            return provider_fn
        return self._default_http_provider()

    def _default_http_provider(self) -> Callable[..., Any]:
        """Return a generic HTTP provider.  Subclasses may override for custom routes."""
        import httpx  # lazy import — not needed in mock mode

        base_url = BACKEND_URL

        def _call(path: str, **kwargs) -> Any:
            resp = httpx.post(f"{base_url}{path}", timeout=30.0, **kwargs)
            return resp

        return _call

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def _mock_run(self) -> Dict[str, Any]:
        """Run evaluation using synthetic / pre-stored data (no network)."""
        ...

    @abstractmethod
    def _real_run(
        self, provider_fn: Optional[Callable[..., Any]] = None
    ) -> Dict[str, Any]:
        """Run evaluation against a real provider.

        Args:
            provider_fn: If provided, use this callable instead of the default
                         HTTP backend.  The exact signature is defined per subclass.
        """
        ...
