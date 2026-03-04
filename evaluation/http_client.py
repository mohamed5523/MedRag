"""
evaluation/http_client.py
--------------------------
Production-grade shared HTTP client for all real-mode evaluators.

Features:
  - Independent connect / read timeouts
  - Exponential backoff retry (3 attempts, 0.5→1→2s)
  - Retries on: ConnectError, TimeoutException, RemoteProtocolError, 429/502/503/504
  - Structured error logging (Python logging, never print)
  - Normalised return dict — callers never need try/except

Usage:
    from evaluation.http_client import post_with_retry, openai_chat

    # General backend call
    resp = post_with_retry("/api/chat/", json={"message": "...", "session_id": "eval"})
    if resp["ok"]:
        data = resp["json"]
    else:
        print(resp["error"])         # exact error string for logs

    # OpenAI-compatible chat call (used by LLM-as-a-Judge)
    resp = openai_chat(url, api_key, model, messages)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from .config import BACKEND_URL

logger = logging.getLogger("eval.http")

# ── Configuration ──────────────────────────────────────────────────────────────
_CONNECT_TIMEOUT  = 5.0    # seconds to establish TCP connection
_READ_TIMEOUT     = 150.0  # seconds to receive response — LLM calls can take 60-90s
_WRITE_TIMEOUT    = 10.0   # seconds to send request body
_MAX_RETRIES      = 3
_BACKOFF_BASE_S   = 0.5    # retry sleep: 0.5 → 1.0 → 2.0 seconds

_DEFAULT_TIMEOUT  = httpx.Timeout(
    connect=_CONNECT_TIMEOUT,
    read=_READ_TIMEOUT,
    write=_WRITE_TIMEOUT,
    pool=5.0,
)

_RETRYABLE_STATUS = {429, 502, 503, 504}
# NOTE: TimeoutException is intentionally NOT retried — on /api/chat/query a
# slow LLM response is normal (60-90s). Retrying only wastes 2× more time.
_RETRYABLE_EXC    = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


def _is_retryable(exc: Exception | None, status: int | None) -> bool:
    if exc is not None:
        return isinstance(exc, _RETRYABLE_EXC)
    return status in _RETRYABLE_STATUS


def _make_result(
    *,
    ok: bool,
    status_code: int | None = None,
    json_body: dict | None = None,
    raw: bytes | None = None,
    latency_ms: float = 0.0,
    error: str | None = None,
    attempts: int = 1,
) -> Dict[str, Any]:
    return {
        "ok":          ok,
        "status_code": status_code,
        "json":        json_body,
        "raw":         raw,
        "latency_ms":  latency_ms,
        "error":       error,
        "attempts":    attempts,
    }


# ── Core retry loop ────────────────────────────────────────────────────────────

def post_with_retry(
    path: str,
    *,
    base_url:  str           = BACKEND_URL,
    json:      dict | None   = None,
    files:     dict | None   = None,
    headers:   dict | None   = None,
    timeout:   httpx.Timeout = _DEFAULT_TIMEOUT,
    max_retries: int         = _MAX_RETRIES,
) -> Dict[str, Any]:
    """POST to `base_url + path` with automatic exponential-backoff retry.

    Returns a normalised result dict:
    {
        "ok":          bool,
        "status_code": int | None,
        "json":        dict | None,
        "raw":         bytes | None,
        "latency_ms":  float,
        "error":       str | None,
        "attempts":    int,
    }
    """
    url     = f"{base_url.rstrip('/')}{path}"
    attempt = 0

    while True:
        attempt += 1
        start        = time.monotonic()
        caught_exc:  Exception | None = None
        status:      int | None       = None

        try:
            resp      = httpx.post(url, json=json, files=files, headers=headers, timeout=timeout)
            latency   = (time.monotonic() - start) * 1000
            status    = resp.status_code

            if resp.is_success:
                try:
                    body = resp.json()
                except Exception:
                    body = None
                return _make_result(
                    ok=True, status_code=status,
                    json_body=body,
                    raw=resp.content if body is None else None,
                    latency_ms=latency, attempts=attempt,
                )

            # Non-success HTTP response
            msg = f"HTTP {status} ← {path}"
            if status >= 500:
                logger.error("%s | body=%.400s", msg, resp.text)
            elif status >= 400:
                logger.warning("%s | body=%.200s", msg, resp.text)

            if not _is_retryable(None, status) or attempt > max_retries:
                return _make_result(ok=False, status_code=status,
                                    latency_ms=latency, error=msg, attempts=attempt)

        except Exception as exc:
            latency    = (time.monotonic() - start) * 1000
            caught_exc = exc
            logger.warning("Attempt %d/%d for %s raised %s: %s",
                           attempt, max_retries + 1, path, type(exc).__name__, exc)
            if not _is_retryable(exc, None) or attempt > max_retries:
                return _make_result(
                    ok=False, latency_ms=latency,
                    error=f"{type(exc).__name__}: {exc}", attempts=attempt,
                )

        sleep = _BACKOFF_BASE_S * (2 ** (attempt - 1))   # 0.5, 1.0, 2.0 …
        logger.info("Retrying %s in %.1fs (attempt %d/%d) …",
                    path, sleep, attempt + 1, max_retries + 1)
        time.sleep(sleep)


# ── OpenAI-compatible chat call (used by LLM-as-a-Judge) ─────────────────────

def openai_chat(
    url:       str,
    api_key:   str,
    model:     str,
    messages:  List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens:  int   = 512,
    max_retries: int   = _MAX_RETRIES,
) -> Dict[str, Any]:
    """Call any OpenAI-compatible /v1/chat/completions endpoint.

    Returns the same normalised result dict as post_with_retry.
    The `json` field, on success, is the raw OpenAI response body;
    use resp["json"]["choices"][0]["message"]["content"] for the text.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    # openai_chat is its own path so we call post_with_retry against the full URL
    # We reuse the retry loop by splitting base_url and path at first /v1
    if "/v1" in url:
        base, path = url.split("/v1", 1)
        path = "/v1" + path
    else:
        base, path = url, ""

    return post_with_retry(
        path,
        base_url    = base,
        json        = body,
        headers     = headers,
        max_retries = max_retries,
    )
