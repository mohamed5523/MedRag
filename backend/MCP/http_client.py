"""Shared HTTP client utilities with retry logic for the clinic MCP server."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

import httpx

from config import get_settings

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=get_settings().request_timeout)
    return _client


async def _retryable_request(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    auth: Optional[Tuple[str, str]] = None,
) -> httpx.Response:
    settings = get_settings()
    delay = settings.retry_backoff
    last_error: Optional[Exception] = None

    for attempt in range(settings.max_retries):
        try:
            response = await _get_client().request(
                method,
                url,
                params=params,
                auth=auth,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500:
                raise RuntimeError(
                    f"Clinic API responded with {exc.response.status_code} ({url})"
                ) from exc
            last_error = exc
        except httpx.RequestError as exc:
            last_error = exc

        if attempt < settings.max_retries - 1:
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError(
        f"Unable to reach Clinic API after {settings.max_retries} attempts"
    ) from last_error


async def fetch_text(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    auth: Optional[Tuple[str, str]] = None,
) -> str:
    response = await _retryable_request(
        "GET",
        url,
        params=params,
        auth=auth,
    )
    return response.text


async def shutdown_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

