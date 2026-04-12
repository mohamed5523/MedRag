from __future__ import annotations

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, ConfigDict

try:  # pragma: no cover - fallback for test environments without pydantic-settings
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    class BaseSettings(BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(extra="ignore")

    def SettingsConfigDict(**_kwargs):
        return {}  # type: ignore[return-value]

# Ensure .env is loaded before reading individual vars
load_dotenv()


class MCPBasicAuth(BaseModel):
    """Optional HTTP basic-auth credentials for the MCP server."""

    username: str
    password: str

    model_config = ConfigDict(extra="ignore")


class MCPSettings(BaseSettings):
    """Runtime configuration for the internal MCP aggregation server.

    Architecture
    ------------
    The backend never calls the external clinic API (192.0.0.192 / 41.32.47.162) directly.
    Instead it routes **all** requests through the internal MCP server container, which:
      - Proxies provider list / schedule / pricing to the real clinic API
        (configured via CLINIC_PROVIDER_LIST_URL, CLINIC_PROVIDER_SCHEDULE_URL,
        CLINIC_SERVICE_PRICE_URL, CLINIC_API_USERNAME, CLINIC_API_PASSWORD in .env)
      - Exposes richer matching endpoints (/providers/match, /clinics/match) locally

    In Docker the MCP server is reached at ``http://mcp-server:8000``
    (injected by docker-compose via MCP_BASE_URL).
    In local dev it defaults to ``http://localhost:8020``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    enabled: bool = True

    # ── Single base URL for the internal MCP aggregation server ────────────────
    # In Docker: overridden to http://mcp-server:8000 by docker-compose.yml
    # Locally:   http://localhost:8020  (port-mapped in docker-compose)
    base_url: AnyHttpUrl = "http://localhost:8020"

    # ── MCP server route paths (must match clinic_server.py custom_route paths) ─
    provider_list_path: str = "/providers"
    provider_schedule_path: str = "/providers/schedule"
    service_price_path: str = "/providers/services/pricing"

    # ── Optional full-URL overrides (take precedence over base_url + path) ─────
    provider_list_url: Optional[AnyHttpUrl] = None
    provider_schedule_url: Optional[AnyHttpUrl] = None
    service_price_url: Optional[AnyHttpUrl] = None

    # ── HTTP client settings ────────────────────────────────────────────────────
    api_key: Optional[str] = None
    basic_auth_username: Optional[str] = None
    basic_auth_password: Optional[str] = None
    request_timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5

    @property
    def basic_auth(self) -> Optional[MCPBasicAuth]:
        if self.basic_auth_username:
            return MCPBasicAuth(
                username=self.basic_auth_username,
                password=self.basic_auth_password or "",
            )
        return None


@lru_cache(maxsize=1)
def get_mcp_settings() -> MCPSettings:
    """Return a cached settings instance."""

    return MCPSettings()

