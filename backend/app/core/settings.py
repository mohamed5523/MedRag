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
    """Runtime configuration for the external MCP clinic server."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    enabled: bool = True
    base_url: AnyHttpUrl = "http://localhost:8020"
    provider_list_path: str = "/providers"
    provider_schedule_path: str = "/providers/schedule"
    service_price_path: str = "/providers/services/pricing"
    provider_list_url: Optional[AnyHttpUrl] = None
    provider_schedule_url: Optional[AnyHttpUrl] = None
    service_price_url: Optional[AnyHttpUrl] = None
    api_key: Optional[str] = None
    basic_auth_username: Optional[str] = None
    basic_auth_password: Optional[str] = None
    request_timeout_seconds: float = 20.0
    connect_timeout_seconds: float = 5.0
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

