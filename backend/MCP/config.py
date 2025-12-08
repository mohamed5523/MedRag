"""Project-wide configuration and shared constants for the clinic MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

from dotenv import load_dotenv

load_dotenv()

DAY_NAME_TO_ID: Dict[str, int] = {
    "saturday": 1,
    "sunday": 2,
    "monday": 3,
    "tuesday": 4,
    "wednesday": 5,
    "thursday": 6,
    "friday": 7,
}


@dataclass(frozen=True)
class APISettings:
    provider_list_url: str
    provider_schedule_url: str
    service_price_url: str
    username: str
    password: str
    request_timeout: float
    max_retries: int
    retry_backoff: float

    @property
    def auth(self) -> Tuple[str, str]:
        return (self.username, self.password)


def _get_env(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    return APISettings(
        provider_list_url=_get_env(
            "CLINIC_PROVIDER_LIST_URL",
            "http://192.0.0.192:3003/api/clinicProviderlist",
        ),
        provider_schedule_url=_get_env(
            "CLINIC_PROVIDER_SCHEDULE_URL",
            "http://192.0.0.192:3002/api/clinicProviderschedule/",
        ),
        service_price_url=_get_env(
            "CLINIC_SERVICE_PRICE_URL",
            "http://192.0.0.192:3005/api/servicePrice",
        ),
        username=_get_env("CLINIC_API_USERNAME", ""),
        password=_get_env("CLINIC_API_PASSWORD", ""),
        request_timeout=float(_get_env("REQUEST_TIMEOUT", "30")),
        max_retries=max(1, int(_get_env("MAX_RETRIES", "3"))),
        retry_backoff=max(0.1, float(_get_env("REQUEST_RETRY_BACKOFF", "0.5"))),
    )

