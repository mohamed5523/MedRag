from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import httpx
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.settings import MCPSettings, get_mcp_settings

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.mcp_client")

DAY_NAME_TO_ID = {
    "saturday": 1,
    "sunday": 2,
    "monday": 3,
    "tuesday": 4,
    "wednesday": 5,
    "thursday": 6,
    "friday": 7,
}


class MCPClientError(Exception):
    """Generic MCP client exception."""


class MCPResponseValidationError(MCPClientError):
    """Raised when a response payload cannot be validated."""


class MCPBaseModel(BaseModel):
    """Base model that allows passthrough fields for future schema changes."""

    model_config = ConfigDict(extra="allow")


class ProviderRecord(MCPBaseModel):
    clinic_id: Optional[int] = None
    clinic_name_ar: Optional[str] = None
    clinic_name_en: Optional[str] = None
    provider_id: Optional[int] = None
    provider_name_ar: Optional[str] = None
    provider_name_en: Optional[str] = None
    specialty: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, ProviderRecord):
            return data.model_dump()
        if not isinstance(data, dict):
            raise TypeError("ProviderRecord expects a mapping.")

        raw = dict(data)
        return {
            "clinic_id": _coerce_int(_pick(raw, ["clinic_id", "clinicid", "clinicId", "clinic"])),
            "clinic_name_ar": _pick(
                raw,
                ["clinic_name_ar", "clinicArabicName", "clinicnamear", "clinic_arabic", "clinicarabicname"],
            ),
            "clinic_name_en": _pick(
                raw,
                ["clinic_name_en", "clinicEnglishName", "clinicnameen", "clinic_name"],
            ),
            "provider_id": _coerce_int(_pick(raw, ["provider_id", "providerid", "providerId", "doctorid"])),
            "provider_name_ar": _pick(
                raw,
                ["provider_name_ar", "providerArabicName", "providernamear", "doctorarabicname"],
            ),
            "provider_name_en": _pick(
                raw,
                ["provider_name_en", "providerEnglishName", "providernameen", "doctorname"],
            ),
            "specialty": _pick(
                raw,
                ["specialty", "speciality", "specialization", "provider_specialty"],
            ),
            "raw": raw,
        }

    def matches_provider(self, name: str) -> bool:
        """Match provider name with flexible partial matching."""
        if not name:
            return False
        
        # Normalize search term
        search_normalized = _normalize_arabic(name.casefold())
        search_words = set(search_normalized.split())
        
        # Check both Arabic and English names
        for candidate in (self.provider_name_ar, self.provider_name_en):
            if not candidate:
                continue
            
            candidate_normalized = _normalize_arabic(candidate.casefold())
            
            # Method 1: Substring match (original behavior)
            if search_normalized in candidate_normalized:
                return True
            
            # Method 2: Word-level matching (any search word in any candidate word)
            candidate_words = set(candidate_normalized.split())
            if search_words & candidate_words:  # Set intersection
                return True
            
            # Method 3: Any search word is substring of any candidate word
            for search_word in search_words:
                if any(search_word in cand_word for cand_word in candidate_words):
                    return True
        
        return False

    def matches_clinic(self, name: str) -> bool:
        """Match clinic name with flexible partial matching."""
        if not name:
            return False
        
        # Normalize search term
        search_normalized = _normalize_arabic(name.casefold())
        search_words = set(search_normalized.split())
        
        # Check both Arabic and English names
        for candidate in (self.clinic_name_ar, self.clinic_name_en):
            if not candidate:
                continue
            
            candidate_normalized = _normalize_arabic(candidate.casefold())
            
            # Method 1: Substring match
            if search_normalized in candidate_normalized:
                return True
            
            # Method 2: Word-level matching
            candidate_words = set(candidate_normalized.split())
            if search_words & candidate_words:
                return True
            
            # Method 3: Partial word matching
            for search_word in search_words:
                if any(search_word in cand_word for cand_word in candidate_words):
                    return True
        
        return False


class ScheduleSlot(MCPBaseModel):
    clinic_id: Optional[int] = None
    provider_id: Optional[int] = None
    day_id: Optional[int] = None
    day_name: Optional[str] = None
    shift_start: Optional[str] = None
    shift_end: Optional[str] = None
    notes: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, ScheduleSlot):
            return data.model_dump()
        if not isinstance(data, dict):
            raise TypeError("ScheduleSlot expects a mapping.")

        raw = dict(data)
        return {
            "clinic_id": _coerce_int(_pick(raw, ["clinic_id", "clinicid", "clinicId"])),
            "provider_id": _coerce_int(_pick(raw, ["provider_id", "providerid", "providerId"])),
            "day_id": _coerce_int(_pick(raw, ["dayid", "day_id", "dayId"])),
            "day_name": _pick(raw, ["day", "day_name", "dayName", "dayArabicName"]),
            "shift_start": _pick(raw, ["shift_start", "start_time", "startTime", "shiftstart"]),
            "shift_end": _pick(raw, ["shift_end", "end_time", "endTime", "shiftend"]),
            "notes": _pick(raw, ["notes", "remarks"]),
            "raw": raw,
        }


class ServicePriceRecord(MCPBaseModel):
    clinic_id: Optional[int] = None
    provider_id: Optional[int] = None
    service_name_ar: Optional[str] = None
    service_name_en: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, ServicePriceRecord):
            return data.model_dump()
        if not isinstance(data, dict):
            raise TypeError("ServicePriceRecord expects a mapping.")

        raw = dict(data)
        return {
            "clinic_id": _coerce_int(_pick(raw, ["clinic_id", "clinicid", "clinicId"])),
            "provider_id": _coerce_int(_pick(raw, ["provider_id", "providerid", "providerId"])),
            "service_name_ar": _pick(
                raw,
                ["service_name_ar", "serviceArabicName", "servicear"],
            ),
            "service_name_en": _pick(
                raw,
                ["service_name_en", "serviceEnglishName", "serviceen", "servicename"],
            ),
            "price": _coerce_float(_pick(raw, ["price", "service_price", "amount"])),
            "currency": _pick(raw, ["currency", "currency_code"]),
            "raw": raw,
        }


class ProviderListPayload(BaseModel):
    providers: List[ProviderRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, ProviderListPayload):
            return data.model_dump()
        if isinstance(data, list):
            return {"providers": data}
        if isinstance(data, dict):
            if "providers" in data:
                return {"providers": data["providers"]}
            if "data" in data:
                return {"providers": data["data"]}
        raise TypeError("Provider list response must be a list or contain 'providers'.")

    def find_provider(self, provider_name: str, clinic_name: Optional[str] = None) -> Optional[ProviderRecord]:
        if not provider_name:
            return None

        for record in self.providers:
            if not record.matches_provider(provider_name):
                continue
            if clinic_name and not record.matches_clinic(clinic_name):
                continue
            return record
        return None

    def find_clinic(self, clinic_name: str) -> Optional[ProviderRecord]:
        if not clinic_name:
            return None
        for record in self.providers:
            if record.matches_clinic(clinic_name):
                return record
        return None
    
    def find_provider_fuzzy(
        self, 
        provider_name: str, 
        clinic_name: Optional[str] = None,
        top_k: int = 3
    ) -> List[tuple[ProviderRecord, float]]:
        """
        Find providers with confidence scores using fuzzy matching.
        Returns list of (provider, confidence_score) tuples, sorted by score.
        
        Args:
            provider_name: Provider name to search for
            clinic_name: Optional clinic name to filter by
            top_k: Maximum number of results to return
            
        Returns:
            List of (ProviderRecord, confidence_score) tuples
        """
        if not provider_name:
            return []
        
        candidates: List[tuple[ProviderRecord, float]] = []
        
        for record in self.providers:
            # Filter by clinic if specified
            if clinic_name and not record.matches_clinic(clinic_name):
                continue
            
            # Calculate similarity to both Arabic and English names
            max_score = 0.0
            
            if record.provider_name_ar:
                score_ar = _calculate_similarity(provider_name, record.provider_name_ar)
                max_score = max(max_score, score_ar)
            
            if record.provider_name_en:
                score_en = _calculate_similarity(provider_name, record.provider_name_en)
                max_score = max(max_score, score_en)
            
            if max_score > 0.3:  # Minimum threshold
                candidates.append((record, max_score))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def find_clinic_fuzzy(
        self, 
        clinic_name: str,
        top_k: int = 3
    ) -> List[tuple[ProviderRecord, float]]:
        """
        Find clinics with confidence scores using fuzzy matching.
        Returns list of (provider_record, confidence_score) tuples, sorted by score.
        Ensures unique clinics by keeping only the best-scoring variant.
        
        Args:
            clinic_name: Clinic name to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of (ProviderRecord, confidence_score) tuples
        """
        if not clinic_name:
            return []
        
        # Use dict to track best score per clinic_id
        best_by_clinic: Dict[int, tuple[ProviderRecord, float]] = {}
        
        for record in self.providers:
            clinic_id = record.clinic_id
            if not clinic_id:
                continue
            
            max_score = 0.0
            if record.clinic_name_ar:
                score_ar = _calculate_similarity(clinic_name, record.clinic_name_ar)
                max_score = max(max_score, score_ar)
            
            if record.clinic_name_en:
                score_en = _calculate_similarity(clinic_name, record.clinic_name_en)
                max_score = max(max_score, score_en)
            
            # Only consider matches above minimum threshold
            if max_score > 0.3:
                # Keep this variant only if it's the best for this clinic_id
                if clinic_id not in best_by_clinic or max_score > best_by_clinic[clinic_id][1]:
                    best_by_clinic[clinic_id] = (record, max_score)
        
        # Convert to list and sort by score
        candidates = list(best_by_clinic.values())
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


class ProviderScheduleResponse(BaseModel):
    slots: List[ScheduleSlot] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, ProviderScheduleResponse):
            return data.model_dump()
        if isinstance(data, list):
            return {"slots": data}
        if isinstance(data, dict):
            if "slots" in data:
                return {"slots": data["slots"]}
            if "data" in data:
                return {"slots": data["data"]}
        raise TypeError("Schedule response must be a list or contain 'slots'.")


class ServicePriceResponse(BaseModel):
    services: List[ServicePriceRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, ServicePriceResponse):
            return data.model_dump()
        if isinstance(data, list):
            return {"services": data}
        if isinstance(data, dict):
            if "services" in data:
                return {"services": data["services"]}
            if "data" in data:
                return {"services": data["data"]}
        raise TypeError("Service price response must be a list or contain 'services'.")


class MCPClient:
    """Async HTTP client for the clinic MCP server."""

    def __init__(self, settings: Optional[MCPSettings] = None):
        self.settings = settings or get_mcp_settings()

    async def get_clinic_provider_list(self) -> ProviderListPayload:
        if not self.settings.enabled:
            raise MCPClientError("MCP client is disabled via configuration.")

        url = self._build_url(self.settings.provider_list_url, self.settings.provider_list_path)
        with tracer.start_as_current_span("mcp.get_clinic_provider_list") as span:
            span.set_attribute("mcp.url", url)
            payload = await self._request_json(url, span=span)
            try:
                result = ProviderListPayload.model_validate(payload)
            except ValidationError as exc:
                raise MCPResponseValidationError(str(exc)) from exc
            span.set_attribute("mcp.providers.count", len(result.providers))
            return result

    async def get_clinic_provider_schedule(
        self,
        clinic_id: int,
        provider_id: Optional[int] = None,
        day_id: Optional[int] = None,
    ) -> ProviderScheduleResponse:
        params = {"clinicid": clinic_id}
        if provider_id:
            params["providerid"] = provider_id
        if day_id:
            params["dayid"] = day_id

        url = self._build_url(self.settings.provider_schedule_url, self.settings.provider_schedule_path)
        with tracer.start_as_current_span("mcp.get_clinic_provider_schedule") as span:
            span.set_attribute("mcp.url", url)
            span.set_attribute("mcp.params.clinicid", clinic_id)
            if provider_id:
                span.set_attribute("mcp.params.providerid", provider_id)
            if day_id:
                span.set_attribute("mcp.params.dayid", day_id)
            payload = await self._request_json(url, params=params, span=span)
            try:
                result = ProviderScheduleResponse.model_validate(payload)
            except ValidationError as exc:
                raise MCPResponseValidationError(str(exc)) from exc
            span.set_attribute("mcp.slots.count", len(result.slots))
            return result

    async def get_service_price(
        self,
        clinic_id: int,
        provider_id: Optional[int] = None,
    ) -> ServicePriceResponse:
        params = {"clinicid": clinic_id}
        if provider_id:
            params["providerid"] = provider_id

        url = self._build_url(self.settings.service_price_url, self.settings.service_price_path)
        with tracer.start_as_current_span("mcp.get_service_price") as span:
            span.set_attribute("mcp.url", url)
            span.set_attribute("mcp.params.clinicid", clinic_id)
            if provider_id:
                span.set_attribute("mcp.params.providerid", provider_id)
            payload = await self._request_json(url, params=params, span=span)
            try:
                result = ServicePriceResponse.model_validate(payload)
            except ValidationError as exc:
                raise MCPResponseValidationError(str(exc)) from exc
            span.set_attribute("mcp.services.count", len(result.services))
            return result

    async def lookup_provider_record(
        self,
        provider_name: Optional[str],
        clinic_name: Optional[str] = None,
    ) -> Optional[ProviderRecord]:
        if not provider_name:
            return None
        provider_list = await self.get_clinic_provider_list()
        return provider_list.find_provider(provider_name, clinic_name=clinic_name)

    async def lookup_clinic_record(self, clinic_name: Optional[str]) -> Optional[ProviderRecord]:
        if not clinic_name:
            return None
        provider_list = await self.get_clinic_provider_list()
        return provider_list.find_clinic(clinic_name)

    async def _request_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        span=None,
    ) -> Any:
        headers = {}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        auth = None
        if self.settings.basic_auth:
            auth = (self.settings.basic_auth.username, self.settings.basic_auth.password)

        timeout = httpx.Timeout(
            timeout=self.settings.request_timeout_seconds,
            connect=self.settings.connect_timeout_seconds,
        )

        attempts = max(self.settings.max_retries, 0) + 1
        delay = self.settings.retry_backoff_seconds
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url, params=params, headers=headers, auth=auth)
                    response.raise_for_status()
                    span and span.add_event(
                        "mcp.response.received",
                        {
                            "status_code": response.status_code,
                            "attempt": attempt,
                        },
                    )
                    return response.json()

            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning("MCP request failed (%s/%s): %s", attempt, attempts, exc)
                span and span.record_exception(exc)
                if attempt >= attempts:
                    break
                await asyncio.sleep(delay * attempt)

        raise MCPClientError(f"MCP request failed after {attempts} attempts: {last_error}") from last_error

    def _build_url(self, override: Optional[str], path: str) -> str:
        if override:
            return override
        base_url = str(self.settings.base_url).rstrip("/")
        return urljoin(f"{base_url}/", path.lstrip("/"))


def _pick(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text by removing diacritics and extra spaces."""
    if not text:
        return ""
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize spaces
    text = ' '.join(text.split())
    return text.strip()


def _calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity score between two strings (0-1).
    Uses multiple strategies for robust matching.
    """
    from difflib import SequenceMatcher
    
    text1_norm = _normalize_arabic(text1.casefold())
    text2_norm = _normalize_arabic(text2.casefold())
    
    if not text1_norm or not text2_norm:
        return 0.0
    
    scores = []
    
    # 1. Sequence matcher (character-level similarity)
    seq_score = SequenceMatcher(None, text1_norm, text2_norm).ratio()
    scores.append(seq_score * 0.5)
    
    # 2. Word overlap (word-level similarity)
    words1 = set(text1_norm.split())
    words2 = set(text2_norm.split())
    if words1 and words2:
        overlap = len(words1 & words2) / len(words1 | words2)
        scores.append(overlap * 0.5)
    
    return sum(scores)

