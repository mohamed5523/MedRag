from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum
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


# ------------------------------------------------------------------------------
# Hybrid Doctor Matching Models
# ------------------------------------------------------------------------------

class HybridMatchStatus(str, Enum):
    """Status of a hybrid doctor match operation."""
    UNAMBIGUOUS_MATCH = "UNAMBIGUOUS_MATCH"
    AMBIGUOUS_NEED_MORE_INFO = "AMBIGUOUS_NEED_MORE_INFO"
    NO_MATCH = "NO_MATCH"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class DoctorMatchResult(BaseModel):
    """A single doctor match result with scoring details."""
    provider_id: str
    clinic_id: str
    clinic_name: str
    name_ar: str
    name_en: str
    score: float = Field(..., description="Final similarity score (0–1)")
    token_overlap: float = Field(..., description="Token overlap (0–1)")
    fuzzy_name_score: float = Field(..., description="Fuzzy full-name score (0–1)")
    position_score: float = Field(..., description="Positional score (0–1)")
    matched_by_first_name: bool = Field(..., description="True if first-name was the main match")
    matched_tokens: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class HybridMatchResponse(BaseModel):
    """Response from the hybrid doctor matching operation."""
    status: HybridMatchStatus
    message: str
    query_tokens: List[str]
    best_match: Optional[DoctorMatchResult] = None
    candidates: List[DoctorMatchResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


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
                [
                    "clinic_name_ar",
                    "clinicArabicName",
                    "clinicnamear",
                    "clinic_arabic",
                    "clinicarabicname",
                    "clinicName",
                    "ClinicName",
                ],
            ),
            "clinic_name_en": _pick(
                raw,
                [
                    "clinic_name_en",
                    "clinicEnglishName",
                    "clinicnameen",
                    "clinic_name",
                    "clinicNameEn",
                    "clinicNameL",
                    "clinicNameLatin",
                ],
            ),
            "provider_id": _coerce_int(_pick(raw, ["provider_id", "providerid", "providerId", "doctorid"])),
            "provider_name_ar": _pick(
                raw,
                [
                    "provider_name_ar",
                    "providerArabicName",
                    "providernamear",
                    "doctorarabicname",
                    "doctornamea",
                    "DoctorNameA",
                ],
            ),
            "provider_name_en": _pick(
                raw,
                [
                    "provider_name_en",
                    "providerEnglishName",
                    "providernameen",
                    "doctorname",
                    "DoctorNameL",
                    "doctorNameEnglish",
                ],
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

        search_normalized = _normalize_arabic(name)
        search_core = _remove_stop_words(search_normalized, PROVIDER_STOP_WORDS)
        search_tokens = _tokenize_variants(search_core, PROVIDER_STOP_WORDS)
        if not search_tokens:
            search_tokens = _tokenize_variants(search_normalized)

        for candidate in (self.provider_name_ar, self.provider_name_en):
            if not candidate:
                continue

            candidate_normalized = _normalize_arabic(candidate)
            candidate_core = _remove_stop_words(candidate_normalized, PROVIDER_STOP_WORDS)
            candidate_tokens = _tokenize_variants(candidate_core, PROVIDER_STOP_WORDS)
            if not candidate_tokens:
                candidate_tokens = _tokenize_variants(candidate_normalized)

            if _strings_close(search_core, candidate_core) or _strings_close(search_core, candidate_normalized):
                return True

            if _tokens_overlap(search_tokens, candidate_tokens):
                return True

        return False

    def matches_clinic(self, name: str) -> bool:
        """Match clinic name with flexible partial matching."""
        if not name:
            return False

        search_normalized = _normalize_arabic(name)
        search_core = _remove_stop_words(search_normalized, CLINIC_STOP_WORDS)
        search_tokens = _tokenize_variants(search_core, CLINIC_STOP_WORDS)
        if not search_tokens:
            search_tokens = _tokenize_variants(search_normalized)

        for candidate in (self.clinic_name_ar, self.clinic_name_en):
            if not candidate:
                continue

            candidate_normalized = _normalize_arabic(candidate)
            candidate_core = _remove_stop_words(candidate_normalized, CLINIC_STOP_WORDS)
            candidate_tokens = _tokenize_variants(candidate_core, CLINIC_STOP_WORDS)
            if not candidate_tokens:
                candidate_tokens = _tokenize_variants(candidate_normalized)

            if _strings_close(search_core, candidate_core) or _strings_close(search_core, candidate_normalized):
                return True

            if _tokens_overlap(search_tokens, candidate_tokens):
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
        clinic_id = _coerce_int(_pick(raw, ["clinic_id", "clinicid", "clinicId"]))
        provider_id = _coerce_int(_pick(raw, ["provider_id", "providerid", "providerId"]))
        day_id = _coerce_int(_pick(raw, ["dayid", "day_id", "dayId"]))
        day_name = _pick(raw, ["day", "day_name", "dayName", "dayArabicName"])

        # Try to read explicit start/end fields first
        shift_start = _pick(raw, ["shift_start", "start_time", "startTime", "shiftstart"])
        shift_end = _pick(raw, ["shift_end", "end_time", "endTime", "shiftend"])

        # Fallback: some MCP payloads use a single combined "shifttime" string, e.g. "07.30م-09.00م"
        # If either side is missing, try to recover it from shifttime, without overwriting explicit values.
        if not shift_start or not shift_end:
            shifttime = _pick(raw, ["shifttime", "shiftTime", "shift_time"])
            if isinstance(shifttime, str) and shifttime.strip():
                parts = shifttime.split("-")
                if len(parts) == 2:
                    if not shift_start:
                        shift_start = parts[0].strip()
                    if not shift_end:
                        shift_end = parts[1].strip()
                elif not shift_start and not shift_end:
                    # If we cannot safely split, and neither side is set, keep the whole value in start
                    shift_start = shifttime.strip()

        notes = _pick(raw, ["notes", "remarks"])

        return {
            "clinic_id": clinic_id,
            "provider_id": provider_id,
            "day_id": day_id,
            "day_name": day_name,
            "shift_start": shift_start,
            "shift_end": shift_end,
            "notes": notes,
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
            # Accept both snake_case and PascalCase IDs used by the MCP payload
            "clinic_id": _coerce_int(
                _pick(raw, ["clinic_id", "clinicid", "clinicId", "ClinicId"])
            ),
            "provider_id": _coerce_int(
                _pick(raw, ["provider_id", "providerid", "providerId", "ProviderId"])
            ),
            # Prefer Arabic name; fall back to generic ServiceName if that's all we have
            "service_name_ar": _pick(
                raw,
                [
                    "service_name_ar",
                    "serviceArabicName",
                    "servicear",
                    "ServiceNameAr",
                    "ServiceNameArabic",
                    "ServiceName",
                ],
            ),
            # English / Latin variants; also fall back to ServiceName if present there
            "service_name_en": _pick(
                raw,
                [
                    "service_name_en",
                    "serviceEnglishName",
                    "serviceen",
                    "servicename",
                    "ServiceNameEn",
                    "ServiceNameLatin",
                    "ServiceName",
                ],
            ),
            # Map ServicePrice as well as older keys
            "price": _coerce_float(
                _pick(raw, ["price", "service_price", "amount", "ServicePrice"])
            ),
            "currency": _pick(
                raw,
                ["currency", "currency_code", "Currency", "currencyCode"],
            ),
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
                raw_list = data["data"]
                if isinstance(raw_list, list) and raw_list:
                    first = raw_list[0]
                    if isinstance(first, dict) and "doctors" in first:
                        flattened: List[Dict[str, Any]] = []
                        for clinic in raw_list:
                            clinic_data = clinic if isinstance(clinic, dict) else {}
                            doctors = clinic_data.get("doctors") or []
                            clinic_meta = {k: v for k, v in clinic_data.items() if k != "doctors"}
                            if doctors:
                                for doctor in doctors:
                                    entry = dict(clinic_meta)
                                    if isinstance(doctor, dict):
                                        entry.update(doctor)
                                    flattened.append(entry)
                            else:
                                flattened.append(clinic_meta)
                        return {"providers": flattened}
                return {"providers": raw_list}
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
                raw_list = data["data"]
                # New MCP shape: top-level provider objects with nested "schedules" arrays
                #   {
                #     "providerId": "...",
                #     "clinicId": "...",
                #     "schedules": [{ "dayId": "...", "dayName": "...", "shifttime": "..." }]
                #   }
                if isinstance(raw_list, list) and raw_list:
                    first = raw_list[0]
                    if isinstance(first, dict) and "schedules" in first:
                        flattened: List[Dict[str, Any]] = []
                        for provider in raw_list:
                            provider_data = provider if isinstance(provider, dict) else {}
                            base_meta = {
                                k: v for k, v in provider_data.items() if k != "schedules"
                            }
                            schedules = provider_data.get("schedules") or []
                            if schedules:
                                for schedule in schedules:
                                    entry = dict(base_meta)
                                    if isinstance(schedule, dict):
                                        entry.update(schedule)
                                    flattened.append(entry)
                            else:
                                # Provider record without schedules – keep as-is so we don't lose information
                                flattened.append(base_meta)
                        return {"slots": flattened}
                # Fallback to legacy behavior: assume "data" already contains slot-like entries
                return {"slots": raw_list}
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
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self.settings.request_timeout_seconds,
                connect=self.settings.connect_timeout_seconds
            )
        )

    async def aclose(self):
        """Close the underlying HTTP client."""
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()

    async def get_clinic_provider_list(self) -> ProviderListPayload:
        if not self.settings.enabled:
            raise MCPClientError("MCP client is disabled via configuration.")

        url = self._build_url(self.settings.provider_list_url, self.settings.provider_list_path)
        with tracer.start_as_current_span("mcp.get_clinic_provider_list") as span:
            span.set_attribute("mcp.url", url)
            payload = await self._request_json(url, span=span)
            # Attach a preview of the raw MCP payload for observability in Phoenix
            try:
                span.set_attribute(
                    "mcp.response.preview",
                    json.dumps(payload, ensure_ascii=False)[:1000],
                )
            except Exception:
                # Best-effort only – never fail the request because of logging
                logger.debug("Failed to serialize MCP provider list payload for preview", exc_info=True)
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
            # Attach a preview of the raw MCP payload for observability in Phoenix
            try:
                span.set_attribute(
                    "mcp.response.preview",
                    json.dumps(payload, ensure_ascii=False)[:1000],
                )
            except Exception:
                logger.debug("Failed to serialize MCP schedule payload for preview", exc_info=True)
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
            # Attach a preview of the raw MCP payload for observability in Phoenix
            try:
                span.set_attribute(
                    "mcp.response.preview",
                    json.dumps(payload, ensure_ascii=False)[:1000],
                )
            except Exception:
                logger.debug("Failed to serialize MCP service price payload for preview", exc_info=True)
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

    async def match_doctor_hybrid(
        self,
        query: str,
        clinic_id: Optional[str] = None,
        clinic_name: Optional[str] = None,
        top_k: int = 5,
        min_score_multi: float = 0.6,
        min_score_single: float = 0.55,
    ) -> HybridMatchResponse:
        """
        Match a doctor name using hybrid token-based and fuzzy matching.
        
        This method calls the MCP server's match_doctor_hybrid tool which provides:
        - Arabic and English name support
        - Partial name matching (first name, last name, etc.)
        - Positional weighting (first name matches score higher)
        - Fuzzy matching for typos and variations
        
        Args:
            query: User text containing doctor name or partial name
            clinic_id: Optional filter by exact clinic ID
            clinic_name: Optional filter by clinic name (partial match supported)
            top_k: Maximum number of candidates to return (default: 5)
            min_score_multi: Minimum score for multi-token matches (default: 0.6)
            min_score_single: Minimum score for single-token matches (default: 0.55)
            
        Returns:
            HybridMatchResponse containing:
            - status: UNAMBIGUOUS_MATCH, AMBIGUOUS_NEED_MORE_INFO, or NO_MATCH
            - message: Human-readable message in Arabic
            - query_tokens: Tokenized query
            - best_match: Best matching doctor (if found)
            - candidates: List of candidate matches with scores
        """
        if not self.settings.enabled:
            raise MCPClientError("MCP client is disabled via configuration.")

        # Build URL for the match endpoint
        url = self._build_url(None, "/providers/match")
        
        params = {
            "query": query,
            "top_k": top_k,
            "min_score_multi": min_score_multi,
            "min_score_single": min_score_single,
        }
        if clinic_id:
            params["clinic_id"] = clinic_id
        if clinic_name:
            params["clinic_name"] = clinic_name

        with tracer.start_as_current_span("mcp.match_doctor_hybrid") as span:
            span.set_attribute("mcp.url", url)
            span.set_attribute("mcp.params.query", query)
            if clinic_id:
                span.set_attribute("mcp.params.clinic_id", clinic_id)
            if clinic_name:
                span.set_attribute("mcp.params.clinic_name", clinic_name)
            
            payload = await self._request_json(url, params=params, span=span)
            
            # Attach a preview of the raw MCP payload for observability in Phoenix
            try:
                span.set_attribute(
                    "mcp.response.preview",
                    json.dumps(payload, ensure_ascii=False)[:1000],
                )
            except Exception:
                logger.debug("Failed to serialize MCP match payload for preview", exc_info=True)
            
            try:
                result = HybridMatchResponse.model_validate(payload)
            except ValidationError as exc:
                raise MCPResponseValidationError(str(exc)) from exc
            
            span.set_attribute("mcp.match.status", result.status.value)
            span.set_attribute("mcp.match.candidates_count", len(result.candidates))
            if result.best_match:
                span.set_attribute("mcp.match.best_score", result.best_match.score)
            
            return result

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

        attempts = max(self.settings.max_retries, 0) + 1
        delay = self.settings.retry_backoff_seconds
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                response = await self._http_client.get(url, params=params, headers=headers, auth=auth)
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
    """
    Normalize Arabic text for fuzzy matching by removing diacritics,
    folding common letter variants, and collapsing whitespace.
    This helps us match user typos such as:
        "عيادة الاسنان" ↔ "عيادة الأسنان"
        "دكتور اله" ↔ "دكتور إلـه"
    """
    if not text:
        return ""

    text = text.casefold()

    # Remove tashkeel + tatweel
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)

    # Normalize lam-alef ligatures first (they expand to two characters)
    for ligature in ("ﻻ", "ﻷ", "ﻹ", "ﻵ"):
        text = text.replace(ligature, "لا")

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ۀ": "ه",
        "گ": "ك",
        "ڨ": "ق",
        "چ": "ج",
        "پ": "ب",
        "ژ": "ز",
    }
    translation_table = str.maketrans(replacements)
    text = text.translate(translation_table)

    # Remove leftover punctuation (keep alphanumeric + Arabic letters + space)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


CLINIC_STOP_WORDS: set[str] = {"عياده", "clinic"}
PROVIDER_STOP_WORDS: set[str] = {"دكتور", "dr", "doctor", "دكتوره", "د"}


def _remove_stop_words(text: str, stopwords: set[str]) -> str:
    if not text:
        return ""
    tokens = [token for token in text.split() if token not in stopwords]
    result = " ".join(tokens).strip()
    return result or text


def _tokenize_variants(text: str, stopwords: Optional[set[str]] = None) -> set[str]:
    if not text:
        return set()
    tokens = text.split()
    variants: set[str] = set()
    for token in tokens:
        if stopwords and token in stopwords:
            continue
        variants.add(token)
        if token.startswith("ال") and len(token) > 2:
            variants.add(token[2:])
    return variants


def _strings_close(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _tokens_overlap(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    if left & right:
        return True
    for l_token in left:
        for r_token in right:
            if not l_token or not r_token:
                continue
            if l_token in r_token or r_token in l_token:
                return True
    return False

