"""Static MCP Server for Clinic Management System."""
from __future__ import annotations

import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from config import DAY_NAME_TO_ID, get_settings
from http_client import fetch_text
from matching_engine import (
    compute_order_score,
    compute_positional_token_weight,
    normalize_arabic,
    normalize_english,
    normalize_mixed_text,
    tokenize_clinic,
    tokenize_name,
)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))


# ------------------------------------------------------------------------------
# Pydantic models for hybrid matching
# ------------------------------------------------------------------------------

class MatchStatus(str, Enum):
    """Status of a doctor match operation."""
    UNAMBIGUOUS_MATCH = "UNAMBIGUOUS_MATCH"
    AMBIGUOUS_NEED_MORE_INFO = "AMBIGUOUS_NEED_MORE_INFO"
    NO_MATCH = "NO_MATCH"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class DoctorMatch(BaseModel):
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


class MatchResponse(BaseModel):
    """Response from the hybrid doctor matching operation."""
    status: MatchStatus
    message: str
    query_tokens: List[str]
    best_match: Optional[DoctorMatch] = None
    candidates: List[DoctorMatch] = Field(default_factory=list)


class DoctorRecord(BaseModel):
    """Internal representation of a doctor for matching."""
    provider_id: str
    clinic_id: str
    clinic_name: str
    name_ar: str
    name_en: str
    norm_name_ar: str
    norm_name_en: str
    tokens: List[str]


class ClinicMatch(BaseModel):
    """A single clinic match result with scoring details."""

    clinic_id: str
    clinic_name: str
    score: float = Field(..., description="Final similarity score (0–1)")
    token_overlap: float = Field(..., description="Token overlap (0–1)")
    fuzzy_name_score: float = Field(..., description="Fuzzy clinic-name score (0–1)")
    order_score: float = Field(..., description="Ordered token score (0–1)")
    matched_tokens: List[str] = Field(default_factory=list)


class ClinicMatchResponse(BaseModel):
    """Response from the hybrid clinic matching operation."""

    status: MatchStatus
    message: str
    query_tokens: List[str]
    best_match: Optional[ClinicMatch] = None
    candidates: List[ClinicMatch] = Field(default_factory=list)


class ClinicRecord(BaseModel):
    """Internal representation of a clinic for matching."""

    clinic_id: str
    clinic_name: str
    norm_name: str
    tokens: List[str]

# Create MCP server instance that also powers the HTTP shim
mcp = FastMCP(
    "Clinic Management System",
    host=SERVER_HOST,
    port=SERVER_PORT,
)
settings = get_settings()

# Provider list cache (helps scalability: avoid refetching on every match call)
_PROVIDER_CACHE_TTL_SECONDS = float(os.getenv("PROVIDER_CACHE_TTL_SECONDS", "60"))
_provider_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}


async def _get_provider_payload() -> Dict[str, Any]:
    """Fetch and cache the provider list payload (JSON-decoded)."""
    now = time.time()
    cached = _provider_cache.get("payload")
    ts = float(_provider_cache.get("ts") or 0.0)
    if cached is not None and (now - ts) <= _PROVIDER_CACHE_TTL_SECONDS:
        return cached

    raw_data = await fetch_text(settings.provider_list_url, auth=settings.auth)
    payload = json.loads(raw_data)
    _provider_cache["payload"] = payload
    _provider_cache["ts"] = now
    return payload


@mcp.tool()
async def get_clinic_provider_list() -> str:
    """Get complete list of all clinics and providers.
    
    This endpoint returns all clinics and providers data to use for filtering schedule.
    No filtering parameters are required.
    
    Returns:
        JSON response containing clinic and provider information with Arabic and Latin names
    """
    return await fetch_text(
        settings.provider_list_url,
        auth=settings.auth,
    )


@mcp.tool()
async def get_clinic_provider_schedule(
    clinicid: int,
    dayid: Optional[int] = None,
    providerid: Optional[int] = None,
    day_name: Optional[str] = None
) -> str:
    """Get clinic provider schedule information.
    
    This endpoint returns a list of clinic providers with their related details and schedules.
    You can filter the results using the available query parameters.
    
    Parameters:
        clinicid (int, required): The Clinic ID you need to retrieve data for
        dayid (int, optional): Day ID from 1-7 (1=Saturday, 2=Sunday, 3=Monday, 4=Tuesday, 5=Wednesday, 6=Thursday, 7=Friday)
        providerid (int, optional): Specific provider doctor ID to filter results
        day_name (str, optional): Day name (e.g., 'Monday', 'Saturday') - will be converted to dayid
    
    Returns:
        JSON response containing provider schedule information including shift times
    """
    if day_name and not dayid:
        dayid = DAY_NAME_TO_ID.get(day_name.lower())
    
    # Build request parameters
    params = {}
    params["clinicid"] = clinicid
    if dayid is not None:
        params["dayid"] = dayid
    if providerid is not None:
        params["providerid"] = providerid
    
    return await fetch_text(
        settings.provider_schedule_url,
        auth=settings.auth,
        params=params,
    )


@mcp.tool()
async def get_service_price(
    clinicid: int,
    providerid: Optional[int] = None
) -> str:
    """Get clinic service pricing information.
    
    This endpoint returns service pricing information for a specific clinic and optionally a specific provider.
    
    Parameters:
        clinicid (int, required): The Clinic ID you need to retrieve data for
        providerid (int, optional): Specific provider doctor ID to filter results
    
    Returns:
        JSON response containing service names, prices, doctor information, and clinic details
    """
    # Build request parameters
    params = {}
    params["clinicid"] = clinicid
    if providerid is not None:
        # params["providerid"] = providerid
        params["providerId"] = providerid 
        
    
    return await fetch_text(
        settings.service_price_url,
        auth=settings.auth,
        params=params,
    )


# ------------------------------------------------------------------------------
# Hybrid Doctor Matching Logic
# ------------------------------------------------------------------------------

def _parse_and_preprocess_providers(data: List[Dict[str, Any]]) -> List[DoctorRecord]:
    """Parse provider list response and preprocess for matching."""
    doctors: List[DoctorRecord] = []
    for clinic in data:
        clinic_id = str(clinic.get("clinicId", "")).strip()
        clinic_name = str(clinic.get("clinicName", "")).strip()
        for d in clinic.get("doctors", []):
            provider_id = str(d.get("providerId", "")).strip()
            if not provider_id:
                continue

            name_ar = (d.get("DoctorNameA") or "").strip()
            name_en = (d.get("DoctorNameL") or "").strip()

            norm_ar = normalize_arabic(name_ar)
            norm_en = normalize_english(name_en)

            tokens = list(
                dict.fromkeys(
                    tokenize_name(name_ar) + tokenize_name(name_en)
                )
            )

            doctors.append(
                DoctorRecord(
                    provider_id=provider_id,
                    clinic_id=clinic_id,
                    clinic_name=clinic_name,
                    name_ar=name_ar,
                    name_en=name_en,
                    norm_name_ar=norm_ar,
                    norm_name_en=norm_en,
                    tokens=tokens,
                )
            )
    return doctors


def _parse_and_preprocess_clinics(data: List[Dict[str, Any]]) -> List[ClinicRecord]:
    """Parse provider list response and preprocess clinic names for matching."""
    clinics: List[ClinicRecord] = []
    seen: set[str] = set()
    for clinic in data:
        clinic_id = str(clinic.get("clinicId", "")).strip()
        clinic_name = str(clinic.get("clinicName", "")).strip()
        if not clinic_id or clinic_id in seen:
            continue
        seen.add(clinic_id)
        clinics.append(
            ClinicRecord(
                clinic_id=clinic_id,
                clinic_name=clinic_name,
                norm_name=normalize_mixed_text(clinic_name),
                tokens=tokenize_clinic(clinic_name),
            )
        )
    return clinics


def _match_clinic_multi_token(
    query_tokens: List[str],
    clinics: List[ClinicRecord],
    top_k: int,
    min_score: float,
) -> ClinicMatchResponse:
    """Match a (tokenized) clinic query against all clinics."""
    q_string = " ".join(query_tokens)
    scored: List[ClinicMatch] = []

    for c in clinics:
        # 1) Token overlap using fuzzy token comparison (continuous, not binary).
        #    This improves recall for typos like "المسا" vs "النسا".
        overlap_sum = 0.0
        matched_tokens: List[str] = []
        for qt in query_tokens:
            best_token_score = 0.0
            for ct in c.tokens:
                s = fuzz.ratio(qt, ct) / 100.0
                if s > best_token_score:
                    best_token_score = s
            overlap_sum += best_token_score
            if best_token_score >= 0.70:
                matched_tokens.append(qt)
        token_overlap = (overlap_sum / len(query_tokens)) if query_tokens else 0.0

        # 2) Ordered sequence score
        order_score = compute_order_score(query_tokens, c.tokens)

        # 3) Fuzzy full-name similarity (WRatio is robust for typos/reordering)
        full = c.norm_name or c.clinic_name
        fuzzy_name_score = fuzz.WRatio(q_string, full) / 100.0 if full else 0.0

        # Final score (tunable weights)
        final_score = 0.45 * token_overlap + 0.20 * order_score + 0.35 * fuzzy_name_score
        if final_score < min_score:
            continue

        scored.append(
            ClinicMatch(
                clinic_id=c.clinic_id,
                clinic_name=c.clinic_name,
                score=round(final_score, 4),
                token_overlap=round(token_overlap, 4),
                fuzzy_name_score=round(fuzzy_name_score, 4),
                order_score=round(order_score, 4),
                matched_tokens=matched_tokens,
            )
        )

    if not scored:
        return ClinicMatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم يتم العثور على عيادة بهذا الاسم.",
            query_tokens=query_tokens,
            candidates=[],
        )

    scored.sort(key=lambda m: (m.score, len(m.matched_tokens)), reverse=True)
    top_matches = scored[:top_k]
    best = top_matches[0]

    if len(top_matches) == 1 or (
        len(top_matches) > 1 and best.score >= 0.80 and (best.score - top_matches[1].score) >= 0.08
    ):
        return ClinicMatchResponse(
            status=MatchStatus.UNAMBIGUOUS_MATCH,
            message="تم العثور على عيادة مطابقة للاسم الذي أدخلته.",
            query_tokens=query_tokens,
            best_match=best,
            candidates=top_matches,
        )

    return ClinicMatchResponse(
        status=MatchStatus.AMBIGUOUS_NEED_MORE_INFO,
        message="يوجد أكثر من عيادة بنفس الاسم أو قريب منه. من فضلك اكتب الاسم بشكل أوضح.",
        query_tokens=query_tokens,
        best_match=best,
        candidates=top_matches,
    )


def _normalize_clinic_for_match(text: str) -> str:
    """Normalize clinic text for forgiving matching (typos/variants, Arabic articles)."""
    t = normalize_mixed_text(text)

    # Common Arabic variants/typos in user input:
    # - "النساؤ" vs "النساء" vs "نسا" (hamza/waaw-hamza forms)
    # - Normalize away standalone hamza forms that often appear in misspellings
    t = t.replace("ؤ", "").replace("ء", "").replace("ئ", "ي")

    # Remove generic clinic words and leading definite article "ال"
    stop = {"عياده", "عيادة", "عيادات", "قسم", "مركز"}
    toks: List[str] = []
    for tok in t.split():
        if tok in stop:
            continue
        if tok.startswith("ال") and len(tok) > 3:
            tok = tok[2:]
        toks.append(tok)

    return " ".join(toks)


def _filter_candidates(
    doctors: List[DoctorRecord],
    clinic_id: Optional[str],
    clinic_name: Optional[str],
) -> List[DoctorRecord]:
    """Filter doctor candidates by clinic ID or name."""
    candidates = doctors
    if clinic_id:
        cid = clinic_id.strip()
        candidates = [d for d in candidates if d.clinic_id == cid]
    if clinic_name:
        norm_c = _normalize_clinic_for_match(clinic_name)
        candidates = [
            d
            for d in candidates
            # Bidirectional substring match: handles cases like
            # "عيادة نسا وتوليد" vs "نسا وتوليد"
            if (
                norm_c in _normalize_clinic_for_match(d.clinic_name)
                or _normalize_clinic_for_match(d.clinic_name) in norm_c
                or (fuzz.partial_ratio(norm_c, _normalize_clinic_for_match(d.clinic_name)) / 100.0)
                >= 0.80
            )
        ]
    return candidates


def _match_single_token(
    token: str,
    candidates: List[DoctorRecord],
    top_k: int,
    min_score: float,
) -> MatchResponse:
    """Match a single-token query against doctor candidates."""
    # Split results into "strong" (>= min_score) and "weak" (near-miss) so we can
    # return LOW_CONFIDENCE suggestions instead of hard NO_MATCH.
    strong: List[DoctorMatch] = []
    weak: List[DoctorMatch] = []

    # Recall-friendly knobs (tunable)
    POS_SIM_BASE = 0.70
    HIT_THRESHOLD = 0.70
    # Respect caller intent: LOW_CONF_MIN must never exceed min_score.
    LOW_CONF_MIN = max(0.0, min_score - 0.20)

    for d in candidates:
        # Positional similarity threshold: be more forgiving for short tokens (common in Arabic surnames)
        pos_threshold = 0.65 if len(token) <= 4 else POS_SIM_BASE
        # Keep threshold logic consistent with positional matching (and with _match_multi_token).
        hit_threshold = 0.65 if len(token) <= 4 else HIT_THRESHOLD
        pos_score, matched_first = compute_positional_token_weight(
            token,
            d.tokens,
            similarity_threshold=pos_threshold,
        )

        # Continuous similarity to any name token (fixes cases like "عبدو" ~ "عبده")
        token_sim = 0.0
        for dt in d.tokens:
            s = fuzz.ratio(token, dt) / 100.0
            if s > token_sim:
                token_sim = s

        full_ar = d.norm_name_ar or d.name_ar
        full_en = d.norm_name_en or d.name_en

        # More forgiving than ratio for partials/variants
        fuzzy_ar = fuzz.WRatio(token, full_ar) / 100.0 if full_ar else 0.0
        fuzzy_en = fuzz.WRatio(token, full_en) / 100.0 if full_en else 0.0
        fuzzy_name_score = max(fuzzy_ar, fuzzy_en)

        # Use continuous overlap rather than binary (max recall)
        token_overlap = token_sim

        # Final score (tunable weights)
        final_score = 0.45 * token_overlap + 0.35 * pos_score + 0.20 * fuzzy_name_score

        if final_score < LOW_CONF_MIN:
            continue

        match = DoctorMatch(
            provider_id=d.provider_id,
            clinic_id=d.clinic_id,
            clinic_name=d.clinic_name,
            name_ar=d.name_ar,
            name_en=d.name_en,
            score=round(final_score, 4),
            token_overlap=round(token_overlap, 4),
            fuzzy_name_score=round(fuzzy_name_score, 4),
            position_score=round(pos_score, 4),
            matched_by_first_name=matched_first,
            matched_tokens=[token] if token_sim >= hit_threshold else [],
        )

        if final_score >= min_score:
            strong.append(match)
        else:
            weak.append(match)

    if not strong:
        if not weak:
            return MatchResponse(
                status=MatchStatus.NO_MATCH,
                message="لم يتم العثور على دكتور بهذا الاسم.",
                query_tokens=[token],
                candidates=[],
            )

        # Return LOW_CONFIDENCE suggestions for near-misses (maximize recall UX)
        weak.sort(
            key=lambda m: (m.score, 1 if m.matched_by_first_name else 0),
            reverse=True,
        )
        top_matches = weak[:top_k]
        best = top_matches[0]
        return MatchResponse(
            status=MatchStatus.LOW_CONFIDENCE,
            message="مش متأكد من الاسم. هل تقصد أحد الأسماء دي؟",
            query_tokens=[token],
            best_match=best,
            candidates=top_matches,
        )

    # Sort by score, then prefer first-name matches as tie-breaker
    strong.sort(
        key=lambda m: (m.score, 1 if m.matched_by_first_name else 0),
        reverse=True,
    )

    top_matches = strong[:top_k]
    first_name_matches = [m for m in top_matches if m.matched_by_first_name]

    # Case 1: exactly one strong first-name match
    if len(first_name_matches) == 1 and first_name_matches[0].score >= 0.8:
        best = first_name_matches[0]
        return MatchResponse(
            status=MatchStatus.UNAMBIGUOUS_MATCH,
            message=f"تم العثور على دكتور واحد مطابق للاسم '{token}'.",
            query_tokens=[token],
            best_match=best,
            candidates=top_matches,
        )

    # Case 2: multiple first-name matches → ask for more info
    if len(first_name_matches) >= 2:
        return MatchResponse(
            status=MatchStatus.AMBIGUOUS_NEED_MORE_INFO,
            message=(
                f"يوجد أكثر من دكتور يبدأ اسمه بـ '{token}'. "
                f"من فضلك اكتب اسم الدكتور مكوّن من اسمين (مثال: {token} + الاسم التاني)."
            ),
            query_tokens=[token],
            candidates=top_matches,
        )

    # Case 3: only last/middle name matches → ask user to clarify
    return MatchResponse(
        status=MatchStatus.AMBIGUOUS_NEED_MORE_INFO,
        message=(
            f"وجدت دكاترة يحملون '{token}' كجزء من الاسم "
            f"وليس كأول اسم. من فضلك اكتب اسمين أوضح للدكتور."
        ),
        query_tokens=[token],
        candidates=top_matches,
    )


def _match_multi_token(
    query_tokens: List[str],
    candidates: List[DoctorRecord],
    top_k: int,
    min_score: float,
) -> MatchResponse:
    """Match a multi-token query against doctor candidates."""
    q_string = " ".join(query_tokens)
    strong: List[DoctorMatch] = []
    weak: List[DoctorMatch] = []

    # Recall-friendly knobs (tunable)
    POS_SIM_BASE = 0.70
    HIT_THRESHOLD = 0.70
    # Respect caller intent: LOW_CONF_MIN must never exceed min_score.
    LOW_CONF_MIN = max(0.0, min_score - 0.20)

    for d in candidates:
        # 1) Continuous token overlap (like clinic matcher): average best fuzzy token score
        overlap_sum = 0.0
        matched_tokens: List[str] = []
        for qt in query_tokens:
            best_token_score = 0.0
            for dt in d.tokens:
                s = fuzz.ratio(qt, dt) / 100.0
                if s > best_token_score:
                    best_token_score = s
            overlap_sum += best_token_score

            # Be more forgiving for short tokens (e.g., "عبدو" vs "عبده")
            hit_threshold = 0.65 if len(qt) <= 4 else HIT_THRESHOLD
            if best_token_score >= hit_threshold:
                matched_tokens.append(qt)
        token_overlap = (overlap_sum / len(query_tokens)) if query_tokens else 0.0

        # 2) Positional score: average positional weight over tokens
        positional_scores: List[float] = []
        any_first = False
        for qt in query_tokens:
            pos_threshold = 0.65 if len(qt) <= 4 else POS_SIM_BASE
            pos_score, matched_first = compute_positional_token_weight(
                qt,
                d.tokens,
                similarity_threshold=pos_threshold,
            )
            positional_scores.append(pos_score)
            any_first = any_first or matched_first
        avg_pos_score = (
            sum(positional_scores) / len(positional_scores)
            if positional_scores
            else 0.0
        )

        # 3) Ordered sequence score
        order_score = compute_order_score(query_tokens, d.tokens)

        # 4) Fuzzy full-name similarity
        full_ar = d.norm_name_ar or d.name_ar
        full_en = d.norm_name_en or d.name_en

        # WRatio is more forgiving for typos/reordering (maximize recall)
        fuzzy_ar = fuzz.WRatio(q_string, full_ar) / 100.0 if full_ar else 0.0
        fuzzy_en = fuzz.WRatio(q_string, full_en) / 100.0 if full_en else 0.0
        fuzzy_name_score = max(fuzzy_ar, fuzzy_en)

        # Final score (tunable weights)
        final_score = (
            0.45 * token_overlap
            + 0.30 * avg_pos_score
            + 0.05 * order_score
            + 0.20 * fuzzy_name_score
        )

        if final_score < LOW_CONF_MIN:
            continue

        match = DoctorMatch(
            provider_id=d.provider_id,
            clinic_id=d.clinic_id,
            clinic_name=d.clinic_name,
            name_ar=d.name_ar,
            name_en=d.name_en,
            score=round(final_score, 4),
            token_overlap=round(token_overlap, 4),
            fuzzy_name_score=round(fuzzy_name_score, 4),
            position_score=round(max(avg_pos_score, order_score), 4),
            matched_by_first_name=any_first,
            matched_tokens=matched_tokens,
        )

        if final_score >= min_score:
            strong.append(match)
        else:
            weak.append(match)

    if not strong:
        if weak:
            weak.sort(
                key=lambda m: (m.score, len(m.matched_tokens), 1 if m.matched_by_first_name else 0),
                reverse=True,
            )
            top_matches = weak[:top_k]
            best = top_matches[0]
            return MatchResponse(
                status=MatchStatus.LOW_CONFIDENCE,
                message="مش متأكد من الاسم. هل تقصد أحد الأسماء دي؟",
                query_tokens=query_tokens,
                best_match=best,
                candidates=top_matches,
            )
        return MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم يتم العثور على دكتور بهذا الاسم.",
            query_tokens=query_tokens,
            candidates=[],
        )

    # Sort by score descending; tie-break: more matched tokens, then first-name usage
    strong.sort(
        key=lambda m: (m.score, len(m.matched_tokens), 1 if m.matched_by_first_name else 0),
        reverse=True,
    )

    top_matches = strong[:top_k]
    best = top_matches[0]

    # Decide ambiguity vs unambiguous
    if len(top_matches) == 1 or (
        len(top_matches) > 1
        and best.score >= 0.8
        and (best.score - top_matches[1].score) >= 0.1
    ):
        return MatchResponse(
            status=MatchStatus.UNAMBIGUOUS_MATCH,
            message="تم العثور على دكتور مطابق للاسم الذي أدخلته.",
            query_tokens=query_tokens,
            best_match=best,
            candidates=top_matches,
        )

    # Multiple close matches → ambiguous
    return MatchResponse(
        status=MatchStatus.AMBIGUOUS_NEED_MORE_INFO,
        message=(
            "يوجد أكثر من دكتور بنفس الاسم أو قريب منه. "
            "من فضلك تأكد من الاسم الكامل أو اكتب التخصص أو العيادة لتحديد الدكتور بدقة."
        ),
        query_tokens=query_tokens,
        best_match=best,
        candidates=top_matches,
    )


@mcp.tool()
async def match_doctor_hybrid(
    query: str,
    clinic_id: Optional[str] = None,
    clinic_name: Optional[str] = None,
    top_k: int = 5,
    min_score_multi: float = 0.6,
    min_score_single: float = 0.55,
) -> str:
    """Match a doctor name using hybrid token-based and fuzzy matching.
    
    This tool provides advanced doctor name matching with support for:
    - Arabic and English names
    - Partial name matching (first name, last name, etc.)
    - Positional weighting (first name matches score higher)
    - Fuzzy matching for typos and variations
    
    Parameters:
        query (str, required): User text containing doctor name or partial name
        clinic_id (str, optional): Filter by exact clinic ID
        clinic_name (str, optional): Filter by clinic name (partial match supported)
        top_k (int, optional): Maximum number of candidates to return (default: 5)
        min_score_multi (float, optional): Minimum score for multi-token matches (default: 0.6)
        min_score_single (float, optional): Minimum score for single-token matches (default: 0.55)
    
    Returns:
        JSON response containing:
        - status: UNAMBIGUOUS_MATCH, AMBIGUOUS_NEED_MORE_INFO, or NO_MATCH
        - message: Human-readable message in Arabic
        - query_tokens: Tokenized query
        - best_match: Best matching doctor (if found)
        - candidates: List of candidate matches with scores
    """
    # Fetch provider list (cached)
    payload = await _get_provider_payload()
    
    if "data" not in payload:
        return json.dumps({
            "status": MatchStatus.NO_MATCH.value,
            "message": "خطأ في جلب بيانات الدكاترة من النظام.",
            "query_tokens": [],
            "best_match": None,
            "candidates": [],
        }, ensure_ascii=False)
    
    # Parse and preprocess doctors
    doctors = _parse_and_preprocess_providers(payload["data"])
    
    # Tokenize query FIRST - check if we can parse the doctor name before filtering
    query_tokens = tokenize_name(query)
    
    # Check query tokenization before filtering by clinic
    # This ensures users get accurate error feedback about the actual problem
    if not query_tokens:
        response = MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم أستطع فهم اسم الدكتور. من فضلك اكتب اسم الدكتور.",
            query_tokens=query_tokens,
        )
        return response.model_dump_json(by_alias=True)
    
    # Filter by clinic if specified (after validating query)
    candidates = _filter_candidates(doctors, clinic_id, clinic_name)
    
    if not candidates:
        response = MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لا يوجد دكاترة في هذا التخصص/العيادة.",
            query_tokens=query_tokens,
        )
        return response.model_dump_json(by_alias=True)
    
    # Choose matching strategy based on token count
    if len(query_tokens) == 1:
        response = _match_single_token(
            query_tokens[0],
            candidates,
            top_k=top_k,
            min_score=min_score_single,
        )
    else:
        response = _match_multi_token(
            query_tokens,
            candidates,
            top_k=top_k,
            min_score=min_score_multi,
        )
    
    return response.model_dump_json(by_alias=True)


@mcp.tool()
async def match_clinic_hybrid(
    query: str,
    top_k: int = 5,
    min_score: float = 0.65,
) -> str:
    """Match a clinic name using hybrid token-based and fuzzy matching."""
    payload = await _get_provider_payload()

    if "data" not in payload:
        return json.dumps(
            {
                "status": MatchStatus.NO_MATCH.value,
                "message": "خطأ في جلب بيانات العيادات من النظام.",
                "query_tokens": [],
                "best_match": None,
                "candidates": [],
            },
            ensure_ascii=False,
        )

    query_tokens = tokenize_clinic(query)
    if not query_tokens:
        response = ClinicMatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم أستطع فهم اسم العيادة. من فضلك اكتب اسم العيادة.",
            query_tokens=[],
        )
        return response.model_dump_json(by_alias=True)

    clinics = _parse_and_preprocess_clinics(payload["data"])

    # ------------------------------------------------------------------
    # Special-case: generic "الجراحة/جراحة" should resolve to base "جراحه"
    # ------------------------------------------------------------------
    # Users often ask: "عيادة الجراحة" as a generic clinic (one-word), and the
    # matcher correctly finds multiple prefixes (جراحه تجميل, جراحه مخ..., ...).
    # For this specific clinic only, we want to skip disambiguation and pick the
    # base clinic record whose tokens are exactly ["جراحه"].
    #
    # Important: apply the rule on *tokenized clinic text*, not on the full
    # user sentence length. This also covers phrases like "الجراحة الاسبوع ده"
    # if the upstream extraction mistakenly includes time context words.
    _SURGERY_TOKEN = "جراحه"
    _SURGERY_CONTEXT_TOKENS = {
        # Common time/context tails that might leak into clinic text
        "اسبوع",
        "شهر",
        "سنه",
        "ده",
        "هذا",
        "هذه",
        "هذة",
        "حالي",
        "قادم",
        "جاي",
    }
    core_tokens = [t for t in query_tokens if t not in _SURGERY_CONTEXT_TOKENS]
    if core_tokens == [_SURGERY_TOKEN]:
        base = next((c for c in clinics if c.tokens == [_SURGERY_TOKEN]), None)
        if base is not None:
            forced = ClinicMatch(
                clinic_id=base.clinic_id,
                clinic_name=base.clinic_name,
                score=1.0,
                token_overlap=1.0,
                fuzzy_name_score=1.0,
                order_score=1.0,
                matched_tokens=[_SURGERY_TOKEN],
            )
            response = ClinicMatchResponse(
                status=MatchStatus.UNAMBIGUOUS_MATCH,
                message="تم العثور على عيادة مطابقة للاسم الذي أدخلته.",
                query_tokens=query_tokens,
                best_match=forced,
                candidates=[forced],
            )
            return response.model_dump_json(by_alias=True)
    response = _match_clinic_multi_token(
        query_tokens=query_tokens,
        clinics=clinics,
        top_k=top_k,
        min_score=min_score,
    )
    return response.model_dump_json(by_alias=True)


def _json_response_from_text(payload: str) -> Response:
    """Return JSONResponse when possible, otherwise fall back to plain text."""
    try:
        return JSONResponse(json.loads(payload))
    except json.JSONDecodeError:
        return PlainTextResponse(payload, media_type="application/json")


def _parse_int(value: Optional[str], field: str) -> int:
    if value is None:
        raise ValueError(f"{field} is required")
    return int(value)


@mcp.custom_route("/providers", methods=["GET"])
async def providers_route(_request: Request) -> Response:
    data = await get_clinic_provider_list()
    return _json_response_from_text(data)


@mcp.custom_route("/clinics/match", methods=["GET", "POST"])
async def clinics_match_route(request: Request) -> Response:
    """HTTP route for hybrid clinic matching."""
    if request.method == "POST":
        try:
            body = await request.json()
            query = body.get("query", "")
            top_k = _safe_int(body.get("top_k"), 5)
            min_score = _safe_float(body.get("min_score"), 0.65)
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    else:
        params = request.query_params
        query = params.get("query", "")
        top_k = _safe_int(params.get("top_k"), 5)
        min_score = _safe_float(params.get("min_score"), 0.65)

    if not query:
        return JSONResponse({"error": "query parameter is required"}, status_code=400)

    data = await match_clinic_hybrid(query=query, top_k=top_k, min_score=min_score)
    return _json_response_from_text(data)


@mcp.custom_route("/providers/schedule", methods=["GET"])
async def providers_schedule_route(request: Request) -> Response:
    params = request.query_params
    try:
        clinic_id = _parse_int(params.get("clinicid"), "clinicid")
    except (ValueError, TypeError):
        return JSONResponse({"error": "clinicid query parameter is required and must be an integer"}, status_code=400)

    day_id_param = params.get("dayid")
    provider_id_param = params.get("providerid")

    try:
        day_id = int(day_id_param) if day_id_param is not None else None
    except ValueError:
        return JSONResponse({"error": "dayid must be an integer"}, status_code=400)

    try:
        provider_id = int(provider_id_param) if provider_id_param is not None else None
    except ValueError:
        return JSONResponse({"error": "providerid must be an integer"}, status_code=400)

    data = await get_clinic_provider_schedule(
        clinicid=clinic_id,
        dayid=day_id,
        providerid=provider_id,
        day_name=params.get("day_name"),
    )
    return _json_response_from_text(data)


@mcp.custom_route("/providers/services/pricing", methods=["GET"])
async def providers_pricing_route(request: Request) -> Response:
    params = request.query_params
    try:
        clinic_id = _parse_int(params.get("clinicid"), "clinicid")
    except (ValueError, TypeError):
        return JSONResponse({"error": "clinicid query parameter is required and must be an integer"}, status_code=400)

    provider_id_param = params.get("providerid")
    try:
        provider_id = int(provider_id_param) if provider_id_param is not None else None
    except ValueError:
        return JSONResponse({"error": "providerid must be an integer"}, status_code=400)

    data = await get_service_price(
        clinicid=clinic_id,
        providerid=provider_id,
    )
    return _json_response_from_text(data)


def _safe_int(value: Any, default: int) -> int:
    """Safely convert a value to int, returning default on failure."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """Safely convert a value to float, returning default on failure."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@mcp.custom_route("/providers/match", methods=["GET", "POST"])
async def providers_match_route(request: Request) -> Response:
    """HTTP route for hybrid doctor matching."""
    # Support both GET (query params) and POST (JSON body)
    if request.method == "POST":
        try:
            body = await request.json()
            query = body.get("query", "")
            clinic_id = body.get("clinic_id")
            clinic_name = body.get("clinic_name")
            # Safely convert numeric parameters with defaults
            top_k = _safe_int(body.get("top_k"), 5)
            min_score_multi = _safe_float(body.get("min_score_multi"), 0.6)
            min_score_single = _safe_float(body.get("min_score_single"), 0.55)
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    else:
        params = request.query_params
        query = params.get("query", "")
        clinic_id = params.get("clinic_id") or params.get("clinicid")
        clinic_name = params.get("clinic_name") or params.get("clinicname")
        top_k = _safe_int(params.get("top_k"), 5)
        min_score_multi = _safe_float(params.get("min_score_multi"), 0.6)
        min_score_single = _safe_float(params.get("min_score_single"), 0.55)
    
    if not query:
        return JSONResponse({"error": "query parameter is required"}, status_code=400)
    
    data = await match_doctor_hybrid(
        query=query,
        clinic_id=clinic_id,
        clinic_name=clinic_name,
        top_k=top_k,
        min_score_multi=min_score_multi,
        min_score_single=min_score_single,
    )
    return _json_response_from_text(data)


# Run the server
if __name__ == "__main__":
    import uvicorn
    # Use the SSE app which includes HTTP routes
    uvicorn.run(
        mcp.sse_app(),
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
