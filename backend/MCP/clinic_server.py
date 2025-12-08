"""Static MCP Server for Clinic Management System."""
from __future__ import annotations

import json
import os
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

# Create MCP server instance that also powers the HTTP shim
mcp = FastMCP(
    "Clinic Management System",
    host=SERVER_HOST,
    port=SERVER_PORT,
)
settings = get_settings()


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
        norm_c = normalize_mixed_text(clinic_name)
        candidates = [
            d
            for d in candidates
            if norm_c in normalize_mixed_text(d.clinic_name)
        ]
    return candidates


def _match_single_token(
    token: str,
    candidates: List[DoctorRecord],
    top_k: int,
    min_score: float,
) -> MatchResponse:
    """Match a single-token query against doctor candidates."""
    scored: List[DoctorMatch] = []

    for d in candidates:
        pos_score, matched_first = compute_positional_token_weight(token, d.tokens)
        if pos_score < min_score:
            continue

        # Fuzzy full-name similarity (Arabic & English)
        q_str = token
        full_ar = d.norm_name_ar or d.name_ar
        full_en = d.norm_name_en or d.name_en

        fuzzy_ar = fuzz.ratio(q_str, full_ar) / 100.0 if full_ar else 0.0
        fuzzy_en = fuzz.ratio(q_str, full_en) / 100.0 if full_en else 0.0
        fuzzy_name_score = max(fuzzy_ar, fuzzy_en)

        # Overlap is binary in single-token (hit or miss)
        token_overlap = 1.0 if pos_score > 0 else 0.0

        # Final score: position is dominant here
        final_score = 0.6 * pos_score + 0.4 * fuzzy_name_score

        scored.append(
            DoctorMatch(
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
                matched_tokens=[token],
            )
        )

    if not scored:
        return MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم يتم العثور على دكتور بهذا الاسم.",
            query_tokens=[token],
            candidates=[],
        )

    # Sort by score, then prefer first-name matches as tie-breaker
    scored.sort(
        key=lambda m: (m.score, 1 if m.matched_by_first_name else 0),
        reverse=True,
    )

    top_matches = scored[:top_k]
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
    scored: List[DoctorMatch] = []

    for d in candidates:
        # 1) Token overlap using fuzzy token comparison
        overlap_count = 0
        matched_tokens: List[str] = []
        for qt in query_tokens:
            best_token_score = 0.0
            for dt in d.tokens:
                s = fuzz.ratio(qt, dt) / 100.0
                if s > best_token_score:
                    best_token_score = s
            if best_token_score >= 0.85:
                overlap_count += 1
                matched_tokens.append(qt)
        token_overlap = overlap_count / len(query_tokens)

        # 2) Positional score: average positional weight over tokens
        positional_scores: List[float] = []
        any_first = False
        for qt in query_tokens:
            pos_score, matched_first = compute_positional_token_weight(qt, d.tokens)
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

        fuzzy_ar = fuzz.ratio(q_string, full_ar) / 100.0 if full_ar else 0.0
        fuzzy_en = fuzz.ratio(q_string, full_en) / 100.0 if full_en else 0.0
        fuzzy_name_score = max(fuzzy_ar, fuzzy_en)

        # Final score (tunable weights)
        final_score = (
            0.35 * token_overlap
            + 0.25 * avg_pos_score
            + 0.2 * order_score
            + 0.2 * fuzzy_name_score
        )

        if final_score < min_score:
            continue

        scored.append(
            DoctorMatch(
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
        )

    if not scored:
        return MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم يتم العثور على دكتور بهذا الاسم.",
            query_tokens=query_tokens,
            candidates=[],
        )

    # Sort by score descending; tie-break: more matched tokens, then first-name usage
    scored.sort(
        key=lambda m: (m.score, len(m.matched_tokens), 1 if m.matched_by_first_name else 0),
        reverse=True,
    )

    top_matches = scored[:top_k]
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
    # Fetch provider list
    raw_data = await fetch_text(
        settings.provider_list_url,
        auth=settings.auth,
    )
    payload = json.loads(raw_data)
    
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
    
    # Tokenize query
    query_tokens = tokenize_name(query)
    
    # Filter by clinic if specified
    candidates = _filter_candidates(doctors, clinic_id, clinic_name)
    
    if not candidates:
        response = MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لا يوجد دكاترة في هذا التخصص/العيادة.",
            query_tokens=query_tokens,
        )
        return response.model_dump_json(by_alias=True)
    
    if not query_tokens:
        response = MatchResponse(
            status=MatchStatus.NO_MATCH,
            message="لم أستطع فهم اسم الدكتور. من فضلك اكتب اسم الدكتور.",
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
            top_k = body.get("top_k", 5)
            min_score_multi = body.get("min_score_multi", 0.6)
            min_score_single = body.get("min_score_single", 0.55)
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    else:
        params = request.query_params
        query = params.get("query", "")
        clinic_id = params.get("clinic_id") or params.get("clinicid")
        clinic_name = params.get("clinic_name") or params.get("clinicname")
        try:
            top_k = int(params.get("top_k", "5"))
        except ValueError:
            top_k = 5
        try:
            min_score_multi = float(params.get("min_score_multi", "0.6"))
        except ValueError:
            min_score_multi = 0.6
        try:
            min_score_single = float(params.get("min_score_single", "0.55"))
        except ValueError:
            min_score_single = 0.55
    
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
