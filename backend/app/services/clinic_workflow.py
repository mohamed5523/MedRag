"""
clinic_workflow.py — Intelligent MCP Orchestration Engine with تداخلات Support
================================================================================
Full support for تداخلات (additional medical procedures):
- Detects procedure names in any question (رسم قلب، كشف كمبيوتر، غيار، إلخ)
- Fans out across ALL clinics that may have the procedure in parallel
- Filters API results to show ONLY the asked procedure
- If procedure not in data yet → routes to correct clinic + helpful message
- Future-proof: new تداخلات added to doctor_prices.json are auto-picked up
  (once the MCP server returns additional_services in get_service_price)
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry import trace
from pydantic import BaseModel, Field

try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        from langchain.docstore.document import Document  # type: ignore

from app.core.intent_router import RouteDecision, RouteMode
from app.core.qa_engine import QAEngine
from app.core.state_manager import ConversationState
from app.core.radiology_knowledge import (
    get_radiology_context,
    get_emergency_radiology_context,
    is_radiology_question,
    is_emergency_radiology,
)
from app.integrations.mcp_client import (
    ClinicMatchResponse,
    DoctorMatchResult,
    DAY_NAME_TO_ID,
    HybridMatchResponse,
    HybridMatchStatus,
    MCPClient,
    ProviderListPayload,
    ProviderRecord,
    ProviderScheduleResponse,
    ScheduleSlot,
    ServicePriceResponse,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.clinic_workflow")


# =============================================================================
# Exceptions & Result Models
# =============================================================================

class MCPWorkflowError(Exception):
    def __init__(self, message: str, *, reason: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = reason
        self.data = data or {}


class ToolAuditEntry(BaseModel):
    name: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ClinicWorkflowResult(BaseModel):
    qa_response: Dict[str, Any]
    tool_audit: List[ToolAuditEntry] = Field(default_factory=list)


# =============================================================================
# TADAKHULAT KNOWLEDGE BASE
# =============================================================================
# Maps procedure keywords to candidate clinics.
# THIS IS THE ONLY PLACE TO EDIT when new تداخلات are added.
#
# Format: (canonical_procedure_name, [keyword_variants], [candidate_clinic_names])
#
# HOW IT WORKS:
#   "بكام رسم القلب"
#   -> _detect_procedure() finds "رسم قلب" -> candidate clinics = [قلب, باطنة, صدر]
#   -> fan-out get_service_price across all candidate clinics in parallel
#   -> _filter_services_by_procedure() keeps only matching rows
#   -> if found: show prices; if not found: _build_procedure_not_found_context()
# =============================================================================

_TADAKHULAT_KNOWLEDGE: List[Tuple[str, List[str], List[str]]] = [
    # -- EXISTS in doctor_prices.json ------------------------------------------
    (
        "كشف كمبيوتر",
        ["كشف كمبيوتر", "كشف بالكمبيوتر", "كمبيوتر"],
        ["عيادة رمد", "عيادة اشعه"],
    ),
    (
        "رسم قلب",
        ["رسم قلب", "تخطيط قلب", "ecg", "ekg", "تخطيط القلب",
         "كهربا قلب", "كهرباء قلب", "رسم الكتروكارديوجرام"],
        ["عيادة قلب", "عيادة باطنة", "عيادة صدر",
         "باطن وروماتيزم وامراض مناعه"],
    ),
    (
        "غيار صغير",
        ["غيار صغير", "غيار الجرح", "تغيير الضمادة", "تضميد", "غيار"],
        ["عيادة جراحة", "جراحه مخ وعمود فقرى"],
    ),
    (
        "استكمال رسم مخ",
        ["رسم مخ", "رسم الدماغ", "تخطيط المخ", "eeg",
         "استكمال رسم مخ", "كهرباء مخ", "كهربا مخ"],
        ["عيادة أعصاب", "امراض مخ واعصاب"],
    ),
    (
        "تركيبات زيركون",
        ["زيركون", "تركيبات زيركون", "سن زيركون", "تاج زيركون",
         "تركيب سنه", "تركيبة سنه", "تركيبات"],
        ["عيادة أسنان"],
    ),
    # -- NOT YET in data — still route correctly --------------------------------
    (
        "ايكو",
        ["ايكو", "echo", "إيكو", "أشعة تلفزيونية على القلب",
         "ايكو على القلب", "سونار القلب"],
        ["عيادة قلب"],
    ),
    (
        "هولتر",
        ["هولتر", "holter", "مراقبة القلب", "holter monitor"],
        ["عيادة قلب"],
    ),
    (
        "قياس ضغط العين",
        ["ضغط عين", "ضغط العيون", "قياس ضغط العين", "tonometry"],
        ["عيادة رمد"],
    ),
    (
        "فحص قاع العين",
        ["قاع عين", "قاع العين", "فحص قاع العين", "fundus"],
        ["عيادة رمد"],
    ),
    (
        "كشف نظارة",
        ["كشف نظارة", "نظارة", "قياس النظر", "عدسات", "نضارة"],
        ["عيادة رمد"],
    ),
    (
        "منظار معدة",
        ["منظار معدة", "منظار المعدة", "endoscopy", "منظار الجهاز الهضمي"],
        ["عيادة باطنة"],
    ),
    (
        "منظار قولون",
        ["منظار قولون", "منظار القولون", "colonoscopy"],
        ["عيادة باطنة"],
    ),
    (
        "حقنة مفصل",
        ["حقنة مفصل", "حقن مفصل", "cortisone", "كورتيزون",
         "حقنة ركبة", "حقنة كتف", "hyaluronic", "هيالورونيك"],
        ["عيادة عظام"],
    ),
    (
        "جلسة علاج طبيعي",
        ["جلسة علاج طبيعي", "جلسة كهربا", "جلسات كهربا", "موجات صوتية",
         "ultrasound therapy", "جلسة ليزر علاجي",
         "تمارين علاجية", "جلسة فيزيوثيرابي"],
        ["عيادة علاج طبيعي"],
    ),
    (
        "بوتوكس",
        ["بوتوكس", "botox", "حقن بوتوكس"],
        ["عيادة جراحه تجميل"],
    ),
    (
        "فيلر",
        ["فيلر", "filler", "حقن فيلر"],
        ["عيادة جراحه تجميل"],
    ),
    (
        "اختبار سمع",
        ["اختبار سمع", "قياس السمع", "audiometry", "فحص الأذن"],
        ["عيادة أنف وأذن وحنجرة"],
    ),
    (
        "خلع ضرس",
        ["خلع ضرس", "خلع سن", "قلع ضرس", "قلع سن", "extraction"],
        ["عيادة أسنان"],
    ),
    (
        "حشو أسنان",
        ["حشو سن", "حشو ضرس", "حشو أسنان", "filling"],
        ["عيادة أسنان"],
    ),
    (
        "تنظيف أسنان",
        ["تنظيف أسنان", "تنظيف الأسنان", "scaling", "جير أسنان"],
        ["عيادة أسنان"],
    ),
]

# Build flat lookup: keyword -> (canonical_name, clinics)
_PROCEDURE_KEYWORD_MAP: Dict[str, Tuple[str, List[str]]] = {}
for _canon, _keywords, _clinics in _TADAKHULAT_KNOWLEDGE:
    for _kw in _keywords:
        _PROCEDURE_KEYWORD_MAP[_kw.casefold()] = (_canon, _clinics)


def _detect_procedure(question: str) -> Optional[Tuple[str, List[str]]]:
    """
    Detect if the question mentions a specific تداخل.
    Returns (canonical_name, [candidate_clinics]) or None.
    Longer keywords matched first to avoid partial matches.
    """
    q = (question or "").casefold()
    for kw in sorted(_PROCEDURE_KEYWORD_MAP.keys(), key=len, reverse=True):
        if kw in q:
            return _PROCEDURE_KEYWORD_MAP[kw]
    return None


def _filter_services_by_procedure(services: list, procedure_name: str) -> list:
    """
    Keep only service rows whose name matches the requested procedure.
    Falls back to all services if nothing matches.
    """
    if not procedure_name:
        return services
    needle = procedure_name.casefold().replace(" ", "")
    matched = [
        s for s in services
        if needle in (s.service_name_ar or "").casefold().replace(" ", "")
        or needle in (s.service_name_en or "").casefold().replace(" ", "")
    ]
    return matched if matched else services


def _build_procedure_not_found_context(procedure_name: str, clinics: List[str]) -> str:
    """
    Helpful context when a تداخل is not in the API data yet.
    Tells the QA engine to redirect the user warmly.
    """
    clinic_str = " أو ".join(clinics) if clinics else "العيادة المختصة"
    return (
        f"[معلومات المستشفى — تداخل غير موجود في البيانات]\n"
        f"الإجراء المطلوب: {procedure_name}\n"
        f"هذا الإجراء غير مسجل حالياً في أسعار المستشفى.\n"
        f"العيادة المناسبة: {clinic_str}\n"
        f"[تعليمات للمساعد: أخبر المريض بأن هذا الإجراء غير متوفر سعره في المستشفى الآن، "
        f"وانصحه بالتوجه لـ {clinic_str} أو الاتصال بالاستقبال للاستفسار عن السعر والمواعيد.]"
    )


def _format_procedure_fanout_context(
    procedure_name: str,
    clinic_results: List[Tuple[str, List]],
) -> str:
    """Format cross-clinic results for a specific تداخل."""
    lines = [f"نتائج البحث عن '{procedure_name}':", ""]
    found_any = False
    for clinic_name, services in clinic_results:
        if not services:
            continue
        found_any = True
        lines.append(f"العيادة: {clinic_name}")
        for s in services:
            doc = f" — الدكتور {s.doctor_name}" if getattr(s, "doctor_name", None) else ""
            price_str = (
                f"{s.price:.2f} {s.currency or 'جنيه'}"
                if s.price is not None else "السعر غير محدد"
            )
            sname = s.service_name_ar or s.service_name_en or procedure_name
            lines.append(f"  - {sname}{doc}: {price_str}")
        lines.append("")
    if not found_any:
        lines.append(f"لم يتم العثور على '{procedure_name}' في بيانات الأسعار الحالية.")
    return "\n".join(lines)


# =============================================================================
# Clinic synonym / normalization maps
# =============================================================================

CLINIC_SYNONYMS: Dict[str, str] = {
    "عيون": "رمد", "العيون": "الرمد", "نظر": "رمد",
    "باطني": "باطنة", "باطنيه": "باطنة", "باطنية": "باطنة", "باطن": "باطنة",
    "نسائية": "نسا وتوليد", "نساء": "نسا وتوليد", "نسا": "نسا وتوليد",
    "ولادة": "نسا وتوليد", "نسا وولادة": "نسا وتوليد",
    "اسنان": "أسنان", "سنان": "أسنان",
    "أعصاب": "امراض مخ واعصاب", "اعصاب": "امراض مخ واعصاب",
    "أطفال": "اطفال",
    "رنين": "اشعه", "مرنانة": "اشعه", "مرنانه": "اشعه", "mri": "اشعه",
    "ماموجرام": "اشعه", "mammogram": "اشعه",
    "ديكسا": "اشعه", "dexa": "اشعه",
    "بت سكان": "اشعه مقطعيه", "pet scan": "اشعه مقطعيه",
    "إكس راي": "اشعه عاديه", "x-ray": "اشعه عاديه", "xray": "اشعه عاديه",
    "دوبلر": "اشعه تلفزيونيه", "doppler": "اشعه تلفزيونيه",
    "سونار": "اشعه تلفزيونيه", "تلفزيوني": "اشعه تلفزيونيه",
    "سكانر": "اشعه مقطعيه", "مقطعي": "اشعه مقطعيه", "مقطعية": "اشعه مقطعيه",
    "بانوراما": "أسنان", "باناراما": "أسنان", "سيفالوميتريك": "أسنان",
    "تجميل": "جراحه تجميل", "بوتوكس": "جراحه تجميل", "فيلر": "جراحه تجميل",
    "نفسية": "نفسيه وعصبيه", "نفسيه": "نفسيه وعصبيه", "طب نفسي": "نفسيه وعصبيه",
    "علاج طبيعي": "علاج طبيعى", "تأهيل": "علاج طبيعى",
    "أورام": "اورام", "سرطان": "اورام",
    "مسالك": "مسالك بوليه", "مسالك بولية": "مسالك بوليه",
    "صدر": "صدريه",
    "جراحة": "جراحه",
}

_HARDCODED_CLINIC_IDS: Dict[str, int] = {
    "عظام": 1119,
    "أسنان": 1108, "اسنان": 1108,
    "رمد": 1116, "عيون": 1116,
    "أطفال": 1109, "اطفال": 1109,
    "باطنة": 1087, "باطنه": 1087,
    "جراحة": 1097, "جراحه": 1097,
    "أنف وأذن وحنجرة": 1110,
    "جلدية": 1115, "جلديه": 1115,
    "نسا وتوليد": 1122, "نسا": 1122,
    "قلب": 1090,
    "صدر": 1118, "صدريه": 1118,
    "أعصاب": 1106, "اعصاب": 1106,
    "امراض مخ واعصاب": 1106,
    "مسالك بولية": 1121, "مسالك بوليه": 1121,
    "سكر وغدد صماء": 1136,
    "نفسية": 1105, "نفسيه": 1105, "نفسيه وعصبيه": 1105,
    "علاج طبيعي": 1120, "علاج طبيعى": 1120,
    "أورام": 1111, "اورام": 1111,
    "جراحه تجميل": 1098,
    "كبد": 1092,
    "اشعه": 1102,
    "اشعه مقطعيه": 1104,
    "اشعه تلفزيونيه": 1102,
    "اشعه عاديه": 1,
    # Exact clinic names from doctor_prices.json
    "Radiology": 1,
    "باطن وروماتيزم وامراض مناعه": 1085,
    "جراحه مخ وعمود فقرى": 1106,
}


def _normalize_clinic_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    key = name.strip()
    for prefix in ("عيادة ", "عياده ", "دكتور "):
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
            break
    for candidate in (key, key.replace("ة", "ه"), key.replace("ه", "ة")):
        if candidate in CLINIC_SYNONYMS:
            return CLINIC_SYNONYMS[candidate]
    return name.strip()


def _hardcoded_clinic_id(name: str) -> Optional[int]:
    key = (name or "").strip()
    for prefix in ("عيادة ", "عياده "):
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
            break
    for candidate in (key, key.replace("ة", "ه"), key.replace("ه", "ة")):
        if candidate in _HARDCODED_CLINIC_IDS:
            return _HARDCODED_CLINIC_IDS[candidate]
    return None


# =============================================================================
# Date helpers
# =============================================================================

_AR_WEEKDAY_TO_EN = {
    "السبت": "saturday", "سبت": "saturday",
    "الأحد": "sunday", "الاحد": "sunday", "احد": "sunday",
    "الاثنين": "monday", "الاتنين": "monday", "اثنين": "monday",
    "الثلاثاء": "tuesday", "التلات": "tuesday", "ثلاثاء": "tuesday",
    "الأربعاء": "wednesday", "الاربعاء": "wednesday", "الأربع": "wednesday",
    "الخميس": "thursday", "خميس": "thursday",
    "الجمعة": "friday", "جمعه": "friday", "جمعة": "friday",
}

_PY_WEEKDAY = {
    "saturday": 5, "sunday": 6, "monday": 0,
    "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
}


def _cairo_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=2)))


def _infer_date_range(
    question: str,
    *,
    now_dt: Optional[datetime] = None,
    date_hint: Optional[str] = None,
) -> Tuple[str, str, bool, bool]:
    base = now_dt or _cairo_now()
    q = (question or "").casefold()

    def _fmt(dt: datetime) -> str:
        return dt.strftime("%d/%m/%Y")

    if date_hint:
        hint = date_hint.lower().strip()
        if hint in ("today", "النهارده", "اليوم"):
            d = _fmt(base); return d, d, True, True
        if hint in ("tomorrow", "بكرة", "بكره", "غدا"):
            t = base + timedelta(days=1); d = _fmt(t); return d, d, True, False
        for ar, en in _AR_WEEKDAY_TO_EN.items():
            if hint in (en, ar):
                days_ahead = (_PY_WEEKDAY[en] - base.weekday()) % 7
                target = base + timedelta(days=days_ahead)
                d = _fmt(target); return d, d, True, False

    if any(tok in q for tok in ["النهارده", "نهارده", "اليوم", "دلوقتي", "الآن", "حاليا"]):
        d = _fmt(base); return d, d, True, True
    if any(tok in q for tok in ["بكره", "بكرة", "غدا", "غداً"]):
        t = base + timedelta(days=1); d = _fmt(t); return d, d, True, False
    for ar, en in _AR_WEEKDAY_TO_EN.items():
        if ar in q:
            days_ahead = (_PY_WEEKDAY[en] - base.weekday()) % 7
            target = base + timedelta(days=days_ahead)
            d = _fmt(target); return d, d, True, False

    date_from = _fmt(base)
    date_to = _fmt(base)
    return date_from, date_to, False, True


def _infer_target_day_id(
    question: str,
    *,
    now_dt: Optional[datetime] = None,
    date_hint: Optional[str] = None,
) -> int:
    base = now_dt or _cairo_now()
    q = (question or "").casefold()
    if date_hint:
        hint = date_hint.lower()
        if hint in ("today", "النهارده", "اليوم"):
            return DAY_NAME_TO_ID[base.strftime("%A").lower()]
        if hint in ("tomorrow", "بكرة", "بكره"):
            return DAY_NAME_TO_ID[(base + timedelta(days=1)).strftime("%A").lower()]
        for ar, en in _AR_WEEKDAY_TO_EN.items():
            if hint in (en, ar):
                return DAY_NAME_TO_ID[en]
    if any(tok in q for tok in ["بكره", "بكرة", "غدا"]):
        return DAY_NAME_TO_ID[(base + timedelta(days=1)).strftime("%A").lower()]
    if any(tok in q for tok in ["النهارده", "نهارده", "اليوم", "دلوقتي"]):
        return DAY_NAME_TO_ID[base.strftime("%A").lower()]
    for ar, en in _AR_WEEKDAY_TO_EN.items():
        if ar in q:
            return DAY_NAME_TO_ID[en]
    return DAY_NAME_TO_ID[base.strftime("%A").lower()]


def _time_str_to_minutes(t: str) -> int:
    if not t: return 9999
    is_pm = any(m in t for m in ["مساء", "مساءً", "PM", "pm", "م"])
    is_am = any(m in t for m in ["صباح", "صباحًا", "AM", "am", "ص"])
    m = re.search(r"(\d{1,2})[:.](\d{2})", t)
    if not m: return 9999
    h, mn = int(m.group(1)), int(m.group(2))
    if h >= 13: return h * 60 + mn
    if is_pm and h != 12: h += 12
    elif is_am and h == 12: h = 0
    return h * 60 + mn


def _safe_parse_int(value: Any) -> Optional[int]:
    if value is None: return None
    if isinstance(value, int): return value
    if isinstance(value, str):
        if value.strip().lower() in ("", "none", "null"): return None
        try: return int(value)
        except ValueError: return None
    try: return int(value)
    except (ValueError, TypeError): return None


# =============================================================================
# Formatting helpers
# =============================================================================

def _normalize_ampm(value: str) -> str:
    if not value: return value
    out = value
    out = re.sub(r"(?i)\bA\.?M\.?\b", "صباحًا", out)
    out = re.sub(r"(?i)\bP\.?M\.?\b", "مساءً", out)
    out = re.sub(r"(\d{1,2}[:.]\d{2})\s*ص\b", r"\1 صباحًا", out)
    out = re.sub(r"(\d{1,2}[:.]\d{2})\s*م\b", r"\1 مساءً", out)
    out = re.sub(r"صباح(?!\u064b)(?:ا|اً|ً|ًا|)\b", "صباحًا", out)
    out = re.sub(r"مساء(?!\u064c)(?:ا|اً|ً|ًا|)\b", "مساءً", out)
    return out


def _format_price_context(
    price_response: ServicePriceResponse,
    provider_entry: Optional[ProviderRecord],
    question: Optional[str] = None,
    *,
    procedure_name: Optional[str] = None,
) -> str:
    clinic = (provider_entry.clinic_name_ar or provider_entry.clinic_name_en or "") if provider_entry else ""
    doctor = (provider_entry.provider_name_ar or provider_entry.provider_name_en or "") if provider_entry else ""
    lines = []
    if procedure_name:
        lines.append(f"نتيجة البحث عن '{procedure_name}' في {clinic or 'المستشفى'}:")
    elif clinic:
        lines.append(f"أسعار {clinic}:")
    elif doctor:
        lines.append(f"أسعار الدكتور {doctor}:")
    else:
        lines.append("الأسعار المتاحة:")
    valid = [s for s in price_response.services if s.price is not None]
    if procedure_name:
        valid = _filter_services_by_procedure(valid, procedure_name)
    elif doctor:
        target = doctor.casefold().replace(" ", "")
        doc_specific = [
            s for s in valid
            if s.doctor_name and target in str(s.doctor_name).casefold().replace(" ", "")
        ]
        if doc_specific:
            valid = doc_specific
    if not valid:
        lines.append("- السعر غير متوفر حالياً في المستشفى.")
    else:
        # Prevent token overflow for large clinics (e.g. dental)
        # Separate kashf and istishara to ensure correct ordering (Kashf first)
        KASHF_KEYWORDS = ("كشف", "consult")
        ISTISHARA_KEYWORDS = ("استشارة", "استشاره", "متابعة", "متابعه")
        
        kashf = [s for s in valid if any(kw in (s.service_name_ar or "").casefold() for kw in KASHF_KEYWORDS)]
        istishara = [s for s in valid if any(kw in (s.service_name_ar or "").casefold() for kw in ISTISHARA_KEYWORDS) and s not in kashf]
        other = [s for s in valid if s not in kashf and s not in istishara]
        
        capped = (kashf[:3] + istishara[:2] + other[:2])[:7]  # absolute max of 7
        for s in capped:
            name = s.service_name_ar or s.service_name_en or "كشف"
            price = f"{s.price:.2f}"
            currency = s.currency or "جنيه"
            doc_label = f" — {s.doctor_name}" if getattr(s, "doctor_name", None) else ""
            lines.append(f"- {name}{doc_label}: {price} {currency}")
        if len(valid) > len(capped):
            lines.append(f"- (...يوجد أسعار لخدمات أخرى — اسأل الاستقبال)")
    return "\n".join(lines)


def _format_schedule_context(
    response: ProviderScheduleResponse,
    provider_entry: Optional[ProviderRecord],
) -> str:
    doctor = (provider_entry.provider_name_ar or provider_entry.provider_name_en or "") if provider_entry else ""
    clinic = (provider_entry.clinic_name_ar or provider_entry.clinic_name_en or "") if provider_entry else ""
    lines = []
    if clinic and doctor:
        lines.append(f"مواعيد الدكتور {doctor} في {clinic}:")
    elif clinic:
        lines.append(f"مواعيد {clinic}:")
    elif doctor:
        lines.append(f"مواعيد الدكتور {doctor}:")
    else:
        lines.append("المواعيد المتاحة:")
    grouped: Dict[str, List[str]] = defaultdict(list)
    for slot in response.slots:
        day = slot.day_name or f"اليوم #{slot.day_id}"
        start = _normalize_ampm(slot.shift_start or "غير محدد")
        end = _normalize_ampm(slot.shift_end or "غير محدد")
        entry = f"{start} -> {end}"
        status = getattr(slot, "slot_status", None)
        if status:
            entry += f" ({status})"
        elif getattr(slot, "is_excused", False):
            entry += " (معتذر)"
        grouped[day].append(entry)
    for day, entries in grouped.items():
        lines.append(f"\n{day}:")
        for e in entries:
            lines.append(f"  - {e}")
    return "\n".join(lines)


def _format_multi_doctor_schedule(
    clinic_name: str,
    schedules: List[Tuple[ProviderRecord, ProviderScheduleResponse]],
) -> str:
    processed = []
    for provider, response in schedules:
        grouped: Dict[str, List[Tuple[str, str, bool, Optional[str]]]] = defaultdict(list)
        earliest = 9999
        for slot in response.slots:
            day = slot.day_name or f"اليوم #{slot.day_id}"
            start = _normalize_ampm(slot.shift_start or "")
            end = _normalize_ampm(slot.shift_end or "")
            if start and end:
                is_exc = getattr(slot, "is_excused", False)
                status = getattr(slot, "slot_status", None)
                grouped[day].append((start, end, is_exc, status))
                t = _time_str_to_minutes(slot.shift_start or start)
                if t < earliest:
                    earliest = t
        processed.append((provider, grouped, earliest))
    processed.sort(key=lambda x: x[2])
    lines = [f"مواعيد دكاترة {clinic_name} (مرتبة حسب الوقت):", ""]
    for provider, grouped, _ in processed:
        name = provider.provider_name_ar or provider.provider_name_en or "دكتور"
        lines.append(f"دكتور {name}")
        all_slots = [(s, e, ex, st) for day_slots in grouped.values() for s, e, ex, st in day_slots]
        if len(all_slots) == 1:
            s, e, ex, st = all_slots[0]
            label = f" ({st})" if st else (" (معتذر)" if ex else "")
            lines.append(f"  - من {s} لحد {e}{label}")
        else:
            labels = ["الفترة الأولى", "الفترة التانية", "الفترة الثالثة"]
            for i, (s, e, ex, st) in enumerate(all_slots):
                lbl = labels[i] if i < len(labels) else f"الفترة {i+1}"
                status_str = f" ({st})" if st else (" (معتذر)" if ex else "")
                lines.append(f"  - {lbl}: من {s} لحد {e}{status_str}")
        lines.append("")
    return "\n".join(lines)


def _format_combined_price_schedule(
    clinic_name: str,
    doctors: List[Tuple[ProviderRecord, List[str], Optional[ServicePriceResponse]]],
    question: Optional[str] = None,
    *,
    procedure_name: Optional[str] = None,
) -> str:
    lines = [
        f"بيانات دكاترة {clinic_name} — المواعيد والأسعار:", "",
        "[تعليمات: اعرض كل دكتور في فقرة منفصلة مع مواعيده وسعر الكشف]", "",
    ]
    for provider, slot_strs, price_resp in doctors:
        name = provider.provider_name_ar or provider.provider_name_en or "دكتور"
        lines.append(f"دكتور {name}")
        if slot_strs:
            if len(slot_strs) == 1:
                lines.append(f"  - المواعيد: من {slot_strs[0].replace(' -> ', ' لحد ')}")
            else:
                labels = ["الفترة الأولى", "الفترة التانية", "الفترة الثالثة"]
                for i, slot in enumerate(slot_strs):
                    lbl = labels[i] if i < len(labels) else f"الفترة {i+1}"
                    lines.append(f"  - {lbl}: من {slot.replace(' -> ', ' لحد ')}")
        else:
            lines.append("  - المواعيد: غير متاح في هذا اليوم")
        if price_resp:
            valid = [s for s in price_resp.services if s.price is not None]
            target_ar = (provider.provider_name_ar or "").casefold().replace(" ", "")
            target_en = (provider.provider_name_en or "").casefold().replace(" ", "")
            if target_ar or target_en:
                doc_specific = [
                    s for s in valid
                    if s.doctor_name and (
                        (target_ar and target_ar in str(s.doctor_name).casefold().replace(" ", "")) or
                        (target_en and target_en in str(s.doctor_name).casefold().replace(" ", ""))
                    )
                ]
                if doc_specific:
                    valid = doc_specific
            if procedure_name:
                filtered = _filter_services_by_procedure(valid, procedure_name)
                if filtered:
                    valid = filtered
            if valid:
                # Cap specific services per doctor to avoid 100k+ tokens
                # Separate kashf and istishara to ensure correct ordering (Kashf first)
                KASHF_KEYWORDS = ("كشف", "consult")
                ISTISHARA_KEYWORDS = ("استشارة", "استشاره", "متابعة", "متابعه")
                
                kashf = [s for s in valid if any(kw in (s.service_name_ar or "").casefold() for kw in KASHF_KEYWORDS)]
                istishara = [s for s in valid if any(kw in (s.service_name_ar or "").casefold() for kw in ISTISHARA_KEYWORDS) and s not in kashf]
                other = [s for s in valid if s not in kashf and s not in istishara]
                
                capped = (kashf[:2] + istishara[:2] + other[:2])[:5] # Max 5 services per doctor
                for s in capped:
                    sname = s.service_name_ar or s.service_name_en or "كشف"
                    lines.append(f"  - {sname}: {s.price:.2f} {s.currency or 'جنيه'}")
                if len(valid) > len(capped):
                    lines.append(f"  - (...وخدمات أخرى اسأل عنها الاستقبال)")
            else:
                lines.append("  - السعر: غير متوفر في المستشفى")
        else:
            lines.append("  - السعر: غير متوفر في المستشفى")
        lines.append("")
    return "\n".join(lines)


def _format_who_is_present(
    clinic_label: str,
    day_id: int,
    present: List[Tuple[str, List[str]]],
) -> str:
    day_names = {1: "السبت", 2: "الأحد", 3: "الاثنين", 4: "الثلاثاء",
                 5: "الأربعاء", 6: "الخميس", 7: "الجمعة"}
    day = day_names.get(day_id, "اليوم")
    lines = [f"الدكاترة الموجودين في {clinic_label} — {day}:", ""]
    for name, times in present:
        lines.append(f"دكتور {name}")
        for t in times:
            lines.append(f"  - {t}")
        lines.append("")
    return "\n".join(lines)


def _format_provider_list(provider_list: ProviderListPayload) -> str:
    lines = ["قائمة الدكاترة المتاحة:"]
    for r in provider_list.providers:
        doctor = r.provider_name_ar or r.provider_name_en or "دكتور"
        clinic = r.clinic_name_ar or r.clinic_name_en or "عيادة"
        specialty = r.specialty or ""
        suffix = f" — تخصص: {specialty}" if specialty else ""
        lines.append(f"- {doctor} في {clinic}{suffix}")
    return "\n".join(lines)


def _filter_provider_list_by_clinic_id(
    provider_list: ProviderListPayload, clinic_id: int
) -> ProviderListPayload:
    filtered = [p for p in provider_list.providers if p.clinic_id == clinic_id]
    return ProviderListPayload(providers=filtered)


def _slot_strs_from_slots(slots: list) -> List[str]:
    result = []
    for slot in slots:
        start = _normalize_ampm(slot.shift_start or "غير محدد")
        end = _normalize_ampm(slot.shift_end or "غير محدد")
        status = getattr(slot, "slot_status", None)
        label = f" ({status})" if status else (" (معتذر)" if getattr(slot, "is_excused", False) else "")
        result.append(f"{start} -> {end}{label}")
    return result


# =============================================================================
# Main Workflow Service
# =============================================================================

class ClinicWorkflowService:
    """
    Coordinates MCP tool calls for any Egyptian Arabic clinic question,
    including تداخلات (additional procedures).
    """

    def __init__(self, mcp_client: Optional[MCPClient] = None):
        self._client = mcp_client or MCPClient()

    async def aclose(self):
        await self._client.aclose()

    async def run(
        self,
        *,
        decision: RouteDecision,
        state: ConversationState,
        question: str,
        qa_engine: QAEngine,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_gender: str = "male",
    ) -> ClinicWorkflowResult:
        if decision.mode != RouteMode.MCP:
            raise ValueError("ClinicWorkflowService only handles MCP decisions.")

        with tracer.start_as_current_span("clinic_workflow.run") as span:
            span.set_attribute("workflow.intent", decision.intent)
            span.set_attribute("workflow.question", question[:200])

            tool_audit: List[ToolAuditEntry] = []
            docs: List[Document] = []

            try:
                date_hint = decision.enriched_date_hint
                now_dt = _cairo_now()
                time_context = qa_engine.build_time_context(question)
                self._apply_enrichment(state, decision)

                # [FIX] Wrap entire dispatch in a 20-second timeout.
                # Slow/dead MCP connections at off-peak hours previously caused
                # the coroutine to hang until uvicorn killed it with a 500.
                async def _dispatch() -> List[Document]:
                    # STEP 1: Detect تداخل FIRST — highest priority
                    procedure_detected = _detect_procedure(question)
                    if procedure_detected:
                        proc_name, proc_clinics = procedure_detected
                        span.set_attribute("workflow.procedure", proc_name)
                        logger.info("Procedure detected: '%s' -> clinics: %s", proc_name, proc_clinics)
                        return await self._handle_procedure_query(
                            state, tool_audit, question,
                            procedure_name=proc_name,
                            candidate_clinics=proc_clinics,
                        )

                    # STEP 2: Normal intent dispatch
                    intent = decision.intent
                    all_intents = decision.all_intents or [intent]

                    if intent == "ask_price_and_availability" or (
                        "ask_price" in all_intents and (
                            "check_availability" in all_intents or
                            "who_is_present" in all_intents
                        )
                    ):
                        return await self._handle_price_and_availability(
                            state, tool_audit, question,
                            date_hint=date_hint, now_dt=now_dt,
                        )
                    elif intent == "who_is_present":
                        result_docs, _ = await self._handle_who_is_present(
                            state, tool_audit, question,
                            date_hint=date_hint, now_dt=now_dt,
                        )
                        if "ask_price" in all_intents:
                            try:
                                price_docs, _ = await self._handle_pricing(state, tool_audit, question)
                                result_docs.extend(price_docs)
                            except MCPWorkflowError:
                                pass
                        return result_docs
                    elif intent == "ask_price":
                        result_docs, _ = await self._handle_pricing(state, tool_audit, question)
                        return result_docs
                    elif intent in ("check_availability", "book_appointment"):
                        result_docs, _ = await self._handle_schedule(
                            state, tool_audit, question,
                            date_hint=date_hint, now_dt=now_dt,
                        )
                        return result_docs
                    elif intent == "list_doctors":
                        result_docs, _ = await self._handle_list_doctors(state, tool_audit)
                        return result_docs
                    else:
                        logger.warning("Unknown intent '%s' -> defaulting to check_availability", intent)
                        result_docs, _ = await self._handle_schedule(
                            state, tool_audit, question,
                            date_hint=date_hint, now_dt=now_dt,
                        )
                        return result_docs

                try:
                    docs = await asyncio.wait_for(_dispatch(), timeout=60.0)
                except asyncio.TimeoutError:
                    logger.error("MCP workflow timed out after 60s: %s", question[:100])
                    raise MCPWorkflowError(
                        "معلش يا فندم، النظام بطيء شوية دلوقتي. ممكن حضرتك تحاول تاني بعد ثواني، أو لو مستعجل تتواصل مع الاستقبال وهما هيساعدوك فوراً.",
                        reason="timeout",
                    )

                span.set_attribute("workflow.docs_count", len(docs))

            except MCPWorkflowError:
                raise
            except Exception as exc:
                logger.error("Unexpected workflow error: %s", exc, exc_info=True)
                raise MCPWorkflowError(
                    "بعتذر لحضرتك جداً، حصل خطأ بسيط وإحنا بنجيب البيانات. ممكن حضرتك تحاول تسألني تاني أو تتواصل مع الاستقبال وهما هيفيدوك؟",
                    reason="unexpected_error",
                )

            qa_payload = await qa_engine.answer_question(
                question=question,
                contexts=docs,
                time_context=time_context,
                chat_history=chat_history,
                user_gender=user_gender,
            )
            return ClinicWorkflowResult(qa_response=qa_payload, tool_audit=tool_audit)

    # =========================================================================
    # TADAKHULAT HANDLER
    # =========================================================================

    async def _handle_procedure_query(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        *,
        procedure_name: str,
        candidate_clinics: List[str],
    ) -> List[Document]:
        """
        Handle questions about a specific تداخل / additional service.
        Fans out across all candidate clinics in parallel, filters results,
        and returns either found prices or a helpful redirect.
        """
        resolved: Dict[int, str] = {}
        for clinic_name in candidate_clinics:
            normalized = _normalize_clinic_name(clinic_name)
            cid = _hardcoded_clinic_id(normalized or clinic_name)
            if cid is not None and cid not in resolved:
                resolved[cid] = clinic_name
                continue
            try:
                match = await self._client.match_clinic_hybrid(
                    query=normalized or clinic_name, top_k=1, min_score=0.3
                )
                if match.best_match and match.best_match.score >= 0.35:
                    cid = _safe_parse_int(match.best_match.clinic_id)
                    if cid and cid not in resolved:
                        resolved[cid] = match.best_match.clinic_name or clinic_name
            except Exception as exc:
                logger.warning("Clinic match failed for '%s': %s", clinic_name, exc)

        if not resolved:
            context = _build_procedure_not_found_context(procedure_name, candidate_clinics)
            audit.append(ToolAuditEntry(
                name="procedure_search", status="no_clinic_resolved",
                details={"procedure": procedure_name, "candidates": candidate_clinics},
            ))
            return [Document(page_content=context, metadata={"source": "mcp.procedure_not_found"})]

        async def _fetch(cid: int, cname: str):
            try:
                resp = await self._client.get_service_price(clinic_id=cid)
                return cid, cname, resp
            except Exception as exc:
                logger.warning("Price fetch failed clinic %s: %s", cname, exc)
                return cid, cname, None

        results = await asyncio.gather(*(_fetch(cid, cn) for cid, cn in resolved.items()))

        audit.append(ToolAuditEntry(
            name="get_service_price_fanout", status="success",
            details={
                "procedure": procedure_name,
                "clinics_queried": len(resolved),
                "clinics_responded": sum(1 for _, _, r in results if r),
            },
        ))

        clinic_results: List[Tuple[str, List]] = []
        for cid, cname, resp in results:
            if resp is None:
                continue
            matching = _filter_services_by_procedure(
                [s for s in resp.services if s.price is not None],
                procedure_name,
            )
            clinic_results.append((cname, matching))

        found = [(cn, svcs) for cn, svcs in clinic_results if svcs]

        if found:
            context = _format_procedure_fanout_context(procedure_name, found)
            return [Document(
                page_content=context,
                metadata={"source": "mcp.procedure_price", "procedure": procedure_name},
            )]
        else:
            context = _build_procedure_not_found_context(procedure_name, candidate_clinics)
            return [Document(
                page_content=context,
                metadata={"source": "mcp.procedure_not_found", "procedure": procedure_name},
            )]

    # =========================================================================
    # Entity enrichment
    # =========================================================================

    def _apply_enrichment(self, state: ConversationState, decision: RouteDecision) -> None:
        """
        [STABILITY] Always overwrite state entities with what the LLM router extracted.
        The router is the authoritative extraction for THIS turn; stale state from
        previous turns must not shadow a fresh router decision.
        Also keeps provider_id in sync: when doctor changes, clear the old resolved ID.
        """
        try:
            if decision.enriched_clinic:
                state.entities.clinic = decision.enriched_clinic
                state.target_entity_type = "clinic"
                # Clinic changed → old clinic_id is invalid
                state.entities.clinic_id = None

            if decision.enriched_doctor:
                if decision.enriched_doctor != state.entities.doctor:
                    # Doctor identity changed → old provider_id is invalid
                    state.entities.provider_id = None
                state.entities.doctor = decision.enriched_doctor
                state.target_entity_type = "doctor"
            elif decision.enriched_clinic and not decision.enriched_doctor:
                # Router found a clinic but no doctor → this is a clinic-scope query.
                # Clear any stale doctor from a previous turn so we don't scope to wrong provider.
                if state.entities.doctor:
                    logger.info(
                        "Enrichment: clinic-only query → clearing stale doctor '%s' + provider_id",
                        state.entities.doctor,
                    )
                    state.entities.doctor = None
                    state.entities.provider_id = None
        except Exception as exc:
            logger.warning("_apply_enrichment failed (non-critical): %s", exc)

    # =========================================================================
    # Intent handlers
    # =========================================================================

    async def _handle_price_and_availability(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        *,
        date_hint: Optional[str],
        now_dt: datetime,
    ) -> List[Document]:
        # [FIX] Resolve entities once — if this fails with a known disambiguation
        # reason (ambiguous clinic/doctor) let it propagate so the caller can show
        # the user a selection prompt.  Any other exception is caught below.
        try:
            clinic_id, provider_id, provider_entry = await self._resolve_entities(state, audit, question)
        except MCPWorkflowError as exc:
            if exc.reason in {
                "clinic_ambiguous", "provider_ambiguous",
                "provider_clinic_mismatch", "provider_low_confidence",
            }:
                raise  # Let chat.py handle disambiguation prompts
            # For any other resolution error (not_found, network, etc.)
            # fall through with no IDs — the independent handlers below will
            # decide whether to surface useful partial results or raise.
            logger.warning(
                "Entity resolution failed in combined handler (reason=%s): %s — attempting partial fallback",
                exc.reason, exc,
            )
            clinic_id, provider_id, provider_entry = None, None, None

        # [FIX] Fetch price and schedule INDEPENDENTLY.
        # A failure in one must NEVER kill the other — the user asked for both
        # and deserves at least whichever part the API can answer right now.
        docs: List[Document] = []
        price_error: Optional[str] = None
        schedule_error: Optional[str] = None

        # ── Price leg ────────────────────────────────────────────────────────
        try:
            price_docs, _ = await self._handle_pricing(state, audit, question)
            docs.extend(price_docs)
        except MCPWorkflowError as exc:
            # Re-raise disambiguation errors — those need user input
            if exc.reason in {
                "clinic_ambiguous", "provider_ambiguous",
                "provider_clinic_mismatch", "provider_low_confidence",
            }:
                raise
            price_error = exc.reason
            logger.warning("Price leg failed in combined handler (reason=%s): %s", exc.reason, exc)
        except Exception as exc:
            price_error = "price_exception"
            logger.warning("Unexpected price error in combined handler: %s", exc)

        # ── Schedule leg ─────────────────────────────────────────────────────
        try:
            sched_docs, _ = await self._handle_schedule(
                state, audit, question, date_hint=date_hint, now_dt=now_dt
            )
            docs.extend(sched_docs)
        except MCPWorkflowError as exc:
            if exc.reason in {
                "clinic_ambiguous", "provider_ambiguous",
                "provider_clinic_mismatch", "provider_low_confidence",
            }:
                raise
            schedule_error = exc.reason
            logger.warning("Schedule leg failed in combined handler (reason=%s): %s", exc.reason, exc)
        except Exception as exc:
            schedule_error = "schedule_exception"
            logger.warning("Unexpected schedule error in combined handler: %s", exc)

        # ── Append partial-failure notes so the QA model can tell the user ──
        if price_error and not schedule_error and docs:
            docs.append(Document(
                page_content="ملاحظة للمساعد: تعذّر جلب الأسعار حالياً. يُنصح المريض بالاتصال بالاستقبال للاستفسار عن السعر.",
                metadata={"source": "mcp.partial_failure", "failed": "price"},
            ))
        elif schedule_error and not price_error and docs:
            docs.append(Document(
                page_content="ملاحظة للمساعد: تعذّر جلب جدول المواعيد حالياً. يُنصح المريض بالاتصال بالاستقبال للسؤال عن المواعيد.",
                metadata={"source": "mcp.partial_failure", "failed": "schedule"},
            ))

        # ── If BOTH legs failed, raise a single clear error ──────────────────
        if not docs:
            raise MCPWorkflowError(
                "عفواً يا فندم، المستشفى مفيهاش معلومات حالياً عن الأسعار أو المواعيد دي. ياريت تتواصل مع الاستقبال وهما هيساعدوك بكل التفاصيل.",
                reason="both_legs_failed",
            )

        if is_radiology_question(question):
            docs.append(Document(page_content=get_radiology_context(question),
                metadata={"source": "radiology.knowledge"}))
        return docs

    async def _handle_pricing(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        try:
            clinic_id, provider_id, provider_entry = await self._resolve_entities(state, audit, question)
        except MCPWorkflowError as exc:
            if exc.reason in {"clinic_not_found", "clinic_ambiguous"} and is_radiology_question(question):
                clinic_id, provider_id, provider_entry = None, None, None
            else:
                raise

        if clinic_id:
            price_resp = await self._client.get_service_price(clinic_id=clinic_id, provider_id=provider_id)
            audit.append(ToolAuditEntry(name="get_service_price", status="success",
                details={"clinic_id": clinic_id, "services": len(price_resp.services)}))
            if not price_resp.services:
                if is_radiology_question(question):
                    return [Document(page_content=get_radiology_context(question),
                        metadata={"source": "radiology.knowledge"})], audit
                raise MCPWorkflowError("معلش يا فندم، المستشفى لسه مسجلتش تفاصيل الأسعار للخدمة دي عندنا. برجاء التواصل مع الاستقبال وهما هيبلغوك بكل التفاصيل.",
                    reason="empty_price_response")
            if not provider_id:
                now_dt = _cairo_now()
                combined = await self._fetch_combined_price_schedule(
                    state=state, audit=audit, question=question,
                    clinic_id=clinic_id, price_response=price_resp,
                    provider_entry=provider_entry, now_dt=now_dt, date_hint=None)
                if combined:
                    docs = combined
                    if is_radiology_question(question):
                        docs.append(Document(page_content=get_radiology_context(question),
                            metadata={"source": "radiology.knowledge"}))
                    return docs, audit
            docs = [Document(page_content=_format_price_context(price_resp, provider_entry, question),
                metadata={"source": "mcp.service_price"})]
            if is_radiology_question(question):
                docs.append(Document(page_content=get_radiology_context(question),
                    metadata={"source": "radiology.knowledge"}))
            return docs, audit

        candidate_names = _infer_candidate_clinics(question)
        if not candidate_names:
            if is_emergency_radiology(question):
                return [Document(page_content=get_emergency_radiology_context(),
                    metadata={"source": "radiology.emergency"})], audit
            if is_radiology_question(question):
                return [Document(page_content=get_radiology_context(question),
                    metadata={"source": "radiology.knowledge"})], audit
            raise MCPWorkflowError("بعتذر لحضرتك، بس مش قادر أتعرف على العيادة المطلوبة. ممكن حضرتك توضحلي اسم العيادة أو نوع الخدمة اللي بتدور عليها؟",
                reason="missing_clinic")

        async def _fetch(cid: int, cname: str):
            try:
                return cid, cname, await self._client.get_service_price(clinic_id=cid)
            except Exception as exc:
                logger.warning("Fan-out failed %s: %s", cname, exc); return cid, cname, None

        resolved = self._resolve_candidate_ids(candidate_names)
        if is_radiology_question(question) and 1 not in resolved:
            resolved[1] = "Radiology"

        results = await asyncio.gather(*(_fetch(cid, cname) for cid, cname in resolved.items()))
        all_docs: List[Document] = []
        for cid, cname, resp in results:
            if not resp or not resp.services:
                continue
            synthetic = ProviderRecord(clinic_id=cid, clinic_name_ar=cname, clinic_name_en=cname)
            audit.append(ToolAuditEntry(name="get_service_price", status="success",
                details={"clinic_id": cid, "clinic": cname, "services": len(resp.services)}))
            all_docs.append(Document(page_content=_format_price_context(resp, synthetic, question),
                metadata={"source": "mcp.service_price", "clinic_id": cid}))

        if not all_docs:
            if is_radiology_question(question):
                return [Document(
                    page_content=get_radiology_context(question) + "\n\nالسعر: غير متوفر. اتصل بالاستقبال.",
                    metadata={"source": "radiology.knowledge"})], audit
            raise MCPWorkflowError("عفواً يا فندم، المستشفى لسه مسجلتش سعر الخدمة دي عندنا. تقدر تتواصل مع الاستقبال عشان يفيدوك بالسعر المظبوط.", reason="empty_price_response")
        return all_docs, audit

    async def _handle_schedule(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        *,
        date_hint: Optional[str] = None,
        now_dt: Optional[datetime] = None,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        period_kws = ["صبح", "صباح", "ليل", "مساء", "اي وقت", "أي وقت", "كل اليوم", "مش فارق", "بالليل", "يوم", "كلها", "طول"]
        q_lower = (question or "").casefold()
        if not any(kw in q_lower for kw in period_kws):
            raise MCPWorkflowError(
                "عشان أقدر أساعدك أحسن، تحب الموعد يكون فترة الصبح ولا بالليل؟ (أو ممكن تقول 'أي وقت')",
                reason="missing_period"
            )

        clinic_id, provider_id, provider_entry = await self._resolve_entities(state, audit, question)
        if not clinic_id:
            if is_emergency_radiology(question):
                return [Document(page_content=get_emergency_radiology_context(),
                    metadata={"source": "radiology.emergency"})], audit
            if is_radiology_question(question):
                return [Document(page_content=get_radiology_context(question),
                    metadata={"source": "radiology.knowledge"})], audit
            raise MCPWorkflowError("تحت أمرك، بس محتاج أعرف اسم العيادة الأول عشان أقدر أجيبلك المواعيد المظبوطة.", reason="missing_clinic")

        base_dt = now_dt or _cairo_now()
        date_from, date_to, _, _ = _infer_date_range(question, now_dt=base_dt, date_hint=date_hint)
        rad_doc = None
        if is_radiology_question(question):
            rad_doc = Document(page_content=get_radiology_context(question),
                metadata={"source": "radiology.knowledge"})

        if not provider_id:
            try:
                provider_list = await self._client.get_clinic_provider_list()
                clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
            except Exception:
                clinic_providers = []

            bulk = await self._client.get_clinic_provider_schedule(
                clinic_id=clinic_id, provider_id=None, date_from=date_from, date_to=date_to)

            slots_by_provider: Dict[int, List] = defaultdict(list)
            for slot in bulk.slots:
                if slot.provider_id is not None:
                    slots_by_provider[slot.provider_id].append(slot)

            provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}
            active = [(provider_map[pid], ProviderScheduleResponse(slots=slots_by_provider[pid]))
                      for pid in slots_by_provider if slots_by_provider[pid] and pid in provider_map]

            if not active and not bulk.slots:
                sem = asyncio.Semaphore(8)
                async def _per(p: ProviderRecord):
                    async with sem:
                        try:
                            resp = await self._client.get_clinic_provider_schedule(
                                clinic_id=clinic_id, provider_id=p.provider_id,
                                date_from=date_from, date_to=date_to)
                            return p, resp
                        except Exception:
                            return p, ProviderScheduleResponse(slots=[])
                per_results = await asyncio.gather(*(_per(p) for p in clinic_providers))
                active = [(p, r) for p, r in per_results if r.slots]

            if not active:
                raise MCPWorkflowError(
                    f"معلش يا فندم، دورت كويس في المستشفى ومش لاقي مواعيد متاحة للعيادة دي من {date_from} لـ {date_to}. ممكن حضرتك تتواصل مع الاستقبال للتأكد أو لتسجيل اسمك في قائمة الانتظار.",
                    reason="empty_schedule_response")

            # Cap to 9 doctors — prevents token overflow on large clinics (dental, etc.)
            MAX_DOCTORS = 9
            total_doctors = len(active)
            active = active[:MAX_DOCTORS]

            clinic_label = state.entities.clinic or "العيادة"
            if provider_entry:
                clinic_label = provider_entry.clinic_name_ar or provider_entry.clinic_name_en or clinic_label
            context = _format_multi_doctor_schedule(clinic_label, active)
            if total_doctors > MAX_DOCTORS:
                context += f"\n\n[ملاحظة: يوجد {total_doctors} دكتور في العيادة — تم عرض أول {MAX_DOCTORS} فقط. للمزيد اتصل بالاستقبال.]"
            audit.append(ToolAuditEntry(name="get_clinic_schedule_all", status="success",
                details={"clinic_id": clinic_id, "active": total_doctors, "shown": len(active), "date_from": date_from}))
            docs = [Document(page_content=context, metadata={"source": "mcp.clinic_schedule_all"})]
            if rad_doc:
                docs.append(rad_doc)
            return docs, audit

        resp = await self._client.get_clinic_provider_schedule(
            clinic_id=clinic_id, provider_id=provider_id, date_from=date_from, date_to=date_to)
        audit.append(ToolAuditEntry(name="get_clinic_provider_schedule", status="success",
            details={"clinic_id": clinic_id, "provider_id": provider_id,
                     "date_from": date_from, "slots": len(resp.slots)}))
        if not resp.slots:
            raise MCPWorkflowError("بعتذر جداً، مفيش مواعيد مسجلة للدكتور ده في الفترة دي. تقدر حضرتك تتواصل مع الاستقبال وهما هيساعدوك في حجز أقرب موعد متاح.", reason="empty_schedule_response")
        context = _format_schedule_context(resp, provider_entry)
        docs = [Document(page_content=context, metadata={"source": "mcp.provider_schedule"})]
        if rad_doc:
            docs.append(rad_doc)
        return docs, audit

    async def _handle_who_is_present(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        *,
        date_hint: Optional[str] = None,
        now_dt: Optional[datetime] = None,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        clinic_id, _, provider_entry = await self._resolve_entities(state, audit, question)
        if not clinic_id:
            raise MCPWorkflowError("تحت أمرك، بس ياريت توضحلي اسم العيادة عشان أقدر أشوفلك مين من الدكاترة موجود.", reason="missing_clinic")

        base_dt = now_dt or _cairo_now()
        day_id = _infer_target_day_id(question, now_dt=base_dt, date_hint=date_hint)
        date_from, date_to, _, _ = _infer_date_range(question, now_dt=base_dt, date_hint=date_hint)

        try:
            provider_list = await self._client.get_clinic_provider_list()
            clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
        except Exception:
            clinic_providers = []

        bulk = await self._client.get_clinic_provider_schedule(
            clinic_id=clinic_id, provider_id=None, date_from=date_from, date_to=date_to)

        slots_by_provider: Dict[int, List] = defaultdict(list)
        for slot in bulk.slots:
            if slot.provider_id is not None:
                slots_by_provider[slot.provider_id].append(slot)

        provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}
        present: List[Tuple[str, List[str]]] = []
        for pid, slots in slots_by_provider.items():
            p = provider_map.get(pid)
            name = (p.provider_name_ar or p.provider_name_en) if p else "دكتور"
            present.append((name, _slot_strs_from_slots(slots)))

        if not present and not bulk.slots:
            sem = asyncio.Semaphore(8)
            async def _pp(p: ProviderRecord):
                async with sem:
                    try:
                        resp = await self._client.get_clinic_provider_schedule(
                            clinic_id=clinic_id, provider_id=p.provider_id,
                            date_from=date_from, date_to=date_to)
                        return p, resp
                    except Exception:
                        return p, ProviderScheduleResponse(slots=[])
            per_results = await asyncio.gather(*(_pp(p) for p in clinic_providers))
            for p, resp in per_results:
                if resp.slots:
                    name = p.provider_name_ar or p.provider_name_en or "دكتور"
                    present.append((name, _slot_strs_from_slots(resp.slots)))

        audit.append(ToolAuditEntry(name="get_clinic_provider_schedule", status="success",
            details={"clinic_id": clinic_id, "day_id": day_id, "date": date_from, "present": len(present)}))

        if not present:
            raise MCPWorkflowError(f"معلش يا فندم، المستشفى مش مسجلة إن فيه دكاترة موجودين في العيادة دي في {date_from}. ممكن تتواصل مع الاستقبال للتأكد من الجدول.",
                reason="empty_schedule_response")

        clinic_label = state.entities.clinic or "العيادة"
        if provider_entry:
            clinic_label = provider_entry.clinic_name_ar or provider_entry.clinic_name_en or clinic_label
        context = _format_who_is_present(clinic_label, day_id, present)
        return [Document(page_content=context, metadata={"source": "mcp.who_is_present"})], audit

    async def _handle_list_doctors(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        provider_list = await self._client.get_clinic_provider_list()
        clinic_id = getattr(state.entities, "clinic_id", None)
        clinic_name = state.entities.clinic

        if clinic_id:
            filtered = _filter_provider_list_by_clinic_id(provider_list, clinic_id)
        elif clinic_name:
            normalized = _normalize_clinic_name(clinic_name)
            cid = _hardcoded_clinic_id(normalized or clinic_name)
            if cid:
                filtered = _filter_provider_list_by_clinic_id(provider_list, cid)
            else:
                filtered = ProviderListPayload(providers=[
                    p for p in provider_list.providers
                    if p.clinic_name_ar and (
                        clinic_name.casefold() in p.clinic_name_ar.casefold() or
                        (normalized and normalized.casefold() in p.clinic_name_ar.casefold())
                    )
                ] or provider_list.providers[:15])
        else:
            filtered = ProviderListPayload(providers=provider_list.providers[:20])

        audit.append(ToolAuditEntry(name="get_clinic_provider_list", status="success",
            details={"total": len(provider_list.providers), "filtered": len(filtered.providers)}))

        if not filtered.providers:
            raise MCPWorkflowError("عفواً يا فندم، مش لاقي أسماء الدكاترة دي في المستشفى. ممكن حضرتك تتأكد من الاسم أو تتواصل مع الاستقبال للمساعدة.", reason="provider_not_found")
        context = _format_provider_list(filtered)
        return [Document(page_content=context, metadata={"source": "mcp.provider_list"})], audit

    async def _fetch_combined_price_schedule(
        self,
        *,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        clinic_id: int,
        price_response: ServicePriceResponse,
        provider_entry: Optional[ProviderRecord],
        now_dt: datetime,
        date_hint: Optional[str],
    ) -> List[Document]:
        date_from, date_to, _, _ = _infer_date_range(question, now_dt=now_dt, date_hint=date_hint)
        try:
            provider_list = await self._client.get_clinic_provider_list()
            clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
        except Exception:
            return []
        if not clinic_providers:
            return []
        try:
            bulk = await self._client.get_clinic_provider_schedule(
                clinic_id=clinic_id, provider_id=None, date_from=date_from, date_to=date_to)
        except Exception:
            return []

        audit.append(ToolAuditEntry(name="get_clinic_schedule_combined", status="success",
            details={"clinic_id": clinic_id, "date_from": date_from, "slots": len(bulk.slots)}))

        slots_by_provider: Dict[int, List] = defaultdict(list)
        for slot in bulk.slots:
            if slot.provider_id is not None:
                slots_by_provider[slot.provider_id].append(slot)

        provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}

        def _earliest_pid(pid: int) -> int:
            return min((_time_str_to_minutes(s.shift_start or "") for s in slots_by_provider.get(pid, [])), default=9999)

        sorted_pids = sorted([pid for pid in slots_by_provider if slots_by_provider[pid]], key=_earliest_pid)

        # Cap to 9 doctors — prevents token overflow on large clinics (dental, etc.)
        MAX_DOCTORS = 9
        total_pids = len(sorted_pids)
        sorted_pids = sorted_pids[:MAX_DOCTORS]

        doctors_data = []
        for pid in sorted_pids:
            p = provider_map.get(pid) or ProviderRecord(
                clinic_id=clinic_id, clinic_name_ar=state.entities.clinic or "العيادة")
            doctors_data.append((p, _slot_strs_from_slots(slots_by_provider[pid]), price_response))

        if not doctors_data:
            return []

        clinic_label = ((provider_entry.clinic_name_ar or provider_entry.clinic_name_en)
                        if provider_entry else (state.entities.clinic or "العيادة"))
        context = _format_combined_price_schedule(clinic_name=clinic_label, doctors=doctors_data, question=question)
        if total_pids > MAX_DOCTORS:
            context += f"\n\n[ملاحظة: يوجد {total_pids} دكتور في العيادة — تم عرض أول {MAX_DOCTORS} فقط. للمزيد اتصل بالاستقبال.]"
        return [Document(page_content=context, metadata={"source": "mcp.combined_price_schedule"})]

    async def _resolve_entities(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
    ) -> Tuple[Optional[int], Optional[int], Optional[ProviderRecord]]:
        """
        Resolve clinic_id and provider_id from current state entities.

        [STABILITY] Error handling:
        - Network / timeout errors on doctor match → raise MCPWorkflowError so the
          user gets a clear retry message instead of a silent wrong answer.
        - Score-below-threshold on doctor match → clear state.entities.doctor +
          provider_id so the next turn starts clean (no ghost entity).
        - Ambiguous clinic → raise for disambiguation (unchanged).
        - Clinic not found → raise with helpful message (unchanged).
        """
        doctor_name = state.entities.doctor
        clinic_name = _normalize_clinic_name(state.entities.clinic)
        clinic_id: Optional[int] = getattr(state.entities, "clinic_id", None)
        provider_id: Optional[int] = getattr(state.entities, "provider_id", None)
        provider_entry: Optional[ProviderRecord] = None

        # ── Clinic resolution ─────────────────────────────────────────────────
        if clinic_id is None and clinic_name:
            cid = _hardcoded_clinic_id(clinic_name)
            if cid is not None:
                clinic_id = cid
                provider_entry = ProviderRecord(clinic_id=clinic_id,
                    clinic_name_ar=clinic_name, clinic_name_en=clinic_name)
                logger.debug("Hardcoded clinic: '%s' -> id=%s", clinic_name, clinic_id)

        if clinic_id is None and clinic_name:
            try:
                match = await self._client.match_clinic_hybrid(query=clinic_name, top_k=1, min_score=0.3)
                best = match.best_match
                if best and (match.status == HybridMatchStatus.UNAMBIGUOUS_MATCH or best.score >= 0.40):
                    cid = _safe_parse_int(best.clinic_id)
                    if cid:
                        clinic_id = cid
                        provider_entry = ProviderRecord(clinic_id=clinic_id,
                            clinic_name_ar=best.clinic_name or clinic_name,
                            clinic_name_en=best.clinic_name or clinic_name)
                elif match.status == HybridMatchStatus.AMBIGUOUS_MATCH and match.candidates:
                    raise MCPWorkflowError("في أكتر من عيادة بالاسم ده. اختار:",
                        reason="clinic_ambiguous",
                        data={"candidates": [c.__dict__ for c in (match.candidates or [])]})
                elif match.status not in (HybridMatchStatus.UNAMBIGUOUS_MATCH,):
                    raise MCPWorkflowError(f"معلش يا فندم، دورت ومش لاقي عيادة بالاسم ده في المستشفى. ممكن حضرتك تتأكد من اسم العيادة أو تتواصل مع الاستقبال لمساعدتك؟",
                        reason="clinic_not_found")
            except MCPWorkflowError:
                raise
            except Exception as exc:
                logger.warning("Clinic match error: %s", exc)

        # ── Doctor resolution ─────────────────────────────────────────────────
        if doctor_name and provider_id is None:
            _network_error = False
            try:
                match = await self._client.match_doctor_hybrid(
                    query=doctor_name, clinic_id=clinic_id, top_k=3)

                if match and match.status == HybridMatchStatus.UNAMBIGUOUS_MATCH and match.best_match:
                    best = match.best_match
                    provider_id = _safe_parse_int(best.provider_id)
                    if not clinic_id:
                        clinic_id = _safe_parse_int(best.clinic_id)
                    provider_entry = ProviderRecord(provider_id=provider_id, clinic_id=clinic_id,
                        provider_name_ar=best.name_ar, provider_name_en=best.name_en,
                        clinic_name_ar=best.clinic_name, clinic_name_en=best.clinic_name)

                elif match and match.status == HybridMatchStatus.AMBIGUOUS_MATCH:
                    raise MCPWorkflowError("في أكتر من دكتور بالاسم ده. اختار:",
                        reason="provider_ambiguous",
                        data={"candidates": [c.__dict__ for c in (match.candidates or [])]})

                else:
                    # [STABILITY] No match found for this doctor name.
                    # Clear the stale entity so the next turn doesn't reuse it.
                    logger.info(
                        "Doctor '%s' not found in MCP (status=%s) — raising friendly error",
                        doctor_name, match.status if match else "None",
                    )
                    state.entities.doctor = None
                    state.entities.provider_id = None
                    raise MCPWorkflowError(
                        f"معلش يا فندم دورت كويس ومش لاقي دكتور بالاسم ده في المستشفى. ممكن حضرتك تتأكد من الاسم تاني أو تتواصل مع الاستقبال؟",
                        reason="doctor_not_found"
                    )

            except MCPWorkflowError:
                raise
            except Exception as exc:
                # [STABILITY] Network/timeout error
                logger.error("Doctor match network error for '%s': %s", doctor_name, exc)
                state.entities.doctor = None
                state.entities.provider_id = None
                raise MCPWorkflowError(
                    f"معلش يا فندم حصل مشكلة في البحث عن الدكتور. ممكن حضرتك تحاول تاني؟",
                    reason="doctor_match_error",
                )

        audit.append(ToolAuditEntry(name="resolve_entities", status="success",
            details={"clinic_id": clinic_id, "provider_id": provider_id,
                     "clinic_name": clinic_name, "doctor_name": doctor_name}))
        return clinic_id, provider_id, provider_entry

    def _resolve_candidate_ids(self, candidate_names: List[str]) -> Dict[int, str]:
        resolved: Dict[int, str] = {}
        for name in candidate_names:
            cid = _hardcoded_clinic_id(name)
            if cid is not None and cid not in resolved:
                resolved[cid] = name
        return resolved


# =============================================================================
# Clinic inference from question text (keyword fallback)
# =============================================================================

_SERVICE_TO_CLINICS: List[Tuple[List[str], List[str]]] = [
    (["رنين", "مرنانة", "مرنانه", "mri"], ["عيادة اشعه"]),
    (["ماموجرام", "mammogram", "أشعة ثدي"], ["عيادة اشعه"]),
    (["ديكسا", "dexa", "هشاشة عظام", "كثافة عظام"], ["عيادة اشعه"]),
    (["pet scan", "بت سكان", "مسح ذري"], ["عيادة اشعه مقطعيه"]),
    (["بانوراما", "باناراما", "سيفالوميتريك"], ["عيادة أسنان"]),
    (["سكانر", "مقطعي", "ct"], ["عيادة اشعه مقطعيه"]),
    (["سونار", "تلفزيوني", "دوبلر", "doppler", "ultrasound"], ["عيادة اشعه تلفزيونيه"]),
    (["إكس راي", "x-ray", "xray", "أشعة عادية"], ["عيادة اشعه عاديه"]),
    (["اشعة", "اشعه", "أشعة", "أشعه"], ["عيادة اشعه", "عيادة اشعه مقطعيه", "عيادة اشعه تلفزيونيه"]),
    # تداخلات routing hints
    (["رسم قلب", "تخطيط قلب", "ecg", "ekg", "ايكو", "هولتر"], ["عيادة قلب", "عيادة باطنة"]),
    (["رسم مخ", "تخطيط مخ", "eeg", "كهربا مخ"], ["عيادة أعصاب"]),
    (["كشف كمبيوتر", "قاع عين", "ضغط عين", "كشف نظارة"], ["عيادة رمد"]),
    (["غيار", "تضميد", "ضمادة", "خياطة"], ["عيادة جراحة"]),
    (["زيركون", "تركيبات", "حشو", "خلع ضرس", "تقويم"], ["عيادة أسنان"]),
    (["جلسة علاج طبيعي", "كهربا علاج", "موجات صوتية"], ["عيادة علاج طبيعي"]),
    (["منظار معدة", "منظار قولون", "endoscopy"], ["عيادة باطنة"]),
    (["حقنة مفصل", "كورتيزون", "هيالورونيك"], ["عيادة عظام"]),
    # General specialties
    (["تقويم", "حشو", "خلع", "ضرس", "أسنان", "اسنان", "سن", "لثة"], ["عيادة أسنان"]),
    (["كسر", "عظم", "مفصل", "خشونة", "ركبة", "كتف", "ورك"], ["عيادة عظام"]),
    (["جرح", "خياطة", "ضمادة", "تغيير الجرح", "عملية", "زائدة", "مرارة"], ["عيادة جراحة"]),
    (["عيون", "نظر", "نضارة", "عدسة", "شبكية"], ["عيادة رمد"]),
    (["أنف", "أذن", "لوز", "جيوب", "سمع", "شخير"], ["عيادة أنف وأذن وحنجرة"]),
    (["جلد", "بشرة", "حبوب", "حساسية جلد", "طفح"], ["عيادة جلدية"]),
    (["أطفال", "اطفال", "طفل", "رضيع"], ["عيادة أطفال"]),
    (["حمل", "ولادة", "نسا", "دورة", "رحم", "مبيض"], ["عيادة نسا وتوليد"]),
    (["قلب", "ضغط", "شريان"], ["عيادة قلب"]),
    (["صدر", "ربو", "رئة"], ["عيادة صدر"]),
    (["أعصاب", "صداع", "دوخة", "تنميل", "شلل", "مخ"], ["عيادة أعصاب"]),
    (["باطنة", "معدة", "قولون", "كبد", "هضم"], ["عيادة باطنة"]),
    (["مسالك", "كلى", "بروستاتا", "مثانة", "حصى"], ["عيادة مسالك بولية"]),
    (["سكر", "غدة", "هرمون", "درقية"], ["عيادة سكر وغدد صماء"]),
    (["نفسية", "اكتئاب", "قلق", "ارق", "توتر"], ["عيادة نفسية"]),
    (["علاج طبيعي", "فيزيوثيرابي", "تأهيل", "جلسات"], ["عيادة علاج طبيعي"]),
    (["أورام", "سرطان", "كيماوي"], ["عيادة أورام"]),
    (["بوتوكس", "فيلر", "تجميل"], ["عيادة جراحه تجميل"]),
    (["تقسيط", "كاش", "الدفع", "بالكارت", "فيزا"], []),
]


def _infer_candidate_clinics(question: str) -> List[str]:
    q = (question or "").casefold()
    candidates: List[str] = []
    added: set = set()
    for keywords, clinic_names in _SERVICE_TO_CLINICS:
        if any(kw in q for kw in keywords):
            for cn in clinic_names:
                if cn not in added:
                    candidates.append(cn)
                    added.add(cn)
    return candidates