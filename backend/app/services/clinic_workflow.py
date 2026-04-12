"""
clinic_workflow.py — Intelligent MCP Orchestration Engine
==========================================================
Handles ANY Egyptian Arabic clinic-related question by:

1. Understanding the REAL intent (price / availability / who-is-present /
   combined / list-doctors / booking) from RouteDecision
2. Resolving clinic & doctor identifiers via MCP with smart fallbacks
3. Running MCP tool calls in parallel where possible
4. Merging results into rich LLM-ready context
5. Generating a warm, natural Arabic answer via QAEngine

Key improvements over the old version:
- ask_price_and_availability: fetches price + schedule for the CORRECT date
- _handle_pricing_with_schedule: uses _infer_date_range (not hardcoded today)
- who_is_present: properly preserved alongside price in multi-intent
- Payment questions (تقسيط/كاش): treated as ask_price
- All clinic/doctor inferences use RouteDecision.enriched_* fields first
  (from the LLM router) before falling back to regex heuristics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions & Result Models
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Clinic synonym / normalization maps
# (kept minimal — LLM router now handles most normalization)
# ─────────────────────────────────────────────────────────────────────────────

CLINIC_SYNONYMS: Dict[str, str] = {
    "عيون": "رمد", "العيون": "الرمد", "نظر": "رمد",
    "باطني": "باطنة", "باطنيه": "باطنة", "باطنية": "باطنة", "باطن": "باطنة",
    "نسائية": "نسا وتوليد", "نساء": "نسا وتوليد", "نسا": "نسا وتوليد",
    "ولادة": "نسا وتوليد", "نسا وولادة": "نسا وتوليد", "نساء وتوليد": "نسا وتوليد",
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

# Hardcoded clinic ID map — bypasses fuzzy matching for well-known specialties
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
}


def _normalize_clinic_name(name: Optional[str]) -> Optional[str]:
    """Normalize clinic name to canonical form via synonym map."""
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
    """Fast lookup for well-known clinic specialties."""
    key = (name or "").strip()
    for prefix in ("عيادة ", "عياده "):
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
            break
    for candidate in (key, key.replace("ة", "ه"), key.replace("ه", "ة")):
        if candidate in _HARDCODED_CLINIC_IDS:
            return _HARDCODED_CLINIC_IDS[candidate]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Returns (date_from, date_to, is_specific_day, is_today) as DD/MM/YYYY strings.

    Accepts an explicit date_hint from the LLM router ("today" | "tomorrow" |
    "saturday" | ...) which takes priority over keyword scanning.
    """
    base = now_dt or _cairo_now()
    q = (question or "").casefold()

    def _fmt(dt: datetime) -> str:
        return dt.strftime("%d/%m/%Y")

    # ── LLM-provided date hint (highest priority) ─────────────────────────────
    if date_hint:
        hint = date_hint.lower().strip()
        if hint in ("today", "النهارده", "اليوم"):
            d = _fmt(base)
            return d, d, True, True
        if hint in ("tomorrow", "بكرة", "بكره", "غدا"):
            t = base + timedelta(days=1)
            d = _fmt(t)
            return d, d, True, False
        # Weekday name
        for ar, en in _AR_WEEKDAY_TO_EN.items():
            if hint in (en, ar):
                days_ahead = (_PY_WEEKDAY[en] - base.weekday()) % 7
                target = base + timedelta(days=days_ahead)
                d = _fmt(target)
                return d, d, True, False

    # ── Keyword scan fallback ─────────────────────────────────────────────────
    if any(tok in q for tok in ["النهارده", "نهارده", "اليوم", "دلوقتي", "الآن", "حاليا"]):
        d = _fmt(base)
        return d, d, True, True

    if any(tok in q for tok in ["بكره", "بكرة", "غدا", "غداً"]):
        t = base + timedelta(days=1)
        d = _fmt(t)
        return d, d, True, False

    for ar, en in _AR_WEEKDAY_TO_EN.items():
        if ar in q:
            days_ahead = (_PY_WEEKDAY[en] - base.weekday()) % 7
            target = base + timedelta(days=days_ahead)
            d = _fmt(target)
            return d, d, True, False

    # ── No specific day → full week ───────────────────────────────────────────
    date_from = _fmt(base)
    date_to = _fmt(base + timedelta(days=6))
    return date_from, date_to, False, False


def _infer_target_day_id(question: str, *, now_dt: Optional[datetime] = None, date_hint: Optional[str] = None) -> int:
    """Return MCP day_id (1=Sat … 7=Fri)."""
    base = now_dt or _cairo_now()
    q = (question or "").casefold()

    if date_hint:
        hint = date_hint.lower()
        if hint in ("today", "النهارده", "اليوم"):
            return DAY_NAME_TO_ID[base.strftime("%A").lower()]
        if hint in ("tomorrow", "بكرة", "بكره"):
            t = base + timedelta(days=1)
            return DAY_NAME_TO_ID[t.strftime("%A").lower()]
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
    if not t:
        return 9999
    is_pm = any(m in t for m in ["مساء", "مساءً", "PM", "pm", "م"])
    is_am = any(m in t for m in ["صباح", "صباحًا", "AM", "am", "ص"])
    m = re.search(r"(\d{1,2})[:.](\d{2})", t)
    if not m:
        return 9999
    h, mn = int(m.group(1)), int(m.group(2))
    if h >= 13:
        return h * 60 + mn
    if is_pm and h != 12:
        h += 12
    elif is_am and h == 12:
        h = 0
    return h * 60 + mn


def _safe_parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.strip().lower() in ("", "none", "null"):
            return None
        try:
            return int(value)
        except ValueError:
            return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_ampm(value: str) -> str:
    if not value:
        return value
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
) -> str:
    """Format service prices into LLM-friendly Arabic context."""
    clinic = ""
    if provider_entry:
        clinic = provider_entry.clinic_name_ar or provider_entry.clinic_name_en or ""
    doctor = ""
    if provider_entry:
        doctor = provider_entry.provider_name_ar or provider_entry.provider_name_en or ""

    lines = []
    if clinic:
        lines.append(f"أسعار {clinic}:")
    elif doctor:
        lines.append(f"أسعار الدكتور {doctor}:")
    else:
        lines.append("الأسعار المتاحة:")

    valid = [s for s in price_response.services if s.price is not None]

    # Filter by doctor name if we have one
    if doctor:
        target = doctor.casefold().replace(" ", "")
        doctor_specific = [
            s for s in valid
            if s.doctor_name and target in str(s.doctor_name).casefold().replace(" ", "")
        ]
        if doctor_specific:
            valid = doctor_specific

    if not valid:
        lines.append("- السعر غير متوفر حالياً في النظام.")
    else:
        for s in valid:
            name = s.service_name_ar or s.service_name_en or "كشف"
            price = f"{s.price:.2f}"
            currency = s.currency or "جنيه"
            doc_label = f" — {s.doctor_name}" if getattr(s, "doctor_name", None) else ""
            lines.append(f"- {name}{doc_label}: {price} {currency}")

    return "\n".join(lines)


def _format_schedule_context(
    response: ProviderScheduleResponse,
    provider_entry: Optional[ProviderRecord],
) -> str:
    """Format a schedule into readable Arabic context."""
    doctor = ""
    clinic = ""
    if provider_entry:
        doctor = provider_entry.provider_name_ar or provider_entry.provider_name_en or ""
        clinic = provider_entry.clinic_name_ar or provider_entry.clinic_name_en or ""

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
        entry = f"{start} → {end}"
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
    """Format all providers' schedules sorted by earliest shift start."""
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
) -> str:
    """Format a combined per-doctor block showing schedule + price."""
    lines = [
        f"بيانات دكاترة {clinic_name} — المواعيد والأسعار:",
        "",
        "[تعليمات: اعرض كل دكتور في فقرة منفصلة مع مواعيده وسعر الكشف]",
        "",
    ]
    for provider, slot_strs, price_resp in doctors:
        name = provider.provider_name_ar or provider.provider_name_en or "دكتور"
        lines.append(f"دكتور {name}")

        if slot_strs:
            if len(slot_strs) == 1:
                lines.append(f"  - المواعيد: من {slot_strs[0].replace(' → ', ' لحد ')}")
            else:
                labels = ["الفترة الأولى", "الفترة التانية", "الفترة الثالثة"]
                for i, slot in enumerate(slot_strs):
                    lbl = labels[i] if i < len(labels) else f"الفترة {i+1}"
                    lines.append(f"  - {lbl}: من {slot.replace(' → ', ' لحد ')}")
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
            if valid:
                for s in valid:
                    sname = s.service_name_ar or s.service_name_en or "كشف"
                    lines.append(f"  - {sname}: {s.price:.2f} {s.currency or 'جنيه'}")
            else:
                lines.append("  - السعر: غير متوفر في النظام")
        else:
            lines.append("  - السعر: غير متوفر في النظام")

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
        result.append(f"{start} → {end}{label}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main Workflow Service
# ─────────────────────────────────────────────────────────────────────────────

class ClinicWorkflowService:
    """
    Coordinates MCP tool calls and produces LLM-ready context for any
    Egyptian Arabic clinic question.
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
                # ── Determine the effective date hint ─────────────────────────
                date_hint = decision.enriched_date_hint
                now_dt = _cairo_now()
                time_context = qa_engine.build_time_context(question)

                # ── Override clinic/doctor from LLM enrichment if state lacks them ─
                self._apply_enrichment(state, decision)

                # ── Dispatch by intent ─────────────────────────────────────────
                intent = decision.intent
                all_intents = decision.all_intents or [intent]

                if intent == "ask_price_and_availability" or (
                    "ask_price" in all_intents and (
                        "check_availability" in all_intents or
                        "who_is_present" in all_intents
                    )
                ):
                    docs = await self._handle_price_and_availability(
                        state, tool_audit, question,
                        date_hint=date_hint, now_dt=now_dt,
                    )

                elif intent == "who_is_present":
                    docs, _ = await self._handle_who_is_present(
                        state, tool_audit, question,
                        date_hint=date_hint, now_dt=now_dt,
                    )
                    if "ask_price" in all_intents:
                        try:
                            price_docs, _ = await self._handle_pricing(state, tool_audit, question)
                            docs.extend(price_docs)
                        except MCPWorkflowError:
                            pass

                elif intent == "ask_price":
                    docs, _ = await self._handle_pricing(state, tool_audit, question)

                elif intent in ("check_availability", "book_appointment"):
                    docs, _ = await self._handle_schedule(
                        state, tool_audit, question,
                        date_hint=date_hint, now_dt=now_dt,
                    )

                elif intent == "list_doctors":
                    docs, _ = await self._handle_list_doctors(state, tool_audit)

                else:
                    # Unknown MCP intent — try availability as safe default
                    logger.warning("Unknown MCP intent '%s' — defaulting to check_availability", intent)
                    docs, _ = await self._handle_schedule(
                        state, tool_audit, question,
                        date_hint=date_hint, now_dt=now_dt,
                    )

                span.set_attribute("workflow.docs_count", len(docs))

            except MCPWorkflowError:
                raise
            except Exception as exc:
                logger.error("Unexpected error in clinic workflow: %s", exc, exc_info=True)
                raise MCPWorkflowError(
                    "حصل خطأ غير متوقع أثناء جلب بيانات العيادة. ممكن تحاول تاني؟",
                    reason="unexpected_error",
                )

            qa_payload = await qa_engine.answer_question(
                question=question,
                contexts=docs,
                time_context=time_context,
                chat_history=chat_history,
                user_gender=user_gender,
            )

            return ClinicWorkflowResult(
                qa_response=qa_payload,
                tool_audit=tool_audit,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Entity enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_enrichment(self, state: ConversationState, decision: RouteDecision) -> None:
        """Apply LLM-extracted entities from decision to state if state is missing them."""
        try:
            if decision.enriched_clinic and not state.entities.clinic:
                state.entities.clinic = decision.enriched_clinic
                state.target_entity_type = "clinic"

            if decision.enriched_doctor and not state.entities.doctor:
                state.entities.doctor = decision.enriched_doctor
                state.target_entity_type = "doctor"
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Intent handlers
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_price_and_availability(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
        *,
        date_hint: Optional[str],
        now_dt: datetime,
    ) -> List[Document]:
        """
        Handle combined price + availability (and/or who-is-present) queries.
        Fetches price and schedule for the CORRECT requested date in parallel.
        """
        clinic_id, provider_id, provider_entry = await self._resolve_entities(
            state, audit, question
        )

        if not clinic_id:
            # Try pricing path which has its own fan-out fallback
            try:
                docs, _ = await self._handle_pricing(state, audit, question)
                return docs
            except MCPWorkflowError:
                raise MCPWorkflowError(
                    "مش قادر أحدد العيادة أو الدكتور المطلوب. ممكن توضح أكتر؟",
                    reason="missing_clinic",
                )

        # ── Determine date range ─────────────────────────────────────────────
        date_from, date_to, is_specific, is_today = _infer_date_range(
            question, now_dt=now_dt, date_hint=date_hint
        )

        # ── Parallel fetch: price + schedule ─────────────────────────────────
        async def _fetch_price():
            try:
                return await self._client.get_service_price(
                    clinic_id=clinic_id, provider_id=provider_id
                )
            except Exception as exc:
                logger.warning("Price fetch failed: %s", exc)
                return None

        async def _fetch_schedule():
            try:
                return await self._client.get_clinic_provider_schedule(
                    clinic_id=clinic_id,
                    provider_id=None,  # Get all doctors
                    date_from=date_from,
                    date_to=date_to,
                )
            except Exception as exc:
                logger.warning("Schedule fetch failed: %s", exc)
                return None

        price_resp, bulk_schedule = await asyncio.gather(_fetch_price(), _fetch_schedule())

        audit.append(ToolAuditEntry(
            name="get_service_price",
            status="success" if price_resp else "failed",
            details={"clinic_id": clinic_id, "services": len(price_resp.services) if price_resp else 0},
        ))
        audit.append(ToolAuditEntry(
            name="get_clinic_provider_schedule",
            status="success" if bulk_schedule else "failed",
            details={"date_from": date_from, "date_to": date_to, "slots": len(bulk_schedule.slots) if bulk_schedule else 0},
        ))

        # ── Build combined context ────────────────────────────────────────────
        docs: List[Document] = []
        clinic_label = (
            (provider_entry.clinic_name_ar or provider_entry.clinic_name_en)
            if provider_entry else (state.entities.clinic or "العيادة")
        )

        if bulk_schedule and bulk_schedule.slots:
            # Get provider list so we have names
            try:
                provider_list = await self._client.get_clinic_provider_list()
                clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
            except Exception:
                clinic_providers = []

            slots_by_provider: Dict[int, List] = defaultdict(list)
            for slot in bulk_schedule.slots:
                if slot.provider_id is not None:
                    slots_by_provider[slot.provider_id].append(slot)

            provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}

            # Sort providers by earliest slot
            def _earliest(pid: int) -> int:
                slots = slots_by_provider.get(pid, [])
                return min((_time_str_to_minutes(s.shift_start or "") for s in slots), default=9999)

            active_pids = sorted(
                [pid for pid in slots_by_provider if slots_by_provider[pid]],
                key=_earliest,
            )

            if active_pids:
                doctors_data = []
                for pid in active_pids:
                    p = provider_map.get(pid)
                    if p is None:
                        # Synthetic provider record
                        p = ProviderRecord(clinic_id=clinic_id,
                                          clinic_name_ar=clinic_label,
                                          clinic_name_en=clinic_label)
                    slot_strs = _slot_strs_from_slots(slots_by_provider[pid])
                    doctors_data.append((p, slot_strs, price_resp))

                context_text = _format_combined_price_schedule(
                    clinic_name=clinic_label,
                    doctors=doctors_data,
                    question=question,
                )
                docs.append(Document(
                    page_content=context_text,
                    metadata={"source": "mcp.combined_price_schedule", "date": date_from},
                ))
            elif price_resp and price_resp.services:
                # Schedule empty but have price
                docs.append(Document(
                    page_content=_format_price_context(price_resp, provider_entry, question),
                    metadata={"source": "mcp.price_only"},
                ))
                docs.append(Document(
                    page_content=f"ملاحظة: لم يتم العثور على مواعيد محددة لـ{date_from}. يمكنك الاستفسار عن يوم محدد.",
                    metadata={"source": "mcp.no_schedule"},
                ))
        elif price_resp and price_resp.services:
            docs.append(Document(
                page_content=_format_price_context(price_resp, provider_entry, question),
                metadata={"source": "mcp.price_only"},
            ))

        if not docs:
            raise MCPWorkflowError(
                f"مش لاقي أسعار أو مواعيد للعيادة دي في {date_from}. جرب يوم تاني أو اتصل بالاستقبال.",
                reason="empty_combined_response",
            )

        # Enrich radiology questions
        if is_radiology_question(question):
            docs.append(Document(
                page_content=get_radiology_context(question),
                metadata={"source": "radiology.knowledge"},
            ))

        return docs

    async def _handle_pricing(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
    ) -> Tuple[List[Document], List[ToolAuditEntry]]:
        """Handle pure pricing queries with fan-out fallback."""
        try:
            clinic_id, provider_id, provider_entry = await self._resolve_entities(
                state, audit, question
            )
        except MCPWorkflowError as exc:
            if exc.reason in {"clinic_not_found", "clinic_ambiguous"} and is_radiology_question(question):
                clinic_id, provider_id, provider_entry = None, None, None
            else:
                raise

        if clinic_id:
            price_resp = await self._client.get_service_price(
                clinic_id=clinic_id, provider_id=provider_id
            )
            audit.append(ToolAuditEntry(
                name="get_service_price", status="success",
                details={"clinic_id": clinic_id, "services": len(price_resp.services)},
            ))

            if not price_resp.services:
                if is_radiology_question(question):
                    return [Document(
                        page_content=get_radiology_context(question),
                        metadata={"source": "radiology.knowledge"},
                    )], audit
                raise MCPWorkflowError(
                    "مفيش أسعار متاحة في النظام لهذه الخدمة حالياً. اتصل بالاستقبال.",
                    reason="empty_price_response",
                )

            # If no specific doctor → show combined schedule+price for today
            if not provider_id:
                now_dt = _cairo_now()
                combined = await self._fetch_combined_price_schedule(
                    state=state, audit=audit, question=question,
                    clinic_id=clinic_id, price_response=price_resp,
                    provider_entry=provider_entry, now_dt=now_dt,
                    date_hint=None,
                )
                if combined:
                    docs = combined
                    if is_radiology_question(question):
                        docs.append(Document(
                            page_content=get_radiology_context(question),
                            metadata={"source": "radiology.knowledge"},
                        ))
                    return docs, audit

            docs = [Document(
                page_content=_format_price_context(price_resp, provider_entry, question),
                metadata={"source": "mcp.service_price"},
            )]
            if is_radiology_question(question):
                docs.append(Document(
                    page_content=get_radiology_context(question),
                    metadata={"source": "radiology.knowledge"},
                ))
            return docs, audit

        # ── Fan-out: no clinic resolved → try all matching clinics ────────────
        candidate_names = _infer_candidate_clinics(question)
        if not candidate_names:
            if is_emergency_radiology(question):
                return [Document(
                    page_content=get_emergency_radiology_context(),
                    metadata={"source": "radiology.emergency"},
                )], audit
            if is_radiology_question(question):
                return [Document(
                    page_content=get_radiology_context(question),
                    metadata={"source": "radiology.knowledge"},
                )], audit
            raise MCPWorkflowError(
                "مش قادر أحدد العيادة. ممكن تذكر اسم العيادة أو نوع الخدمة؟",
                reason="missing_clinic",
            )

        async def _fetch(cid: int, cname: str):
            try:
                return cid, cname, await self._client.get_service_price(clinic_id=cid)
            except Exception as exc:
                logger.warning("Price fan-out failed for %s: %s", cname, exc)
                return cid, cname, None

        resolved = self._resolve_candidate_ids(candidate_names)
        if is_radiology_question(question) and 1 not in resolved:
            resolved[1] = "Radiology"

        results = await asyncio.gather(*(_fetch(cid, cname) for cid, cname in resolved.items()))
        all_docs: List[Document] = []
        for cid, cname, resp in results:
            if not resp or not resp.services:
                continue
            synthetic = ProviderRecord(clinic_id=cid, clinic_name_ar=cname, clinic_name_en=cname)
            audit.append(ToolAuditEntry(
                name="get_service_price", status="success",
                details={"clinic_id": cid, "clinic": cname, "services": len(resp.services)},
            ))
            all_docs.append(Document(
                page_content=_format_price_context(resp, synthetic, question),
                metadata={"source": "mcp.service_price", "clinic_id": cid},
            ))

        if not all_docs:
            if is_radiology_question(question):
                return [Document(
                    page_content=get_radiology_context(question) +
                    "\n\nالسعر: غير متوفر حالياً. يرجى الاتصال بالاستقبال.",
                    metadata={"source": "radiology.knowledge"},
                )], audit
            raise MCPWorkflowError(
                "مفيش أسعار متاحة في النظام لهذه الخدمة.", reason="empty_price_response"
            )

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
        """Handle schedule / availability queries."""
        clinic_id, provider_id, provider_entry = await self._resolve_entities(
            state, audit, question
        )
        if not clinic_id:
            if is_emergency_radiology(question):
                return [Document(
                    page_content=get_emergency_radiology_context(),
                    metadata={"source": "radiology.emergency"},
                )], audit
            if is_radiology_question(question):
                return [Document(
                    page_content=get_radiology_context(question),
                    metadata={"source": "radiology.knowledge"},
                )], audit
            raise MCPWorkflowError(
                "محتاج أعرف اسم العيادة عشان أجيب المواعيد.", reason="missing_clinic"
            )

        base_dt = now_dt or _cairo_now()
        date_from, date_to, is_specific, is_today = _infer_date_range(
            question, now_dt=base_dt, date_hint=date_hint
        )

        rad_doc = None
        if is_radiology_question(question):
            rad_doc = Document(
                page_content=get_radiology_context(question),
                metadata={"source": "radiology.knowledge"},
            )

        if not provider_id:
            # All doctors in clinic
            try:
                provider_list = await self._client.get_clinic_provider_list()
                clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
            except Exception:
                clinic_providers = []

            bulk = await self._client.get_clinic_provider_schedule(
                clinic_id=clinic_id,
                provider_id=None,
                date_from=date_from,
                date_to=date_to,
            )

            slots_by_provider: Dict[int, List] = defaultdict(list)
            for slot in bulk.slots:
                if slot.provider_id is not None:
                    slots_by_provider[slot.provider_id].append(slot)

            provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}
            active = [
                (provider_map[pid], ProviderScheduleResponse(slots=slots_by_provider[pid]))
                for pid in slots_by_provider
                if slots_by_provider[pid]
            ]

            if not active and not bulk.slots:
                # Per-provider fallback
                sem = asyncio.Semaphore(8)
                async def _per_provider(p: ProviderRecord):
                    async with sem:
                        try:
                            resp = await self._client.get_clinic_provider_schedule(
                                clinic_id=clinic_id, provider_id=p.provider_id,
                                date_from=date_from, date_to=date_to,
                            )
                            return p, resp
                        except Exception:
                            return p, ProviderScheduleResponse(slots=[])

                results = await asyncio.gather(*(_per_provider(p) for p in clinic_providers))
                active = [(p, r) for p, r in results if r.slots]

            if not active:
                raise MCPWorkflowError(
                    f"مفيش مواعيد متاحة في هذه العيادة للفترة من {date_from} لـ {date_to}.",
                    reason="empty_schedule_response",
                )

            clinic_label = state.entities.clinic or "العيادة"
            if provider_entry:
                clinic_label = provider_entry.clinic_name_ar or provider_entry.clinic_name_en or clinic_label

            context = _format_multi_doctor_schedule(clinic_label, active)
            audit.append(ToolAuditEntry(
                name="get_clinic_schedule_all", status="success",
                details={"clinic_id": clinic_id, "active_doctors": len(active), "date_from": date_from},
            ))
            docs = [Document(page_content=context, metadata={"source": "mcp.clinic_schedule_all"})]
            if rad_doc:
                docs.append(rad_doc)
            return docs, audit

        # ── Specific provider ─────────────────────────────────────────────────
        resp = await self._client.get_clinic_provider_schedule(
            clinic_id=clinic_id, provider_id=provider_id,
            date_from=date_from, date_to=date_to,
        )
        audit.append(ToolAuditEntry(
            name="get_clinic_provider_schedule", status="success",
            details={"clinic_id": clinic_id, "provider_id": provider_id,
                     "date_from": date_from, "slots": len(resp.slots)},
        ))

        if not resp.slots:
            raise MCPWorkflowError(
                "مفيش مواعيد متاحة للدكتور ده في الفترة دي.", reason="empty_schedule_response"
            )

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
        """Handle 'who is present today / on a given day' queries."""
        clinic_id, _, provider_entry = await self._resolve_entities(state, audit, question)
        if not clinic_id:
            raise MCPWorkflowError("محتاج اسم العيادة عشان أعرف مين موجود.", reason="missing_clinic")

        base_dt = now_dt or _cairo_now()
        day_id = _infer_target_day_id(question, now_dt=base_dt, date_hint=date_hint)
        date_from, date_to, _, _ = _infer_date_range(question, now_dt=base_dt, date_hint=date_hint)

        try:
            provider_list = await self._client.get_clinic_provider_list()
            clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
        except Exception:
            clinic_providers = []

        bulk = await self._client.get_clinic_provider_schedule(
            clinic_id=clinic_id,
            provider_id=None,
            date_from=date_from,
            date_to=date_to,
        )

        slots_by_provider: Dict[int, List] = defaultdict(list)
        for slot in bulk.slots:
            if slot.provider_id is not None:
                slots_by_provider[slot.provider_id].append(slot)

        provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}

        present: List[Tuple[str, List[str]]] = []
        for pid, slots in slots_by_provider.items():
            p = provider_map.get(pid)
            name = (p.provider_name_ar or p.provider_name_en) if p else "دكتور"
            time_strs = _slot_strs_from_slots(slots)
            present.append((name, time_strs))

        if not present and not bulk.slots:
            # Per-provider fallback
            sem = asyncio.Semaphore(8)
            async def _pp(p: ProviderRecord):
                async with sem:
                    try:
                        resp = await self._client.get_clinic_provider_schedule(
                            clinic_id=clinic_id, provider_id=p.provider_id,
                            date_from=date_from, date_to=date_to,
                        )
                        return p, resp
                    except Exception:
                        return p, ProviderScheduleResponse(slots=[])

            results = await asyncio.gather(*(_pp(p) for p in clinic_providers))
            for p, resp in results:
                if resp.slots:
                    name = p.provider_name_ar or p.provider_name_en or "دكتور"
                    present.append((name, _slot_strs_from_slots(resp.slots)))

        audit.append(ToolAuditEntry(
            name="get_clinic_provider_schedule", status="success",
            details={"clinic_id": clinic_id, "day_id": day_id,
                     "date": date_from, "present_count": len(present)},
        ))

        if not present:
            raise MCPWorkflowError(
                f"مفيش دكاترة موجودين في هذه العيادة في {date_from}.",
                reason="empty_schedule_response",
            )

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
        """Handle 'list all doctors' queries."""
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

        audit.append(ToolAuditEntry(
            name="get_clinic_provider_list", status="success",
            details={"total": len(provider_list.providers), "filtered": len(filtered.providers)},
        ))

        if not filtered.providers:
            raise MCPWorkflowError("مش لاقي دكاترة مسجلين.", reason="provider_not_found")

        context = _format_provider_list(filtered)
        return [Document(page_content=context, metadata={"source": "mcp.provider_list"})], audit

    # ─────────────────────────────────────────────────────────────────────────
    # Combined price + schedule helper (used by _handle_pricing when no doctor)
    # ─────────────────────────────────────────────────────────────────────────

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
        """Merge today's schedule with prices for a clinic (no specific doctor)."""
        date_from, date_to, _, _ = _infer_date_range(
            question, now_dt=now_dt, date_hint=date_hint
        )

        try:
            provider_list = await self._client.get_clinic_provider_list()
            clinic_providers = _filter_provider_list_by_clinic_id(provider_list, clinic_id).providers
        except Exception:
            return []

        if not clinic_providers:
            return []

        try:
            bulk = await self._client.get_clinic_provider_schedule(
                clinic_id=clinic_id,
                provider_id=None,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception:
            return []

        audit.append(ToolAuditEntry(
            name="get_clinic_schedule_combined", status="success",
            details={"clinic_id": clinic_id, "date_from": date_from, "slots": len(bulk.slots)},
        ))

        slots_by_provider: Dict[int, List] = defaultdict(list)
        for slot in bulk.slots:
            if slot.provider_id is not None:
                slots_by_provider[slot.provider_id].append(slot)

        provider_map = {p.provider_id: p for p in clinic_providers if p.provider_id}

        def _earliest_pid(pid: int) -> int:
            return min(
                (_time_str_to_minutes(s.shift_start or "") for s in slots_by_provider.get(pid, [])),
                default=9999,
            )

        sorted_pids = sorted(
            [pid for pid in slots_by_provider if slots_by_provider[pid]],
            key=_earliest_pid,
        )

        doctors_data = []
        for pid in sorted_pids:
            p = provider_map.get(pid, ProviderRecord(
                clinic_id=clinic_id,
                clinic_name_ar=state.entities.clinic or "العيادة",
            ))
            slot_strs = _slot_strs_from_slots(slots_by_provider[pid])
            doctors_data.append((p, slot_strs, price_response))

        if not doctors_data:
            return []

        clinic_label = (
            (provider_entry.clinic_name_ar or provider_entry.clinic_name_en)
            if provider_entry else (state.entities.clinic or "العيادة")
        )
        context = _format_combined_price_schedule(
            clinic_name=clinic_label,
            doctors=doctors_data,
            question=question,
        )
        return [Document(page_content=context, metadata={"source": "mcp.combined_price_schedule"})]

    # ─────────────────────────────────────────────────────────────────────────
    # Entity resolution
    # ─────────────────────────────────────────────────────────────────────────

    async def _resolve_entities(
        self,
        state: ConversationState,
        audit: List[ToolAuditEntry],
        question: str,
    ) -> Tuple[Optional[int], Optional[int], Optional[ProviderRecord]]:
        """
        Resolve clinic_id and provider_id using:
        1. Hardcoded fast-path for known specialties
        2. MCP hybrid matching for everything else
        """
        doctor_name = state.entities.doctor
        clinic_name = _normalize_clinic_name(state.entities.clinic)
        clinic_id: Optional[int] = getattr(state.entities, "clinic_id", None)
        provider_id: Optional[int] = getattr(state.entities, "provider_id", None)
        provider_entry: Optional[ProviderRecord] = None

        # ── Fast-path: hardcoded clinic ID ────────────────────────────────────
        if clinic_id is None and clinic_name:
            cid = _hardcoded_clinic_id(clinic_name)
            if cid is not None:
                clinic_id = cid
                provider_entry = ProviderRecord(
                    clinic_id=clinic_id,
                    clinic_name_ar=clinic_name,
                    clinic_name_en=clinic_name,
                )
                logger.debug("Hardcoded clinic: '%s' → id=%s", clinic_name, clinic_id)

        # ── Hybrid MCP matching for clinic ────────────────────────────────────
        if clinic_id is None and clinic_name:
            try:
                match = await self._client.match_clinic_hybrid(
                    query=clinic_name, top_k=1, min_score=0.3
                )
                best = match.best_match
                if best and (match.status == HybridMatchStatus.UNAMBIGUOUS_MATCH or best.score >= 0.40):
                    cid = _safe_parse_int(best.clinic_id)
                    if cid:
                        clinic_id = cid
                        provider_entry = ProviderRecord(
                            clinic_id=clinic_id,
                            clinic_name_ar=best.clinic_name or clinic_name,
                            clinic_name_en=best.clinic_name or clinic_name,
                        )
                elif match.status == HybridMatchStatus.AMBIGUOUS_MATCH and match.candidates:
                    raise MCPWorkflowError(
                        "في أكتر من عيادة بالاسم ده. اختار:",
                        reason="clinic_ambiguous",
                        data={"candidates": [c.__dict__ for c in (match.candidates or [])]},
                    )
                elif match.status not in (HybridMatchStatus.UNAMBIGUOUS_MATCH,):
                    raise MCPWorkflowError(
                        f"مش لاقي عيادة بـ '{clinic_name}'. ممكن تكتب الاسم بالكامل؟",
                        reason="clinic_not_found",
                    )
            except MCPWorkflowError:
                raise
            except Exception as exc:
                logger.warning("Clinic match error: %s", exc)

        # ── Doctor resolution ─────────────────────────────────────────────────
        if doctor_name and provider_id is None:
            try:
                match = await self._client.match_doctor_hybrid(
                    query=doctor_name,
                    clinic_id=clinic_id,
                    top_k=3,
                )
                if match and match.status == HybridMatchStatus.UNAMBIGUOUS_MATCH and match.best_match:
                    best = match.best_match
                    provider_id = _safe_parse_int(best.provider_id)
                    if not clinic_id:
                        clinic_id = _safe_parse_int(best.clinic_id)
                    provider_entry = ProviderRecord(
                        provider_id=provider_id,
                        clinic_id=clinic_id,
                        provider_name_ar=best.name_ar,
                        provider_name_en=best.name_en,
                        clinic_name_ar=best.clinic_name,
                        clinic_name_en=best.clinic_name,
                    )
                elif match and match.status == HybridMatchStatus.AMBIGUOUS_MATCH:
                    raise MCPWorkflowError(
                        "في أكتر من دكتور بالاسم ده. اختار:",
                        reason="provider_ambiguous",
                        data={"candidates": [c.__dict__ for c in (match.candidates or [])]},
                    )
            except MCPWorkflowError:
                raise
            except Exception as exc:
                logger.warning("Doctor match error: %s", exc)

        audit.append(ToolAuditEntry(
            name="resolve_entities", status="success",
            details={
                "clinic_id": clinic_id, "provider_id": provider_id,
                "clinic_name": clinic_name, "doctor_name": doctor_name,
            },
        ))

        return clinic_id, provider_id, provider_entry

    def _resolve_candidate_ids(self, candidate_names: List[str]) -> Dict[int, str]:
        """Map candidate clinic names to IDs using hardcoded map."""
        resolved: Dict[int, str] = {}
        for name in candidate_names:
            cid = _hardcoded_clinic_id(name)
            if cid is not None and cid not in resolved:
                resolved[cid] = name
        return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Clinic inference from question text
# (fallback when LLM router didn't extract a clinic)
# ─────────────────────────────────────────────────────────────────────────────

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
    (["تقسيط", "كاش", "الدفع", "بالكارت", "فيزا"], []),  # payment only → no specific clinic
]


def _infer_candidate_clinics(question: str) -> List[str]:
    """Infer candidate clinic names from question text."""
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