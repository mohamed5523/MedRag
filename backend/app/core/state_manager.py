"""
state_manager.py — Conversation State Extractor
================================================
Extracts structured state from Egyptian Arabic (Ammya) medical conversations.
Improvements:
- Payment questions (تقسيط/كاش/فيزا) correctly classified as ask_price
- Better triage: symptoms → clinic specialty → MCP intent
- Seniority titles never mistaken for doctor names
- [STABILITY] Doctor name confidence gate: name rejected unless ≥ 2 tokens and not a
  seniority/specialty/non-name word — prevents ghost doctor entities from vague input.
- [STABILITY] merge_states: doctor entity cleared (+ provider_id) when new state
  explicitly has no doctor, avoiding stale doctor leaking into clinic-scope queries.
- [STABILITY] clinic entity cleared when new state explicitly has no clinic.
- [STABILITY] extract_state prompt explicitly forbids extracting doctor from a full
  patient name (e.g. when user types their own name as a greeting or context).
"""

import logging
import os
import re
from typing import List, Literal, Optional, Set

from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.state_manager")

# ---------------------------------------------------------------------------
# Doctor-name guard: words that are NOT personal names even when preceded by دكتور
# ---------------------------------------------------------------------------
_NON_NAME_TOKENS: Set[str] = {
    # Seniority / titles
    "استشاري", "الاستشاري", "استشارى", "اخصائي", "أخصائي", "اخصائى", "أخصائى",
    "مقيم", "رئيس", "قسم", "ممارس", "عام", "مدير",
    # Specialties / clinics (partial list — LLM covers the rest)
    "أطفال", "اطفال", "نسا", "نساء", "توليد", "عظام", "أسنان", "اسنان",
    "باطنة", "باطنه", "باطن", "قلب", "جلدية", "جلديه", "رمد", "عيون",
    "أعصاب", "اعصاب", "جراحة", "جراحه", "أنف", "أذن", "حنجرة", "صدر",
    "صدريه", "مسالك", "سكر", "غدد", "نفسية", "نفسيه", "علاج", "طبيعي",
    "أورام", "اورام", "تجميل", "اشعه", "أشعة", "روماتيزم", "مناعه",
    # Presence / time words (already in prompt but guard anyway)
    "موجود", "متاح", "جاى", "جايه", "بيجي", "رايح", "ماشي", "بيكشف", "بيفتح",
    "الضهر", "الصبح", "العصر", "النهارده", "بكرة", "دلوقتي", "الساعه",
    "السبت", "الأحد", "الاحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة",
}

_MIN_DOCTOR_NAME_TOKENS = 2  # must have at least first + last name


def _is_valid_doctor_name(name: Optional[str]) -> bool:
    """
    Return True only if the extracted doctor name looks like a real personal name.
    Guards against: single-word specialty names, seniority titles, time words.
    """
    if not name:
        return False
    tokens = [t for t in name.strip().split() if t]
    if len(tokens) < _MIN_DOCTOR_NAME_TOKENS:
        # Single-token names are rejected unless they look like a proper Arabic given name.
        # We allow them only when they are NOT in the non-name blocklist.
        if len(tokens) == 1 and tokens[0] not in _NON_NAME_TOKENS:
            # Still accept single token if it starts with uppercase (Latin) — e.g. "Maria"
            # or looks like a proper Arabic name (starts with common name prefixes).
            # Be conservative: reject to avoid false positives.
            return False
        return False
    # Reject if ANY token is a known non-name word
    for tok in tokens:
        if tok in _NON_NAME_TOKENS:
            return False
    return True


class Entities(BaseModel):
    doctor: Optional[str] = Field(None, description="Doctor personal name only (not specialty), or null")
    clinic: Optional[str] = Field(None, description="Clinic/specialty name in Arabic, or null")
    clinic_id: Optional[int] = Field(None, description="Resolved clinic ID if known, or null")
    provider_id: Optional[int] = Field(None, description="Resolved provider ID if known, or null")
    hospital: Optional[str] = Field(None, description="Hospital name, or null")
    symptoms: List[str] = Field(default_factory=list)
    specialty: Optional[str] = Field(None, description="Medical specialty, or null")
    location: Optional[str] = Field(None)
    appointment_time: Optional[str] = Field(None)


class ConversationState(BaseModel):
    entities: Entities
    intent: str = Field(..., description="Short intent code")
    target_entity_type: Literal["doctor", "clinic", "hospital", "unknown"] = "unknown"
    last_user_question: str = ""
    needs_followup: bool = False


_SYSTEM_PROMPT = """\
أنت نظام استخراج حالة من محادثات طبية مصرية. أخرج JSON يطابق الـ Schema المحدد تماماً.

━━━ Schema ━━━
{schema}

━━━ قواعد مهمة ━━━

## 1. Intent Detection

المقصود بكل intent:
- ask_price: أي سؤال عن السعر أو التكلفة أو طريقة الدفع
  ← كلمات: "بكام"، "سعر"، "تكلفة"، "تكلف"، "كام الكشف"
  ← طريقة دفع: "تقسيط"، "بالتقسيط"، "كاش"، "فيزا"، "بالكارت"، "هل فيه تقسيط"
     ⚠️ أي سؤال عن طريقة الدفع = ask_price حتماً
- check_availability: مواعيد وجدول العيادة
  ← كلمات: "مواعيد"، "موعد"، "متاح"، "موجود"، "امتى"، "الجدول"
- list_doctors: طلب قائمة أسماء الأطباء
- book_appointment: طلب حجز موعد
- who_is_present: مين موجود/بيكشف في وقت محدد
- ask_price_and_availability: طلب السعر والمواعيد معاً في نفس السؤال
  ← مثال: "بكام ومين موجود"، "سعر الكشف + مواعيد"
- hospital_info: معلومات عامة عن المستشفى (مش عيادة محددة)
- describe_symptoms: وصف أعراض بدون تخصص واضح
- general_inquiry: استفسار عام

## 2. إذا اجتمع سعر + مواعيد/من موجود في نفس السؤال → intent = ask_price_and_availability

## 3. الأعراض → تخصص (CRITICAL)
إذا الأعراض تحدد تخصصاً واضحاً، ضع الكلينيك والـ intent المناسب:
- "درسي/ضرسي بيوجعني"، "أسناني" → clinic="أسنان", intent=check_availability
- "ايدي/رجلي/ركبتي بتوجعني"، "عندي كسر"، "مفصل" → clinic="عظام", intent=check_availability
- "عيني بتوجعني"، "مش شايف" → clinic="عيون", intent=check_availability
- "طفلي/ابني/رضيع تعبان" → clinic="أطفال", intent=check_availability
- "قلبي بيدق"، "ضغط عالي" → clinic="قلب", intent=check_availability
- "بشرتي فيها حساسية"، "حبوب في وجهي" → clinic="جلدية", intent=check_availability
- "إذني بتوجعني"، "التهاب اللوز" → clinic="أنف وأذن وحنجرة", intent=check_availability
- "محتاج اشعه"، "عايز اعمل رنين"، "سكانر" → clinic="اشعه", intent=check_availability
- "تقويم أسنان"، "حشو" → clinic="أسنان", intent=ask_price (if بكام)
- "بطني بتوجعني" + معدة/قولون → clinic="باطنة", intent=check_availability

## 4. Doctor vs Specialty/Clinic
- "دكتور أطفال" → clinic="أطفال", doctor=null (مش اسم شخص!)
- "دكتور النسا"، "دكتور الباطنة" → clinic="نسا وتوليد"/"باطنة", doctor=null
- "دكتور أحمد" / "دكتور سامي خليل" → doctor="أحمد"/"سامي خليل"
- الألقاب التالية ليست أسماء شخصية: استشاري، أخصائي، مقيم، الاستشاري، رئيس قسم

⚠️ قاعدة مهمة جداً: لا تستخرج doctor إلا إذا سبق الاسم كلمة "دكتور" أو "د." أو "Dr" صراحةً في الجملة.
إذا كتب المستخدم اسمه الشخصي (مثال: "ماجد مجدى عبدالملاك" بدون كلمة دكتور) → doctor=null.
الاسم الشخصي للمستخدم لا يُعامَل كاسم دكتور أبداً.
doctor يجب أن يكون اسمًا شخصيًا (اسم + لقب على الأقل) وليس تخصصاً أو لقباً وظيفياً.

## 5. كلمات ليست أسماء دكاترة (إذا جاءت بعد "دكتور"):
الحضور/الغياب: موجود، متاح، جاى، جايه، بيجي، رايح، ماشي، بيكشف، بيفتح
الوقت: الضهر، الصبح، العصر، النهارده، بكرة، دلوقتي، الساعه
الأيام: السبت، الأحد، الاثنين، الثلاثاء، الأربعاء، الخميس، الجمعة
مثال: "دكتور جاى الضهر" → doctor=null
مثال: "دكتور الباطنة الاستشاري بكام" → clinic="باطنة", doctor=null, intent=ask_price

## 6. Entity Merging
- إذا ما ذُكر كيان جديد والـ intent نفسه → احتفظ بالكيانات القديمة
- إذا تغير الموضوع → امسح الكيانات القديمة المرتبطة بالموضوع القديم

## 7. Target Entity Type
- دكتور + اسم شخصي → "doctor"
- عيادة/تخصص → "clinic"  
- مستشفى → "hospital"
- غير واضح → "unknown"

أجب بـ JSON فقط.
"""


class StateManager:
    def __init__(self, model: str = "gpt-4o"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("LLM_MODEL", model)
        self.schema_json = ConversationState.model_json_schema()

    def extract_state(
        self,
        current_query: str,
        chat_history: List[dict],
        previous_state: Optional[ConversationState] = None,
    ) -> ConversationState:
        if not self.client:
            return self._fallback_state(current_query)

        system_prompt = _SYSTEM_PROMPT.format(schema=self.schema_json)

        history_text = "<conversation>\n"
        for msg in chat_history[-6:]:  # last 6 messages for context
            role = "user" if msg["role"] == "user" else "assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "</conversation>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": history_text},
            {"role": "user", "content": f"الحالة السابقة:\n{previous_state.model_dump_json() if previous_state else 'لا يوجد'}"},
            {"role": "user", "content": f"السؤال الحالي:\n{current_query}"},
        ]

        with tracer.start_as_current_span("state.extract") as span:
            span.set_attribute("state.query", current_query[:200])
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=ConversationState,
                )
                parsed: ConversationState = response.choices[0].message.parsed
                parsed.last_user_question = current_query

                # ── [STABILITY] Doctor name confidence gate ───────────────────
                # Reject extracted doctor names that don't look like real personal
                # names (single token, seniority title, specialty word, time word).
                if parsed.entities.doctor and not _is_valid_doctor_name(parsed.entities.doctor):
                    logger.info(
                        "Doctor name '%s' rejected by confidence gate — clearing",
                        parsed.entities.doctor,
                    )
                    parsed.entities.doctor = None
                    parsed.entities.provider_id = None
                    # If doctor was the sole reason for target_entity_type=doctor, reset
                    if parsed.target_entity_type == "doctor":
                        parsed.target_entity_type = "clinic" if parsed.entities.clinic else "unknown"
                span.set_attribute("state.intent", parsed.intent)
                span.set_attribute("state.entity_type", parsed.target_entity_type)
                span.set_attribute("state.clinic", parsed.entities.clinic or "")
                span.set_attribute("state.doctor", parsed.entities.doctor or "")
                logger.info(
                    "State: intent=%s clinic=%s doctor=%s type=%s",
                    parsed.intent, parsed.entities.clinic,
                    parsed.entities.doctor, parsed.target_entity_type,
                )
                return parsed
            except Exception as exc:
                span.record_exception(exc)
                logger.error("State extraction failed: %s", exc)
                return self._fallback_state(current_query)

    def _fallback_state(self, query: str) -> ConversationState:
        return ConversationState(
            entities=Entities(),
            intent="unknown",
            target_entity_type="unknown",
            last_user_question=query,
            needs_followup=False,
        )

    def merge_states(self, prev: ConversationState, new: ConversationState) -> ConversationState:
        """
        Merge previous and new state.

        [STABILITY] Clearing rules:
        - If the new state extracted NO doctor AND the intent changed (different topic),
          clear prev doctor + provider_id so stale doctor doesn't leak into clinic queries.
        - If new state extracted NO clinic AND intent changed, clear prev clinic + clinic_id.
        - Symptoms always accumulate.
        - Other fields: new value wins if truthy, else prev is kept.
        """
        merged = prev.model_copy(deep=True)
        merged.intent = new.intent
        merged.last_user_question = new.last_user_question
        merged.needs_followup = new.needs_followup

        intent_changed = prev.intent != new.intent

        # Doctor: new value wins; but also clear stale doctor when intent changed and
        # new LLM call found no doctor (meaning user switched to a clinic/general query).
        if new.entities.doctor:
            merged.entities.doctor = new.entities.doctor
            merged.entities.provider_id = new.entities.provider_id  # sync resolved ID
        elif intent_changed:
            # User changed topic and there's no doctor in the new state — clear stale.
            merged.entities.doctor = None
            merged.entities.provider_id = None

        # Clinic: same clearing logic
        if new.entities.clinic:
            merged.entities.clinic = new.entities.clinic
            merged.entities.clinic_id = new.entities.clinic_id
        elif intent_changed:
            merged.entities.clinic = None
            merged.entities.clinic_id = None

        # Other scalar fields: new wins if truthy
        for field in ["hospital", "specialty", "location", "appointment_time"]:
            val = getattr(new.entities, field)
            if val:
                setattr(merged.entities, field, val)

        merged.entities.symptoms = list(set(merged.entities.symptoms + new.entities.symptoms))

        if new.target_entity_type != "unknown":
            merged.target_entity_type = new.target_entity_type
        return merged

    def rewrite_query(self, state: ConversationState) -> str:
        parts = []
        if state.intent == "ask_price":
            parts.append("سعر كشف")
        elif state.intent == "ask_price_and_availability":
            parts.append("سعر كشف ومواعيد")
        elif state.intent == "book_appointment":
            parts.append("حجز موعد")
        elif state.intent == "list_doctors":
            parts.append("أطباء")
        elif state.intent in ("check_availability", "who_is_present"):
            parts.append("مواعيد أطباء")
        elif state.intent == "hospital_info":
            parts.append("معلومات مستشفى")

        if state.entities.doctor and state.intent != "list_doctors":
            parts.append(f"عند الدكتور {state.entities.doctor}")
        elif state.entities.clinic:
            parts.append(f"في عيادة {state.entities.clinic}")
        elif state.entities.hospital:
            parts.append(f"في مستشفى {state.entities.hospital}")

        if state.entities.specialty:
            parts.append(f"تخصص {state.entities.specialty}")

        if not parts and state.entities.symptoms:
            parts.extend(state.entities.symptoms)

        return " ".join(parts) if parts else state.last_user_question


# Global instance
state_manager = StateManager()