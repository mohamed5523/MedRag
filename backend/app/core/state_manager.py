"""
state_manager.py — Conversation State Extractor
================================================
Extracts structured state from Egyptian Arabic (Ammya) medical conversations.
Improvements:
- Payment questions (تقسيط/كاش/فيزا) correctly classified as ask_price
- Better triage: symptoms → clinic specialty → MCP intent
- Seniority titles never mistaken for doctor names
"""

import logging
import os
from typing import List, Literal, Optional

from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.state_manager")


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
        merged = prev.model_copy(deep=True)
        merged.intent = new.intent
        merged.last_user_question = new.last_user_question
        merged.needs_followup = new.needs_followup
        for field in ["doctor", "clinic", "hospital", "specialty", "location", "appointment_time"]:
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