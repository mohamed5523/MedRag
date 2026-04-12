"""
intent_router.py — Fully LLM-Driven Intent Router
===================================================
Replaces the old hybrid rule/LLM router with a single, powerful LLM call
that understands any Egyptian Arabic (Ammya) question and decides:

1. mode  → MCP (clinic data) or RAG (hospital knowledge)
2. intent → what the user actually wants
3. entities → any clinic/doctor/date/specialty mentioned

Design principles:
- ONE LLM call does ALL the work (no brittle regex chains)
- Deterministic guards only for true emergencies (life-threatening)
- Every MCP intent maps to a concrete tool plan
- Graceful fallback if LLM fails
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Any, List, Optional

from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from app.core.state_manager import ConversationState

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.intent_router")


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Models
# ─────────────────────────────────────────────────────────────────────────────

class RouteMode(str, Enum):
    MCP = "mcp"
    RAG = "rag"


class ToolInvocationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    requires_provider_resolution: bool = False
    requires_clinic_resolution: bool = False


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mode: RouteMode
    intent: str
    reason: str
    tool_sequence: List[ToolInvocationPlan] = Field(default_factory=list)
    entities_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Enriched fields the LLM extracts directly
    enriched_clinic: Optional[str] = None
    enriched_doctor: Optional[str] = None
    enriched_date_hint: Optional[str] = None   # "today" | "tomorrow" | "saturday" | None
    all_intents: List[str] = Field(default_factory=list)  # multi-intent support

    @property
    def uses_mcp(self) -> bool:
        return self.mode == RouteMode.MCP

    @property
    def requires_provider_resolution(self) -> bool:
        return any(t.requires_provider_resolution for t in self.tool_sequence)


class LLMRoutingDecision(BaseModel):
    """Full structured output from the routing LLM."""
    mode: RouteMode
    primary_intent: str = Field(
        description=(
            "ask_price | check_availability | list_doctors | book_appointment | "
            "who_is_present | ask_price_and_availability | hospital_info | "
            "describe_symptoms | general_inquiry | unknown"
        )
    )
    all_intents: List[str] = Field(
        default_factory=list,
        description="All intents detected, ordered by priority (primary first)"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    # Entity extraction (saves an extra LLM call)
    clinic_name: Optional[str] = Field(None, description="Canonical clinic/specialty name in Arabic, or null")
    doctor_name: Optional[str] = Field(None, description="Doctor's personal name only (not specialty), or null")
    date_hint: Optional[str] = Field(
        None,
        description="'today' | 'tomorrow' | 'saturday' | 'sunday' | etc. — or null if not mentioned"
    )
    payment_question: bool = Field(
        False,
        description="True if user is asking about payment method (cash/installments/visa)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MCP Intent → Tool Plan mapping
# ─────────────────────────────────────────────────────────────────────────────

MCP_INTENTS = {
    "ask_price",
    "check_availability",
    "list_doctors",
    "book_appointment",
    "who_is_present",
    "ask_price_and_availability",
}

TOOL_PLAN: dict[str, List[ToolInvocationPlan]] = {
    "ask_price": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Resolve clinic & provider identifiers.",
            requires_provider_resolution=True,
            requires_clinic_resolution=True,
        ),
        ToolInvocationPlan(
            name="get_service_price",
            description="Fetch pricing for the resolved clinic/provider.",
        ),
    ],
    "check_availability": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Locate provider and clinic IDs.",
            requires_provider_resolution=True,
            requires_clinic_resolution=True,
        ),
        ToolInvocationPlan(
            name="get_clinic_provider_schedule",
            description="Retrieve the official clinic schedule.",
        ),
    ],
    "book_appointment": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Locate provider and clinic IDs prior to booking.",
            requires_provider_resolution=True,
            requires_clinic_resolution=True,
        ),
        ToolInvocationPlan(
            name="get_clinic_provider_schedule",
            description="Retrieve slots to recommend booking times.",
        ),
    ],
    "list_doctors": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="List available doctors for the requested clinic or specialty.",
            requires_clinic_resolution=True,
        ),
    ],
    "who_is_present": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Get all doctors for the clinic.",
            requires_clinic_resolution=True,
        ),
        ToolInvocationPlan(
            name="get_clinic_provider_schedule",
            description="Check who is present/scheduled right now or on a given day.",
        ),
    ],
    "ask_price_and_availability": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Resolve clinic & provider identifiers.",
            requires_provider_resolution=True,
            requires_clinic_resolution=True,
        ),
        ToolInvocationPlan(
            name="get_service_price",
            description="Fetch pricing for the resolved clinic/provider.",
        ),
        ToolInvocationPlan(
            name="get_clinic_provider_schedule",
            description="Retrieve schedule to show alongside prices.",
        ),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Emergency guard — only hard-coded pattern (life-threatening, no pricing)
# ─────────────────────────────────────────────────────────────────────────────

_CRITICAL_EMERGENCY_PATTERNS = [
    r"مش\s+قادر?\s+(?:يتنفس|اتنفس|أتنفس)",
    r"(?:فاقد|فاقده)\s+(?:الوعي|وعيه|وعيها)",
    r"(?:نزيف|بينزف|ينزف)\s+(?:شديد|كتير|جامد)",
    r"(?:تشنج|تشنجات)",
    r"(?:مش\s+بيتكلم|مش\s+بيحس|مش\s+صاحي)",
    r"(?:ابن|طفل|ولد|بنت|رضيع).{0,35}(?:بلع|ابتلع|ابلع)",
    r"(?:بلع|ابتلع|ابلع).{0,25}(?:عملة|معدنية|حاجة|جسم|بطارية|مسمار|دبوس)",
]


def _is_critical_life_emergency(text: str) -> bool:
    """Only true emergencies where no MCP data is relevant."""
    q = (text or "").casefold()
    return any(re.search(p, q) for p in _CRITICAL_EMERGENCY_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Router
# ─────────────────────────────────────────────────────────────────────────────

_ROUTING_SYSTEM_PROMPT = """\
أنت نظام توجيه ذكي لتطبيق مستشفى مصري. مهمتك تحليل سؤال المريض وتحديد:
1. هل يحتاج بيانات من نظام العيادات (MCP) أم من قاعدة معرفة المستشفى (RAG)
2. ما النية الحقيقية وراء السؤال
3. أي كيانات مذكورة (عيادة، دكتور، تاريخ)

━━━ متى تستخدم MCP (نظام العيادات) ━━━
استخدم MCP لأي سؤال عن:
• أسعار الكشف أو أي خدمة طبية: "بكام"، "سعر"، "تكلفة"، "الكشف كام"
• طريقة الدفع: "تقسيط"، "كاش"، "فيزا"، "بالكارت" → هذا سؤال سعر
• مواعيد وجدول العيادات: "امتى"، "موعد"، "مواعيد"، "الجدول"
• مين موجود أو متاح: "مين موجود"، "مين بيكشف"، "مين النهارده"
• أسماء الأطباء: "مين الدكاترة"، "دكاترة الأسنان"
• حجز مواعيد: "احجز"، "عايز موعد"
• أعراض تحدد تخصص بوضوح: "درسي بيوجعني" → أسنان، "ايدي بتوجعني" → عظام أو أشعة

━━━ متى تستخدم RAG (قاعدة المعرفة) ━━━
• معلومات عامة عن المستشفى وخدماتها
• نصائح طبية وأمراض عامة
• حالات طوارئ حرجة تحتاج توجيه عاجل

━━━ الـ Intents المتاحة ━━━
- ask_price: سؤال عن الأسعار أو طريقة الدفع
- check_availability: سؤال عن المواعيد والجدول
- list_doctors: طلب قائمة الأطباء أو اسماءهم
- book_appointment: طلب حجز موعد
- who_is_present: مين موجود/بيكشف الآن أو في يوم محدد
- ask_price_and_availability: سؤال مدمج عن السعر والمواعيد في نفس الوقت ← مهم جداً
- hospital_info: معلومات عامة عن المستشفى
- describe_symptoms: وصف أعراض بدون تخصص واضح
- general_inquiry: استفسار عام
- unknown: مش واضح

━━━ قواعد ذهبية ━━━
1. السؤال المدمج "بكام + مين موجود" أو "سعر + مواعيد" → intent = ask_price_and_availability
2. "تقسيط/كاش/فيزا/بالكارت" → سؤال سعر → ask_price أو ask_price_and_availability
3. اسم الدكتور = الاسم الشخصي فقط (مش التخصص): "دكتور أطفال" → clinic="أطفال", doctor=null
4. "مين موجود بكرة/النهارده/الضهر" → who_is_present مع date_hint
5. اسم العيادة: استخدم الاسم العربي الشائع (أسنان، عظام، نسا وتوليد، إلخ)
6. إذا كان السؤال عن يوم محدد → ضع اليوم في date_hint
7. في all_intents: ضع كل النوايا المكتشفة بالترتيب من الأهم

━━━ أمثلة مهمة ━━━
"دكتور النسا بيكشف بكام ومين موجود بكرة الضهر؟"
→ mode=mcp, primary_intent=ask_price_and_availability, clinic_name="نسا وتوليد", date_hint="tomorrow"

"هو دكتور العظام بيكشف بكام النهارده ومين موجود دلوقتي؟"
→ mode=mcp, primary_intent=ask_price_and_availability, clinic_name="عظام", date_hint="today"

"تركيب تقويم الستان المعدني بكام وهل فيه تقسيط ولا الدفع كاش بس؟"
→ mode=mcp, primary_intent=ask_price, clinic_name="أسنان", payment_question=true

"مواعيد دكاترة الجراحة الأسبوع ده"
→ mode=mcp, primary_intent=check_availability, clinic_name="جراحة"

"درسي بيوجعني محتاج دكتور"
→ mode=mcp, primary_intent=check_availability, clinic_name="أسنان"

"عايز اعمل رنين على ركبتي بكام"
→ mode=mcp, primary_intent=ask_price, clinic_name="اشعه"

أجب بـ JSON فقط بدون أي كلام إضافي.
"""


class LLMRouter:
    """Single LLM call handles routing + entity extraction for any Arabic question."""

    def __init__(self, model: str = "gpt-4o"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", model)
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found — using rule-based fallback.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def decide_route(
        self,
        state: ConversationState,
        user_query: str,
    ) -> LLMRoutingDecision:
        """Full LLM routing + entity extraction in one call."""

        if not self.client:
            return self._rule_based_fallback(state, user_query)

        user_content = f"""السؤال: {user_query}

السياق المستخرج:
- Intent السابق: {state.intent or "غير محدد"}
- نوع الكيان: {state.target_entity_type}
- دكتور: {state.entities.doctor or "لم يُذكر"}
- عيادة: {state.entities.clinic or "لم تُذكر"}
- مستشفى: {state.entities.hospital or "لم يُذكر"}
- تخصص: {state.entities.specialty or "لم يُذكر"}"""

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=LLMRoutingDecision,
                temperature=0.0,
            )
            decision = response.choices[0].message.parsed
            logger.info(
                "LLM routing: mode=%s intent=%s clinic=%s doctor=%s date=%s confidence=%.2f",
                decision.mode, decision.primary_intent,
                decision.clinic_name, decision.doctor_name,
                decision.date_hint, decision.confidence,
            )
            return decision
        except Exception as exc:
            logger.error("LLM routing failed: %s — using fallback", exc)
            return self._rule_based_fallback(state, user_query)

    def _rule_based_fallback(self, state: ConversationState, query: str = "") -> LLMRoutingDecision:
        """Simple fallback when LLM is unavailable."""
        q = (query or "").casefold()

        # Hospital
        if state.target_entity_type == "hospital" or state.entities.hospital:
            return LLMRoutingDecision(
                mode=RouteMode.RAG, primary_intent="hospital_info",
                all_intents=["hospital_info"], confidence=0.9,
                reasoning="سؤال عن المستشفى"
            )

        # Price + availability combined
        has_price = any(kw in q for kw in ["بكام", "سعر", "تكلفة", "تكلف", "تقسيط", "كاش", "الدفع"])
        has_avail = any(kw in q for kw in ["موجود", "متاح", "مواعيد", "موعد", "النهارده", "بكرة", "دلوقتي"])
        has_who = any(kw in q for kw in ["مين موجود", "مين بيكشف", "مين متاح"])

        if has_price and (has_avail or has_who):
            return LLMRoutingDecision(
                mode=RouteMode.MCP, primary_intent="ask_price_and_availability",
                all_intents=["ask_price_and_availability"],
                confidence=0.85, reasoning="سعر + مواعيد",
                clinic_name=state.entities.clinic,
            )

        if has_price:
            return LLMRoutingDecision(
                mode=RouteMode.MCP, primary_intent="ask_price",
                all_intents=["ask_price"], confidence=0.85,
                reasoning="سؤال سعر", clinic_name=state.entities.clinic,
            )

        if has_who:
            return LLMRoutingDecision(
                mode=RouteMode.MCP, primary_intent="who_is_present",
                all_intents=["who_is_present"], confidence=0.8,
                reasoning="مين موجود", clinic_name=state.entities.clinic,
            )

        if state.intent in MCP_INTENTS:
            return LLMRoutingDecision(
                mode=RouteMode.MCP, primary_intent=state.intent,
                all_intents=[state.intent], confidence=0.75,
                reasoning="نية MCP من الحالة",
                clinic_name=state.entities.clinic,
                doctor_name=state.entities.doctor,
            )

        if state.entities.clinic or state.target_entity_type == "clinic":
            return LLMRoutingDecision(
                mode=RouteMode.MCP, primary_intent="check_availability",
                all_intents=["check_availability"], confidence=0.7,
                reasoning="ذُكرت عيادة",
                clinic_name=state.entities.clinic,
            )

        return LLMRoutingDecision(
            mode=RouteMode.RAG, primary_intent="general_inquiry",
            all_intents=["general_inquiry"], confidence=0.5,
            reasoning="استفسار عام"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global router singleton
# ─────────────────────────────────────────────────────────────────────────────

_router_instance: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


# ─────────────────────────────────────────────────────────────────────────────
# Main routing entry point
# ─────────────────────────────────────────────────────────────────────────────

def route_conversation(
    state: ConversationState,
    user_query: Optional[str] = None,
) -> RouteDecision:
    """
    Determine routing for a user query.

    Flow:
    1. Critical life-emergency guard → RAG immediately
    2. LLM decides everything else in one smart call
    3. Build tool plan from intent
    4. Patch state with any newly extracted entities
    """
    if not user_query:
        user_query = state.last_user_question or ""

    if not user_query:
        user_query = f"استفسار عن {state.intent or 'معلومات عامة'}"

    with tracer.start_as_current_span("routing.decide") as span:
        span.set_attribute("routing.query", user_query[:200])
        span.set_attribute("routing.state_intent", state.intent or "")

        # ── Guard: true life-threatening emergency ────────────────────────────
        if _is_critical_life_emergency(user_query):
            logger.info("Critical emergency detected — routing to RAG")
            span.set_attribute("routing.decision_source", "guard.critical_emergency")
            llm_decision = LLMRoutingDecision(
                mode=RouteMode.RAG,
                primary_intent="describe_symptoms",
                all_intents=["describe_symptoms"],
                confidence=1.0,
                reasoning="حالة طوارئ حرجة — RAG للتوجيه الفوري",
            )
        else:
            # ── Full LLM routing (handles everything else) ────────────────────
            router = get_router()
            llm_decision = router.decide_route(state, user_query)
            span.set_attribute("routing.decision_source", "llm")

        span.set_attribute("routing.mode", llm_decision.mode.value)
        span.set_attribute("routing.intent", llm_decision.primary_intent)
        span.set_attribute("routing.confidence", llm_decision.confidence)

        # ── Patch state with entities the LLM extracted ───────────────────────
        _patch_state_from_llm(state, llm_decision)

        # ── Build tool plan ───────────────────────────────────────────────────
        tools: List[ToolInvocationPlan] = []
        if llm_decision.mode == RouteMode.MCP:
            intent = llm_decision.primary_intent
            tools = [t.model_copy(deep=True) for t in TOOL_PLAN.get(intent, TOOL_PLAN.get("check_availability", []))]
            if not tools:
                tools = [ToolInvocationPlan(
                    name="get_clinic_provider_list",
                    description="Get provider information",
                    requires_clinic_resolution=True,
                )]

        decision = RouteDecision(
            mode=llm_decision.mode,
            intent=llm_decision.primary_intent,
            reason=f"{llm_decision.reasoning} (confidence: {llm_decision.confidence:.0%})",
            tool_sequence=tools,
            entities_snapshot=state.entities.model_dump(),
            enriched_clinic=llm_decision.clinic_name,
            enriched_doctor=llm_decision.doctor_name,
            enriched_date_hint=llm_decision.date_hint,
            all_intents=llm_decision.all_intents or [llm_decision.primary_intent],
        )

        logger.info(
            "Route: %s | Intent: %s | Clinic: %s | Doctor: %s | Date: %s | Intents: %s",
            decision.mode.value, decision.intent,
            decision.enriched_clinic, decision.enriched_doctor,
            decision.enriched_date_hint, decision.all_intents,
        )

        return decision


def _patch_state_from_llm(state: ConversationState, decision: LLMRoutingDecision) -> None:
    """
    Apply LLM-extracted entities back to state so clinic_workflow can use them
    without needing to re-extract from text.
    """
    try:
        if decision.clinic_name and not state.entities.clinic:
            state.entities.clinic = decision.clinic_name
            state.target_entity_type = "clinic"

        if decision.doctor_name and not state.entities.doctor:
            state.entities.doctor = decision.doctor_name
            state.target_entity_type = "doctor"

        if decision.primary_intent in MCP_INTENTS and state.intent not in MCP_INTENTS:
            state.intent = decision.primary_intent
    except Exception as exc:
        logger.debug("State patch failed (non-critical): %s", exc)