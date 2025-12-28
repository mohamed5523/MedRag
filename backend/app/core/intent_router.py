from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, List, Optional

from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from app.core.state_manager import ConversationState

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.intent_router")


class RouteMode(str, Enum):
    MCP = "mcp"
    RAG = "rag"


class ToolInvocationPlan(BaseModel):
    """Describe a single MCP tool call that may be executed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    requires_provider_resolution: bool = False
    requires_clinic_resolution: bool = False


class RouteDecision(BaseModel):
    """Structured description of how a query should be routed."""

    model_config = ConfigDict(extra="ignore")

    mode: RouteMode
    intent: str
    reason: str
    tool_sequence: List[ToolInvocationPlan] = Field(default_factory=list)
    entities_snapshot: dict[str, Any] = Field(default_factory=dict)
    fallback_reason: Optional[str] = None

    @property
    def uses_mcp(self) -> bool:
        return self.mode == RouteMode.MCP

    @property
    def requires_provider_resolution(self) -> bool:
        return any(tool.requires_provider_resolution for tool in self.tool_sequence)


# LLM-based routing decision schema
class LLMRoutingDecision(BaseModel):
    """Schema for LLM routing decisions."""
    
    mode: RouteMode = Field(
        ..., 
        description="Whether to use MCP (clinic operations) or RAG (hospital knowledge base)"
    )
    intent: str = Field(
        ...,
        description="The detected intent: ask_price, book_appointment, check_availability, list_doctors, hospital_info, describe_symptoms, general_inquiry, unknown"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation in Arabic for the routing decision"
    )


MCP_INTENTS = {
    "ask_price",
    "book_appointment",
    "check_availability",
    "list_doctors",
}

HOSPITAL_INTENTS = {
    "hospital_info",
}

TOOL_PLAN = {
    "ask_price": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Resolve clinic & provider identifiers from free text.",
            requires_provider_resolution=True,
            requires_clinic_resolution=True,
        ),
        ToolInvocationPlan(
            name="get_service_price",
            description="Fetch the latest pricing info for the resolved clinic/provider.",
        ),
    ],
    "check_availability": [
        ToolInvocationPlan(
            name="get_clinic_provider_list",
            description="Locate provider and clinic IDs to query availability.",
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
}


class LLMRouter:
    """LLM-based intelligent router for MCP vs RAG decisions."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. Falling back to rule-based routing.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.model = model
    
    def decide_route(
        self, 
        state: ConversationState, 
        user_query: str
    ) -> LLMRoutingDecision:
        """Use LLM to decide routing with structured output."""
        
        if not self.client:
            # Fallback to rule-based
            return self._rule_based_fallback(state)
        
        system_prompt = """
أنت نظام توجيه ذكي في تطبيق طبي. مهمتك هي تحديد ما إذا كان السؤال يتعلق بـ:

## MCP (نظام العيادات):
استخدم MCP للأسئلة عن:
- **أسعار الكشف والخدمات**: "كام سعر الكشف؟", "بكم الجلسة؟", "التكلفة"
- **مواعيد العيادات**: "مواعيد العيادة؟", "متى العيادة؟", "جدول المواعيد", "امتى الدكتور موجود"
- **حجز المواعيد**: "احجز موعد", "عايز ميعاد", "أحجز كشف"
- **قائمة الأطباء**: "مين الدكاترة؟", "أطباء العيادة", "دكاترة الأسنان"
- **توافر الأطباء**: "الدكتور موجود؟", "مين متاح؟"

**ملحوظة مهمة**: كلمة "عيادة" تعني MCP دائماً (عيادة الأسنان، عيادة الجراحة، عيادة الباطنة، إلخ)

## RAG (قاعدة معرفة المستشفى):
استخدم RAG للأسئلة عن:
- **معلومات عامة عن المستشفى**: "خدمات المستشفى", "أقسام المستشفى"
- **أمراض ونصائح طبية عامة**: "أعراض السكري", "علاج الضغط"
- **معلومات عن التخصصات الطبية**: "ما هي الجراحة العامة؟"
- **نصائح وإرشادات**: "نصائح للحوامل", "إزاي أحافظ على صحتي"

### القواعد الذهبية:
1. إذا ذُكرت كلمة "عيادة" (بأي تخصص) → استخدم MCP
2. إذا ذُكر اسم دكتور محدد → استخدم MCP
3. إذا السؤال عن سعر أو موعد → استخدم MCP
4. إذا السؤال عن معلومات طبية عامة أو خدمات المستشفى → استخدم RAG
5. في حالة الشك، فضّل MCP إذا كان هناك إشارة لطبيب أو عيادة أو موعد

### الـ Intents المتاحة:
- ask_price: سؤال عن الأسعار
- check_availability: سؤال عن المواعيد والتوافر
- book_appointment: طلب حجز موعد
- list_doctors: طلب قائمة الأطباء
- hospital_info: معلومات عامة عن المستشفى
- describe_symptoms: وصف أعراض
- general_inquiry: استفسار عام
- unknown: غير واضح

حلل السؤال وأعطِ قرارك بثقة عالية.
"""

        user_content = f"""
**السؤال الحالي من المستخدم:**
{user_query}

**السياق المستخرج من المحادثة:**
- Intent: {state.intent}
- Target Entity: {state.target_entity_type}
- Doctor: {state.entities.doctor or "لم يُذكر"}
- Clinic: {state.entities.clinic or "لم يُذكر"}
- Hospital: {state.entities.hospital or "لم يُذكر"}
- Specialty: {state.entities.specialty or "لم يُذكر"}

حدد المسار الصحيح (MCP أو RAG) والـ Intent بناءً على السؤال والسياق.
"""

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=LLMRoutingDecision,
                temperature=0.0,  # Deterministic routing for consistent behavior
            )
            
            decision = response.choices[0].message.parsed
            logger.info(
                f"LLM routing decision: mode={decision.mode}, "
                f"intent={decision.intent}, confidence={decision.confidence:.2f}"
            )
            return decision
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}. Falling back to rules.")
            return self._rule_based_fallback(state)
    
    def _rule_based_fallback(self, state: ConversationState) -> LLMRoutingDecision:
        """Fallback to simple rule-based routing."""
        
        # Hospital context
        if state.target_entity_type == "hospital" or state.entities.hospital:
            return LLMRoutingDecision(
                mode=RouteMode.RAG,
                intent="hospital_info",
                confidence=0.8,
                reasoning="سؤال عن المستشفى - استخدام RAG"
            )
        
        # MCP intents
        if state.intent in MCP_INTENTS:
            return LLMRoutingDecision(
                mode=RouteMode.MCP,
                intent=state.intent,
                confidence=0.7,
                reasoning="سؤال عن عيادة أو دكتور - استخدام MCP"
            )
        
        # Clinic mentioned
        if state.entities.clinic or state.target_entity_type == "clinic":
            return LLMRoutingDecision(
                mode=RouteMode.MCP,
                intent="check_availability",
                confidence=0.6,
                reasoning="ذُكرت عيادة - استخدام MCP"
            )
        
        # Doctor mentioned
        if state.entities.doctor or state.target_entity_type == "doctor":
            return LLMRoutingDecision(
                mode=RouteMode.MCP,
                intent="check_availability",
                confidence=0.6,
                reasoning="ذُكر دكتور - استخدام MCP"
            )
        
        # Default to RAG
        return LLMRoutingDecision(
            mode=RouteMode.RAG,
            intent=state.intent or "general_inquiry",
            confidence=0.5,
            reasoning="استفسار عام - استخدام RAG"
        )


# Global router instance
_router_instance: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    """Get or create the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


def route_conversation(
    state: ConversationState, 
    user_query: Optional[str] = None
) -> RouteDecision:
    """Determine whether to use MCP tools or the existing RAG flow using LLM."""
    
    if not user_query:
        user_query = state.last_user_question
    
    # Ensure we have a valid query string for the LLM router
    if not user_query:
        # Fallback: construct a query from state if nothing is provided
        user_query = f"استفسار عن {state.intent or 'معلومات عامة'}"
        logger.warning(f"No user query provided, using fallback: {user_query}")
    
    with tracer.start_as_current_span("routing.decide") as span:
        span.set_attribute("routing.state_intent", state.intent)
        span.set_attribute("routing.target_entity", state.target_entity_type)
        span.set_attribute("routing.query", user_query[:100])
        
        # Deterministic guards (rules-first) for stability:
        # - Hospital context should always use RAG
        # - If extracted state intent is an MCP intent, preserve it (avoid intent drift)
        if state.target_entity_type == "hospital" or state.entities.hospital:
            llm_decision = LLMRoutingDecision(
                mode=RouteMode.RAG,
                intent="hospital_info",
                confidence=1.0,
                reasoning="سؤال عن المستشفى - استخدام RAG",
            )
            span.set_attribute("routing.decision_source", "guard.hospital")
        elif state.intent in MCP_INTENTS:
            llm_decision = LLMRoutingDecision(
                mode=RouteMode.MCP,
                intent=state.intent,
                confidence=1.0,
                reasoning="النية مستخرجة كاستعلام عيادات - الحفاظ على MCP",
            )
            span.set_attribute("routing.decision_source", "guard.mcp_intent")
        else:
            router = get_router()
            llm_decision = router.decide_route(state, user_query)
            span.set_attribute("routing.decision_source", "llm")
            
        span.set_attribute("routing.llm_mode", llm_decision.mode.value)
        span.set_attribute("routing.llm_intent", llm_decision.intent)
        span.set_attribute("routing.llm_confidence", llm_decision.confidence)
        span.set_attribute("routing.llm_reasoning", llm_decision.reasoning)
        
        # Build tool sequence if MCP mode
        tools: List[ToolInvocationPlan] = []
        if llm_decision.mode == RouteMode.MCP:
            # Use the LLM's intent to get the tool plan
            normalized_intent = llm_decision.intent
            if normalized_intent in TOOL_PLAN:
                tools = [tool.model_copy(deep=True) for tool in TOOL_PLAN[normalized_intent]]
            elif normalized_intent == "list_doctors":
                tools = TOOL_PLAN["list_doctors"]
            else:
                # Default MCP tool plan for unknown MCP intents
                tools = [
                    ToolInvocationPlan(
                        name="get_clinic_provider_list",
                        description="Get provider information",
                        requires_clinic_resolution=True,
                    )
                ]
            
            span.set_attribute("routing.tools", [tool.name for tool in tools])
        
        # Create the final routing decision
        decision = RouteDecision(
            mode=llm_decision.mode,
            intent=llm_decision.intent,
            reason=f"{llm_decision.reasoning} (confidence: {llm_decision.confidence:.0%})",
            tool_sequence=tools,
            entities_snapshot=state.entities.model_dump(),
        )
        
        logger.info(
            f"Final route: {decision.mode.value} | "
            f"Intent: {decision.intent} | "
            f"Confidence: {llm_decision.confidence:.2f}"
        )
        
        return decision

