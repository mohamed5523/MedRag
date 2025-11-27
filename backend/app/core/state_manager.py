import logging
import os
from typing import List, Literal, Optional

from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.state_manager")

# ------------------------------------------------------------------------------
# State Models
# ------------------------------------------------------------------------------

class Entities(BaseModel):
    doctor: Optional[str] = Field(None, description="Doctor name or null")
    clinic: Optional[str] = Field(None, description="Clinic name or null")
    hospital: Optional[str] = Field(None, description="Hospital name or null")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms mentioned")
    specialty: Optional[str] = Field(None, description="Medical specialty if mentioned or null")
    location: Optional[str] = Field(None, description="Location or null")
    appointment_time: Optional[str] = Field(None, description="Time mentioned or null")

class ConversationState(BaseModel):
    entities: Entities
    intent: str = Field(..., description="Short description of what the user is trying to do")
    target_entity_type: Literal["doctor", "clinic", "hospital", "unknown"] = Field(..., description="The primary entity type being targeted")
    last_user_question: str = Field(..., description="The exact last question from the user")
    needs_followup: bool = Field(..., description="Whether the bot needs to ask a follow-up question to clarify intent")
    
# ------------------------------------------------------------------------------
# State Manager
# ------------------------------------------------------------------------------

class StateManager:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.model = model
        self.schema_json = ConversationState.model_json_schema()

    # --------------------------------------------------------------------------
    #  Extract State
    # --------------------------------------------------------------------------
    def extract_state(
        self,
        current_query: str,
        chat_history: List[dict],
        previous_state: Optional[ConversationState] = None
    ) -> ConversationState:

        if not self.client:
            return self._fallback_state(current_query)

        system_prompt = f"""
You are an expert assistant that extracts structured conversation state 
from Arabic medical conversations. You MUST output valid JSON that matches 
the exact schema provided.

----------------
JSON Schema:
{self.schema_json}
----------------

### IMPORTANT RULES ###

1. **Entity Merging**
   - If the user does NOT explicitly mention a new doctor/clinic/hospital,
     you MUST keep the old one from previous_state.
   - Example: User says "سعر كشفه كام؟" -> doctor remains the one in previous_state.
   - Do NOT overwrite entities with null unless user explicitly resets context.

2. **Symptoms**
   - Append new symptoms to existing ones.
   - Do not remove symptoms unless user states they made a mistake.

3. **Target Entity Type Classification**
   - If text mentions: "دكتور", person name → target = "doctor"
   - If mentions: "عيادة" → target = "clinic"
   - If mentions: "مستشفى" → target = "hospital"
   - If ambiguous → inherit previous target entity type.

4. **Intent Detection Examples**
   - "سعر الكشف", "سعره", "بكام", "التكلفة", "كم السعر" → ask_price
   - "احجز", "ميعاد", "حجز", "موعد", "أحجز" → book_appointment
   - "مين الدكاترة", "أسماء الأطباء", "قائمة", "دكاترة ايه الموجودين" → list_doctors
   - "مين متاح", "موجود؟", "المواعيد", "مواعيد", "جدول العيادة", "متى", "امتى" → check_availability
   - "بطني بتوجعني", "عندي ألم" → describe_symptoms
   - أي سؤال عن مستشفى أو خدماتها العامة → hospital_info
   - Follow natural language meaning.

5. **Clinic vs Hospital Distinction**
   - "عيادة" (clinic) + schedules/doctors → target = "clinic" (use MCP tools)
   - "مستشفى" (hospital) + general info → target = "hospital" (use RAG)
   - Surgery clinic, dental clinic, etc. are CLINICs not hospitals

6. **Ambiguity**
   - If the request lacks essential details → needs_followup = true.

7. **NEVER hallucinate new entities**
   - ONLY use entities found in history, previous_state, or current query.

----------------
Now process the conversation.
"""

        # Format conversation history safely
        history_text = "<conversation>\n"
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "</conversation>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation History:\n{history_text}"},
            {"role": "user", "content": f"Previous State:\n{previous_state.model_dump_json() if previous_state else None}"},
            {"role": "user", "content": f"Current User Input:\n{current_query}"}
        ]

        with tracer.start_as_current_span("state.extract") as span:
            span.set_attribute("state.input.query", current_query[:200])
            span.set_attribute("state.input.history_len", len(chat_history))
            span.set_attribute("state.input.has_previous_state", previous_state is not None)
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=ConversationState
                )
                parsed: ConversationState = response.choices[0].message.parsed
                # Output attributes for Phoenix
                span.set_attribute("state.output.intent", parsed.intent)
                span.set_attribute("state.output.target_entity_type", parsed.target_entity_type)
                span.set_attribute("state.output.entities.doctor", parsed.entities.doctor or "")
                span.set_attribute("state.output.entities.clinic", parsed.entities.clinic or "")
                span.set_attribute("state.output.entities.hospital", parsed.entities.hospital or "")
                span.set_attribute("state.output.entities.specialty", parsed.entities.specialty or "")
                span.set_attribute("state.output.needs_followup", parsed.needs_followup)
                return parsed
            except Exception as e:
                span.record_exception(e)
                logger.error(f"State extraction failed: {e}")
                return self._fallback_state(current_query)

    # --------------------------------------------------------------------------
    # Fallback state
    # --------------------------------------------------------------------------
    def _fallback_state(self, query: str) -> ConversationState:
        return ConversationState(
            entities=Entities(),
            intent="unknown",
            target_entity_type="unknown",
            last_user_question=query,
            needs_followup=False
        )

    # --------------------------------------------------------------------------
    # Merge State (optional helper)
    # --------------------------------------------------------------------------
    def merge_states(self, prev: ConversationState, new: ConversationState) -> ConversationState:
        merged = prev.model_copy(deep=True)

        # Always update intent, last question, followup flag
        merged.intent = new.intent
        merged.last_user_question = new.last_user_question
        merged.needs_followup = new.needs_followup

        # Merge entity fields safely
        for field in ["doctor", "clinic", "hospital", "specialty", "location", "appointment_time"]:
            new_value = getattr(new.entities, field)
            if new_value:  # only overwrite when new value is not null
                setattr(merged.entities, field, new_value)

        # Merge symptoms (append unique)
        merged.entities.symptoms = list(set(
            merged.entities.symptoms + new.entities.symptoms
        ))

        # Target entity type: inherit unless new one is explicit
        if new.target_entity_type != "unknown":
            merged.target_entity_type = new.target_entity_type

        return merged


    # --------------------------------------------------------------------------
    # Hybrid Query Rewriting (Rule-Based + LLM)
    # --------------------------------------------------------------------------
    def rewrite_query(self, state: ConversationState) -> str:
        """
        Rule-based rewriting first. If insufficient, fallback to LLM rewrite.
        """
        rb_query = self._rule_based_rewrite(state)

        if len(rb_query) < 5:  # too weak → use LLM rewrite
            return self._llm_rewrite(state)
        return rb_query

    def _rule_based_rewrite(self, state: ConversationState) -> str:
        parts = []

        if state.intent == "ask_price":
            parts.append("سعر كشف")

        elif state.intent == "book_appointment":
            parts.append("حجز موعد")
            
        elif state.intent == "list_doctors":
            parts.append("أطباء")

        elif state.intent == "check_availability":
            parts.append("مواعيد أطباء")

        elif state.intent == "hospital_info":
            parts.append("معلومات مستشفى")

        # If listing doctors, ignore specific doctor entity to broaden search
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

        return " ".join(parts) if parts else ""

    # --------------------------------------------------------------------------
    # LLM-based query rewrite
    # --------------------------------------------------------------------------
    def _llm_rewrite(self, state: ConversationState) -> str:
        if not self.client:
            return state.last_user_question

        prompt = f"""
Rewrite the user's question into a complete search query using this state:

{state.model_dump_json()}

Rules:
- If doctor/clinic/hospital is known, mention it explicitly in the query.
- Never invent new entities.
- Stay concise and in Arabic.
"""

        messages = [
            {"role": "system", "content": "You rewrite Arabic medical questions into full search queries."},
            {"role": "user", "content": prompt}
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=50
            )
            return resp.choices[0].message["content"]

        except Exception as e:
            logger.error(f"LLM rewrite failed: {e}")
            return state.last_user_question

# Global instance
state_manager = StateManager()
