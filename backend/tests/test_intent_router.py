"""
Unit tests for app.core.intent_router

Covers:
- MCP intent routing (ask_price, check_availability, book_appointment, list_doctors)
- RAG hospital info routing
- Default RAG fallback
- Rule-based fallback when no API key
- LLM routing with mocked OpenAI response
"""
import os
from unittest.mock import MagicMock, patch

import pytest


def _make_state(
    intent: str = "",
    target_entity_type: str = "unknown",
    doctor: str = None,
    clinic: str = None,
    hospital: str = None,
    specialty: str = None,
    clinic_id: int = None,
    provider_id: int = None,
):
    """Build a minimal ConversationState-like object for testing."""
    # BUG FIX: state_manager uses 'Entities' not 'ExtractedEntities'; also
    # ConversationState requires needs_followup field (was missing).
    from app.core.state_manager import ConversationState, Entities
    entities = Entities(
        doctor=doctor,
        clinic=clinic,
        hospital=hospital,
        specialty=specialty,
        clinic_id=clinic_id,
        provider_id=provider_id,
    )
    state = ConversationState(
        intent=intent,
        target_entity_type=target_entity_type,
        entities=entities,
        last_user_question="test query",
        needs_followup=False,
    )
    return state


class TestRuleBasedFallback:
    """Tests for LLMRouter._rule_based_fallback."""

    def setup_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            from app.core.intent_router import LLMRouter
            self.router = LLMRouter()
            assert self.router.client is None  # No API key → rule-based only

    def test_hospital_state_routes_to_rag(self):
        """State with hospital entity should route to RAG."""
        state = _make_state(hospital="مستشفى نيل بدر")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.RAG

    def test_hospital_target_routes_to_rag(self):
        """State with target_entity_type=hospital should route to RAG."""
        state = _make_state(target_entity_type="hospital")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.RAG

    def test_ask_price_routes_to_mcp(self):
        """ask_price intent should route to MCP."""
        state = _make_state(intent="ask_price")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.MCP
        assert decision.intent == "ask_price"

    def test_check_availability_routes_to_mcp(self):
        """check_availability intent should route to MCP."""
        state = _make_state(intent="check_availability")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.MCP

    def test_book_appointment_routes_to_mcp(self):
        """book_appointment intent should route to MCP."""
        state = _make_state(intent="book_appointment")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.MCP

    def test_list_doctors_routes_to_mcp(self):
        """list_doctors intent should route to MCP."""
        state = _make_state(intent="list_doctors")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.MCP

    def test_clinic_entity_routes_to_mcp(self):
        """State with clinic entity but no specific MCP intent should default to MCP."""
        state = _make_state(clinic="عيادة الأسنان")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.MCP

    def test_doctor_entity_routes_to_mcp(self):
        """State with doctor entity should route to MCP."""
        state = _make_state(doctor="دكتور أحمد")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.MCP

    def test_general_inquiry_defaults_to_rag(self):
        """State with no specific entity should default to RAG."""
        state = _make_state(intent="general_inquiry")
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.RAG

    def test_empty_state_defaults_to_rag(self):
        """Completely empty state should fall back to RAG."""
        state = _make_state()
        decision = self.router._rule_based_fallback(state)
        from app.core.intent_router import RouteMode
        assert decision.mode == RouteMode.RAG


class TestRouteConversation:
    """Tests for the top-level route_conversation function."""

    def test_hospital_guard_bypasses_llm(self):
        """Hospital state should route to RAG without calling LLM."""
        from app.core.intent_router import route_conversation, RouteMode
        state = _make_state(hospital="مستشفى")
        decision = route_conversation(state, "معلومات عن المستشفى")
        assert decision.mode == RouteMode.RAG
        assert decision.intent == "hospital_info"

    def test_mcp_intent_guard_bypasses_llm(self):
        """MCP intent from state extraction should bypass LLM routing."""
        from app.core.intent_router import route_conversation, RouteMode
        state = _make_state(intent="ask_price")
        decision = route_conversation(state, "بكم الكشف؟")
        assert decision.mode == RouteMode.MCP
        assert decision.intent == "ask_price"

    def test_mcp_decision_has_tool_sequence(self):
        """MCP routing for ask_price should include tool_sequence."""
        from app.core.intent_router import route_conversation, RouteMode
        state = _make_state(intent="ask_price")
        decision = route_conversation(state, "بكم سعر الكشف؟")
        assert decision.mode == RouteMode.MCP
        assert len(decision.tool_sequence) > 0

    def test_no_user_query_uses_state_question(self):
        """If user_query is None, state.last_user_question should be used."""
        from app.core.intent_router import route_conversation
        state = _make_state(intent="ask_price")
        # Should not raise even without explicit query
        decision = route_conversation(state, user_query=None)
        assert decision is not None

    def test_route_decision_has_required_fields(self):
        """RouteDecision must have mode, intent, reason, entities_snapshot."""
        from app.core.intent_router import route_conversation
        state = _make_state(intent="list_doctors", clinic="عيادة الجراحة")
        decision = route_conversation(state, "مين الدكاترة؟")
        assert hasattr(decision, "mode")
        assert hasattr(decision, "intent")
        assert hasattr(decision, "reason")
        assert hasattr(decision, "entities_snapshot")
        assert isinstance(decision.entities_snapshot, dict)


class TestLLMRouterWithMock:
    """Tests for LLMRouter.decide_route with mocked OpenAI response."""

    def test_decide_route_uses_llm_structured_output(self):
        """decide_route should parse LLM structured output into LLMRoutingDecision."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"}):
            from app.core.intent_router import LLMRouter, RouteMode

            router = LLMRouter()

            # Mock the OpenAI beta parse response
            mock_parsed = MagicMock()
            mock_parsed.mode = RouteMode.MCP
            mock_parsed.intent = "ask_price"
            mock_parsed.confidence = 0.95
            mock_parsed.reasoning = "سعر"

            mock_choice = MagicMock()
            mock_choice.message.parsed = mock_parsed

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            router.client = MagicMock()
            router.client.beta.chat.completions.parse.return_value = mock_response

            state = _make_state(intent="ask_price")
            result = router.decide_route(state, "بكم الكشف؟")

            assert result.mode == RouteMode.MCP
            assert result.intent == "ask_price"
            assert result.confidence == 0.95

    def test_decide_route_falls_back_on_exception(self):
        """If LLM call raises, should fall back to rule-based routing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"}):
            from app.core.intent_router import LLMRouter

            router = LLMRouter()
            router.client = MagicMock()
            router.client.beta.chat.completions.parse.side_effect = RuntimeError("LLM down")

            state = _make_state(intent="ask_price")
            result = router.decide_route(state, "بكم الكشف؟")
            # Should still return a valid decision via fallback
            assert result is not None
            assert result.mode is not None
