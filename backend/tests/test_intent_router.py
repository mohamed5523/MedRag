from app.core.intent_router import RouteMode, route_conversation
from app.core.state_manager import ConversationState, Entities


def _base_state(**kwargs):
    entities = Entities(**kwargs.get("entities", {}))
    return ConversationState(
        entities=entities,
        intent=kwargs.get("intent", "unknown"),
        target_entity_type=kwargs.get("target_entity_type", "unknown"),
        last_user_question=kwargs.get("last_user_question", "test"),
        needs_followup=False,
    )


def test_mcp_route_for_pricing_intent():
    state = _base_state(
        intent="ask_price",
        entities={"clinic": "عيادة الأطفال"},
        target_entity_type="clinic",
    )

    decision = route_conversation(state, "عايز أعرف سعر كشف العيادة")

    assert decision.mode == RouteMode.MCP
    assert decision.tool_sequence
    assert decision.intent == "ask_price"


def test_hospital_queries_force_rag():
    state = _base_state(
        intent="ask_price",
        entities={"hospital": "مستشفى السلام"},
        target_entity_type="hospital",
    )

    decision = route_conversation(state, "مستشفى السلام فيها أي خدمات؟")

    assert decision.mode == RouteMode.RAG
    assert decision.intent == "hospital_info"

