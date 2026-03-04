import asyncio

import pytest

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.models.schemas import ChatRequest
from app.services.clinic_workflow import MCPWorkflowError


pytest.importorskip("fastapi")


class _Msg:
    def __init__(self, role: str, content: str, timestamp: float = 0.0):
        self.role = role
        self.content = content
        self.timestamp = timestamp


class FakeMemory:
    def __init__(self):
        self._messages = []
        self._state = None
        self._pending = None

    def add_message(self, session_id, role, content, timestamp=None):
        self._messages.append(_Msg(role, content, 0.0))

    def get_messages(self, session_id, limit=5):
        return self._messages[-limit:]

    def get_formatted_context(self, session_id, last_n=5):
        return ""

    def get_state(self, session_id):
        return self._state

    def save_state(self, session_id, state):
        self._state = state

    def save_pending_action(self, session_id, payload):
        self._pending = payload

    def get_pending_action(self, session_id):
        return self._pending

    def clear_pending_action(self, session_id):
        self._pending = None


class FakeSessionManager:
    def get_or_create(self, x_session_id=None):
        return "sess"

    # Some endpoints (or future refactors) may create a fresh session id when no header is provided.
    # Keep this mock compatible with SessionManager's public API.
    def create_session(self, session_id=None):
        return "sess"


class FakeStateManager:
    def extract_state(self, current_query, chat_history, previous_state=None):
        return ConversationState(
            entities=Entities(doctor="بيمن", clinic=None, hospital=None, symptoms=[], specialty=None),
            intent="ask_price",
            target_entity_type="doctor",
            last_user_question=current_query,
            needs_followup=False,
        )

    def rewrite_query(self, state):
        return state.last_user_question


def test_query_with_voice_stores_pending_action_on_ambiguous_provider(monkeypatch):
    from app.api import chat as chat_api

    fake_mem = FakeMemory()
    monkeypatch.setattr(chat_api, "short_term_memory", fake_mem, raising=True)
    monkeypatch.setattr(chat_api, "session_manager", FakeSessionManager(), raising=True)
    monkeypatch.setattr(chat_api, "state_manager", FakeStateManager(), raising=True)
    monkeypatch.setattr(chat_api.qa_engine, "is_available", lambda: True, raising=False)

    def fake_route(state, user_query=None):
        return RouteDecision(mode=RouteMode.MCP, intent="ask_price", reason="test", tool_sequence=[], entities_snapshot={})

    monkeypatch.setattr(chat_api, "route_conversation", fake_route, raising=True)

    class FakeWorkflow:
        async def run(self, **kwargs):
            raise MCPWorkflowError(
                "يوجد أكثر من دكتور بنفس الاسم.",
                reason="provider_ambiguous",
                data={
                    "candidates": [
                        {"name_ar": "بيمن عادل عزيز بساده", "provider_id": 1, "clinic_id": 10, "clinic_name": "عيادة الباطنة"},
                        {"name_ar": "بيمن عادل عزيز بساده 2", "provider_id": 2, "clinic_id": 10, "clinic_name": "عيادة الباطنة"},
                    ]
                },
            )

    monkeypatch.setattr(chat_api, "clinic_workflow", FakeWorkflow(), raising=True)

    resp = asyncio.run(chat_api.query_with_voice_response(ChatRequest(query="سعر كشف دكتور بيمن"), x_session_id=None))

    assert "اختار رقم" in resp.answer
    pending = fake_mem.get_pending_action("sess")
    assert pending is not None
    assert pending["type"] == "provider_disambiguation"
    assert pending["intent"] == "ask_price"
    assert len(pending["candidates"]) == 2


def test_query_with_voice_symptom_triage_forces_list_doctors(monkeypatch):
    from app.api import chat as chat_api

    fake_mem = FakeMemory()
    monkeypatch.setattr(chat_api, "short_term_memory", fake_mem, raising=True)
    monkeypatch.setattr(chat_api, "session_manager", FakeSessionManager(), raising=True)
    monkeypatch.setattr(chat_api, "state_manager", FakeStateManager(), raising=True)
    monkeypatch.setattr(chat_api.qa_engine, "is_available", lambda: True, raising=False)

    captured = {"intent": None, "specialty": None}

    def fake_route(state, user_query=None):
        captured["intent"] = state.intent
        captured["specialty"] = state.entities.specialty
        return RouteDecision(mode=RouteMode.MCP, intent=state.intent, reason="test", tool_sequence=[], entities_snapshot={})

    monkeypatch.setattr(chat_api, "route_conversation", fake_route, raising=True)

    class FakeWorkflow:
        async def run(self, *, decision, state, question, qa_engine, chat_history=None, user_gender=None):
            class _Res:
                qa_response = {
                    "answer": "دكاترة الباطنة: دكتور أ، دكتور ب",
                    "sources": ["mcp.provider_list"],
                    "context_count": 1,
                    "model_used": "fake",
                    "tokens_used": 0,
                }
                tool_audit = []
            return _Res()

    monkeypatch.setattr(chat_api, "clinic_workflow", FakeWorkflow(), raising=True)

    resp = asyncio.run(
        chat_api.query_with_voice_response(ChatRequest(query="بطني بتوجعني النهاردة اروح لمين؟"), x_session_id=None)
    )

    assert "دكاترة" in resp.answer
    assert captured["intent"] == "list_doctors"
    assert captured["specialty"] == "باطنة"


def test_query_with_voice_can_abandon_pending_action_when_user_changes_request(monkeypatch):
    """
    Regression test: user should be able to override a pending disambiguation prompt by
    providing a fresh request (e.g., "أنا أقصد دكتور ...") instead of being stuck in the turn.
    """
    from app.api import chat as chat_api

    fake_mem = FakeMemory()
    # Seed a pending provider/clinic mismatch action (the scenario that commonly loops)
    fake_mem.save_pending_action(
        "sess",
        {
            "type": "provider_clinic_mismatch",
            "intent": "check_availability",
            "turns_remaining": 2,
            "original_question": "مواعيد دكتور بيمن عادل عيادة نسا وتوليد",
            "requested_clinic": "نسا وتوليد",
            "candidates": [
                {
                    "name_ar": "بيمن عادل عزيز بساده",
                    "provider_id": 10151,
                    "clinic_id": 1097,
                    "clinic_name": "جراحه",
                }
            ],
        },
    )

    monkeypatch.setattr(chat_api, "short_term_memory", fake_mem, raising=True)
    monkeypatch.setattr(chat_api, "session_manager", FakeSessionManager(), raising=True)

    class _StateMgr(FakeStateManager):
        def extract_state(self, current_query, chat_history, previous_state=None):
            return ConversationState(
                entities=Entities(doctor="فادي فوزي", clinic=None, hospital=None, symptoms=[], specialty=None),
                intent="check_availability",
                target_entity_type="doctor",
                last_user_question=current_query,
                needs_followup=False,
            )

    monkeypatch.setattr(chat_api, "state_manager", _StateMgr(), raising=True)
    monkeypatch.setattr(chat_api.qa_engine, "is_available", lambda: True, raising=False)

    def fake_route(state, user_query=None):
        return RouteDecision(mode=RouteMode.MCP, intent="check_availability", reason="test", tool_sequence=[], entities_snapshot={})

    monkeypatch.setattr(chat_api, "route_conversation", fake_route, raising=True)

    class FakeWorkflow:
        async def run(self, **kwargs):
            class _Res:
                qa_response = {"answer": "ok-new", "sources": [], "context_count": 0, "model_used": "fake", "tokens_used": 0}
                tool_audit = []
            return _Res()

    monkeypatch.setattr(chat_api, "clinic_workflow", FakeWorkflow(), raising=True)

    resp = asyncio.run(
        chat_api.query_with_voice_response(ChatRequest(query="اه انت صح، انا كنت اقصد دكتور فادي فوزي"), x_session_id=None)
    )

    assert resp.answer == "ok-new"
    assert fake_mem.get_pending_action("sess") is None


def test_query_with_voice_does_not_abandon_pending_action_on_numeric_selection(monkeypatch):
    from app.api import chat as chat_api

    fake_mem = FakeMemory()
    fake_mem.save_pending_action(
        "sess",
        {
            "type": "provider_disambiguation",
            "intent": "ask_price",
            "turns_remaining": 2,
            "original_question": "سعر كشف دكتور بيمن",
            "candidates": [
                {"name_ar": "بيمن عادل عزيز بساده", "provider_id": 1, "clinic_id": 10, "clinic_name": "عيادة الباطنة"},
                {"name_ar": "بيمن عادل عزيز بساده 2", "provider_id": 2, "clinic_id": 10, "clinic_name": "عيادة الباطنة"},
            ],
        },
    )

    monkeypatch.setattr(chat_api, "short_term_memory", fake_mem, raising=True)
    monkeypatch.setattr(chat_api, "session_manager", FakeSessionManager(), raising=True)
    monkeypatch.setattr(chat_api, "state_manager", FakeStateManager(), raising=True)
    monkeypatch.setattr(chat_api.qa_engine, "is_available", lambda: True, raising=False)

    # If the user sends "2" it should resolve selection (not abandon)
    class FakeWorkflow:
        async def run(self, **kwargs):
            class _Res:
                qa_response = {"answer": "ok-resolved", "sources": [], "context_count": 0, "model_used": "fake", "tokens_used": 0}
                tool_audit = []
            return _Res()

    monkeypatch.setattr(chat_api, "clinic_workflow", FakeWorkflow(), raising=True)
    monkeypatch.setattr(chat_api, "route_conversation", lambda state, user_query=None: RouteDecision(mode=RouteMode.MCP, intent="ask_price", reason="t", tool_sequence=[], entities_snapshot={}), raising=True)

    resp = asyncio.run(chat_api.query_with_voice_response(ChatRequest(query="2"), x_session_id=None))
    assert resp.answer == "ok-resolved"
    assert fake_mem.get_pending_action("sess") is None


def test_query_with_voice_clinic_disambiguation_selection_uses_selected_clinic_id(monkeypatch):
    """
    Regression test: clinic disambiguation must become specific after user selects a number.
    Otherwise the system can get stuck re-asking the same clinic list ("جراحه" family).
    """
    from app.api import chat as chat_api

    fake_mem = FakeMemory()
    fake_mem.save_pending_action(
        "sess",
        {
            "type": "clinic_disambiguation",
            "intent": "check_availability",
            "turns_remaining": 2,
            "original_question": "مين موجود النهارده فى عيادة الجراحة؟",
            "candidates": [
                {"clinic_id": 1097, "clinic_name": "جراحه", "score": 0.9},
                {"clinic_id": 1200, "clinic_name": "جراحه تجميل", "score": 0.89},
            ],
        },
    )

    monkeypatch.setattr(chat_api, "short_term_memory", fake_mem, raising=True)
    monkeypatch.setattr(chat_api, "session_manager", FakeSessionManager(), raising=True)
    monkeypatch.setattr(chat_api.qa_engine, "is_available", lambda: True, raising=False)

    # Minimal state extractor; selection should override clinic+clinic_id regardless.
    class _StateMgr(FakeStateManager):
        def extract_state(self, current_query, chat_history, previous_state=None):
            return ConversationState(
                entities=Entities(doctor=None, clinic=None, hospital=None, symptoms=[], specialty=None),
                intent="check_availability",
                target_entity_type="clinic",
                last_user_question=current_query,
                needs_followup=False,
            )

    monkeypatch.setattr(chat_api, "state_manager", _StateMgr(), raising=True)

    monkeypatch.setattr(
        chat_api,
        "route_conversation",
        lambda state, user_query=None: RouteDecision(
            mode=RouteMode.MCP,
            intent="check_availability",
            reason="test",
            tool_sequence=[],
            entities_snapshot={},
        ),
        raising=True,
    )

    class FakeWorkflow:
        async def run(self, *, state, **kwargs):
            # The key assertion: clinic_id is fixed from selected candidate.
            assert getattr(state.entities, "clinic_id", None) == 1097
            # Clinic selection should not leave a stale doctor hanging around (prevents provider_not_found loops).
            assert getattr(state.entities, "doctor", None) in (None, "")
            class _Res:
                qa_response = {"answer": "ok-clinic", "sources": [], "context_count": 0, "model_used": "fake", "tokens_used": 0}
                tool_audit = []
            return _Res()

    monkeypatch.setattr(chat_api, "clinic_workflow", FakeWorkflow(), raising=True)

    # Use Arabic-Indic digit selection
    resp = asyncio.run(chat_api.query_with_voice_response(ChatRequest(query="١"), x_session_id=None))
    assert resp.answer == "ok-clinic"
    assert fake_mem.get_pending_action("sess") is None


def test_query_with_voice_explicit_doctor_clears_stale_clinic_context(monkeypatch):
    """
    Regression: user can change topic after resolving a clinic. If the new query explicitly
    mentions a doctor (and does not explicitly mention a clinic), we must clear the stale
    clinic/clinic_id so we don't get stuck re-asking clinic disambiguation.
    """
    from app.api import chat as chat_api

    fake_mem = FakeMemory()
    fake_mem.save_state(
        "sess",
        ConversationState(
            entities=Entities(clinic="جراحه", clinic_id=1097, doctor=None, provider_id=None),
            intent="check_availability",
            target_entity_type="clinic",
            last_user_question="مين موجود النهارده فى عيادة الجراحة؟",
            needs_followup=False,
        ),
    )

    monkeypatch.setattr(chat_api, "short_term_memory", fake_mem, raising=True)
    monkeypatch.setattr(chat_api, "session_manager", FakeSessionManager(), raising=True)
    monkeypatch.setattr(chat_api.qa_engine, "is_available", lambda: True, raising=False)

    # Simulate the state extractor "merging" and keeping the old clinic even though the user changed topic.
    class _StickyClinicStateMgr:
        def extract_state(self, current_query, chat_history, previous_state=None):
            return ConversationState(
                entities=Entities(doctor="بيمن عادل عزيز", clinic=previous_state.entities.clinic if previous_state else "جراحه", clinic_id=None),
                intent="ask_price",
                target_entity_type="doctor",
                last_user_question=current_query,
                needs_followup=False,
            )

        def rewrite_query(self, state):
            return state.last_user_question

    monkeypatch.setattr(chat_api, "state_manager", _StickyClinicStateMgr(), raising=True)
    monkeypatch.setattr(
        chat_api,
        "route_conversation",
        lambda state, user_query=None: RouteDecision(mode=RouteMode.MCP, intent="ask_price", reason="t", tool_sequence=[], entities_snapshot={}),
        raising=True,
    )

    class FakeWorkflow:
        async def run(self, *, state, **kwargs):
            assert state.entities.doctor
            assert getattr(state.entities, "clinic", None) in (None, "")
            assert getattr(state.entities, "clinic_id", None) is None
            class _Res:
                qa_response = {"answer": "ok-doctor", "sources": [], "context_count": 0, "model_used": "fake", "tokens_used": 0}
                tool_audit = []
            return _Res()

    monkeypatch.setattr(chat_api, "clinic_workflow", FakeWorkflow(), raising=True)

    resp = asyncio.run(
        chat_api.query_with_voice_response(
            ChatRequest(query="مواعيد عيادة دكتور بيمن عادل عزيز و سعر كشفه"),
            x_session_id=None,
        )
    )
    assert resp.answer == "ok-doctor"

