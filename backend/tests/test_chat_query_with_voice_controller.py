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
        async def run(self, *, decision, state, question, qa_engine, chat_history=None):
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

