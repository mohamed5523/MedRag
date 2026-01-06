import asyncio

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import DoctorMatchResult, HybridMatchResponse, HybridMatchStatus
from app.services.clinic_workflow import ClinicWorkflowService


class FakeQAEngine:
    model = "fake-model"

    def build_time_context(self, question, now_dt=None):
        return {"is_arabic": True, "tz_name": "Africa/Cairo", "date_hint": "اليوم", "time_context_message": "stub"}

    async def answer_question(self, *, question, contexts, time_context, chat_history=None):
        return {"answer": "ok", "sources": [], "context_count": len(contexts), "model_used": self.model, "tokens_used": 0}


class FakeMCPClient:
    def __init__(self):
        self.match_clinic_calls = 0
        self.match_doctor_calls = 0

    async def match_clinic_hybrid(self, **kwargs):
        self.match_clinic_calls += 1
        raise AssertionError("match_clinic_hybrid must NOT be called when doctor is explicitly present and clinic is not.")

    async def match_doctor_hybrid(self, *, query: str, clinic_id=None, clinic_name=None, **kwargs):
        self.match_doctor_calls += 1
        # Return an unambiguous doctor match with some clinic id/name.
        return HybridMatchResponse(
            status=HybridMatchStatus.UNAMBIGUOUS_MATCH,
            message="ok",
            query_tokens=[],
            best_match=DoctorMatchResult(
                provider_id="10151",
                clinic_id="1097",
                clinic_name="جراحه",
                name_ar="بيمن عادل عزيز بساده",
                name_en="",
                score=0.9,
                token_overlap=1.0,
                fuzzy_name_score=0.9,
                position_score=0.9,
                matched_by_first_name=True,
                matched_tokens=[],
            ),
            candidates=[],
        )

    async def get_clinic_provider_schedule(self, *args, **kwargs):
        # Not used in this test (we only care about resolution stage)
        raise AssertionError("schedule should not be called")


def _decision(intent="ask_price"):
    return RouteDecision(mode=RouteMode.MCP, intent=intent, reason="test", tool_sequence=[], entities_snapshot={})


def test_doctor_query_does_not_infer_clinic_from_specialty():
    client = FakeMCPClient()
    workflow = ClinicWorkflowService(mcp_client=client)

    # Specialty is set (from previous turns), but user is now explicitly asking about a doctor.
    state = ConversationState(
        entities=Entities(doctor="بيمن عادل عزيز", clinic=None, clinic_id=None, specialty="جراحه"),
        intent="check_availability",
        target_entity_type="doctor",
        last_user_question="مواعيد دكتور بيمن عادل عزيز و سعر كشفه",
        needs_followup=False,
    )

    # Call the internal resolver via the public schedule handler path
    # (the key assertion is that match_clinic_hybrid is never called).
    asyncio.run(workflow._resolve_entities(state, [], state.last_user_question))
    assert client.match_doctor_calls >= 1
    assert client.match_clinic_calls == 0


