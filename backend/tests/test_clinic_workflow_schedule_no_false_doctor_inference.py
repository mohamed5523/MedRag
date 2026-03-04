import asyncio

import pytest

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import ProviderScheduleResponse, ScheduleSlot
from app.services.clinic_workflow import ClinicWorkflowService


class FakeQAEngine:
    model = "fake-model"

    def build_time_context(self, question, now_dt=None):
        return {"is_arabic": True, "tz_name": "Africa/Cairo", "date_hint": "اليوم", "time_context_message": "stub"}

    async def answer_question(self, *, question, contexts, time_context, chat_history=None, user_gender=None):
        return {"answer": "ok", "sources": [], "context_count": len(contexts), "model_used": self.model, "tokens_used": 0}


class FakeMCPClientNoDoctorInference:
    def __init__(self):
        self.match_doctor_calls = 0
        self.schedule_calls = 0

    async def match_clinic_hybrid(self, **kwargs):
        raise AssertionError("match_clinic_hybrid should not be called when clinic_id is already known")

    async def match_doctor_hybrid(self, **kwargs):
        self.match_doctor_calls += 1
        raise AssertionError("match_doctor_hybrid must NOT be called for clinic-only query 'مواعيد جراحه'")

    async def get_clinic_provider_schedule(self, clinic_id: int, date_from: str, date_to: str, provider_id=None):
        self.schedule_calls += 1
        # Return at least one slot so workflow doesn't fail with empty schedule
        return ProviderScheduleResponse(
            slots=[
                ScheduleSlot(
                    clinic_id=clinic_id,
                    provider_id=None,
                    day_id=getattr(self, "_expected_day_id", 1),
                    day_name="Saturday",
                    shift_start="09:00",
                    shift_end="10:00",
                )
            ]
        )

    async def get_clinic_provider_list(self):
        # Return a provider for clinic_id=1097 so _handle_schedule can proceed
        # past the empty-list guard. The real assertion is that match_doctor_hybrid
        # is NOT called for a clinic-only query like "مواعيد جراحه".
        from app.integrations.mcp_client import ProviderListPayload, ProviderRecord
        return ProviderListPayload(providers=[
            ProviderRecord(
                clinic_id=1097,
                clinic_name_ar="جراحه",
                provider_id=9001,
                provider_name_ar="دكتور اختبار",
            )
        ])


def _decision(intent="check_availability"):
    return RouteDecision(mode=RouteMode.MCP, intent=intent, reason="test", tool_sequence=[], entities_snapshot={})


def test_schedule_clinic_only_query_does_not_infer_doctor_from_mo3ayed():
    """
    Regression: `_infer_doctor_from_text` must not treat the trailing 'د' in 'مواعيد'
    as a standalone doctor marker ('د '), which would incorrectly trigger doctor matching.
    """
    fake_client = FakeMCPClientNoDoctorInference()
    workflow = ClinicWorkflowService(mcp_client=fake_client)

    state = ConversationState(
        entities=Entities(clinic="جراحه", clinic_id=1097, doctor=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مواعيد جراحه",
        needs_followup=False,
    )

    res = asyncio.run(
        workflow.run(
            decision=_decision("check_availability"),
            state=state,
            question=state.last_user_question,
            qa_engine=FakeQAEngine(),
            chat_history=[],
        )
    )
    assert res.qa_response["answer"] == "ok"
    assert fake_client.match_doctor_calls == 0
    assert fake_client.schedule_calls >= 1


