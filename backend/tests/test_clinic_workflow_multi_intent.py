import asyncio

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import ProviderListPayload, ProviderRecord, ProviderScheduleResponse, ScheduleSlot, ServicePriceResponse
from app.services.clinic_workflow import ClinicWorkflowService


class FakeQAEngine:
    model = "fake-model"

    def build_time_context(self, question, now_dt=None):
        return {"is_arabic": True, "tz_name": "Africa/Cairo", "date_hint": "اليوم", "time_context_message": "stub"}

    async def answer_question(self, *, question, contexts, time_context, chat_history=None, user_gender=None):
        # Return a simple merged answer so the test doesn't depend on LLM behavior.
        return {
            "answer": "ok-multi",
            "sources": [c.metadata.get("source", "mcp") for c in contexts],
            "context_count": len(contexts),
            "model_used": self.model,
            "tokens_used": 0,
        }


class FakeMCPClient:
    def __init__(self):
        self.calls: list[str] = []

    async def get_clinic_provider_list(self):
        self.calls.append("get_clinic_provider_list")
        # Minimal provider list
        return ProviderListPayload(
            providers=[
                ProviderRecord.model_validate(
                    {"clinicid": 1097, "clinicArabicName": "جراحه", "providerid": 10151, "DoctorNameA": "بيمن عادل عزيز بساده"}
                )
            ]
        )

    async def get_clinic_provider_schedule(self, clinic_id: int, date_from: str, date_to: str, provider_id=None):
        self.calls.append("get_clinic_provider_schedule")
        return ProviderScheduleResponse(
            slots=[
                ScheduleSlot(
                    clinic_id=clinic_id,
                    provider_id=provider_id,
                    day_id=getattr(self, "_expected_day_id", 1),
                    day_name="Tuesday",
                    shift_start="20:00",
                    shift_end="22:00",
                )
            ]
        )

    async def get_service_price(self, clinic_id, provider_id=None):
        self.calls.append("get_service_price")
        return ServicePriceResponse.model_validate(
            [{"serviceArabicName": "كشف", "price": 300, "currency": "EGP"}]
        )

    # Multi-intent flow might resolve clinic/doctor:
    async def match_clinic_hybrid(self, **kwargs):
        self.calls.append("match_clinic_hybrid")
        raise AssertionError("This test uses explicit clinic_id; no matching needed.")

    async def match_doctor_hybrid(self, **kwargs):
        self.calls.append("match_doctor_hybrid")
        raise AssertionError("This test is clinic-only; doctor matching should not be required.")


def _decision(intent="check_availability"):
    return RouteDecision(mode=RouteMode.MCP, intent=intent, reason="test", tool_sequence=[], entities_snapshot={})


def test_multi_intent_schedule_and_price_calls_both_tools():
    client = FakeMCPClient()
    workflow = ClinicWorkflowService(mcp_client=client)
    qa = FakeQAEngine()

    state = ConversationState(
        entities=Entities(clinic="جراحه", clinic_id=1097, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مواعيد عيادة الجراحة و سعر الكشف",
        needs_followup=False,
    )

    res = asyncio.run(
        workflow.run(
            decision=_decision("check_availability"),
            state=state,
            question=state.last_user_question,
            qa_engine=qa,
            chat_history=[],
        )
    )
    assert res.qa_response["answer"] == "ok-multi"
    assert "get_clinic_provider_schedule" in client.calls
    assert "get_service_price" in client.calls


def test_multi_intent_list_doctors_schedule_and_price_triggers_all():
    client = FakeMCPClient()
    workflow = ClinicWorkflowService(mcp_client=client)
    qa = FakeQAEngine()

    state = ConversationState(
        entities=Entities(clinic="جراحه", clinic_id=1097, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="عاوز أسماء الدكاترة ومواعيد عيادة الجراحة و سعر الكشف",
        needs_followup=False,
    )

    res = asyncio.run(
        workflow.run(
            decision=_decision("check_availability"),
            state=state,
            question=state.last_user_question,
            qa_engine=qa,
            chat_history=[],
        )
    )
    assert res.qa_response["answer"] == "ok-multi"
    assert "get_clinic_provider_list" in client.calls
    assert "get_clinic_provider_schedule" in client.calls
    assert "get_service_price" in client.calls


