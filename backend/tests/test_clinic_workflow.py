import asyncio

import pytest

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import ProviderRecord, ServicePriceResponse
from app.services.clinic_workflow import ClinicWorkflowService, MCPWorkflowError


class FakeQAEngine:
    model = "fake-model"

    def build_time_context(self, question, now_dt=None):
        return {
            "is_arabic": True,
            "tz_name": "Africa/Cairo",
            "date_hint": "اليوم",
            "time_context_message": "stub",
        }

    def answer_question(self, *, question, contexts, time_context, chat_history=None):
        self.last_question = question
        self.last_contexts = contexts
        return {
            "answer": "تم تلخيص بيانات MCP.",
            "sources": ["mcp"],
            "context_count": len(contexts),
            "model_used": self.model,
            "tokens_used": 0,
        }


class FakeMCPClient:
    def __init__(self):
        self.calls = []

    async def lookup_provider_record(self, doctor_name, clinic_name=None):
        self.calls.append("lookup_provider_record")
        return ProviderRecord.model_validate(
            {
                "clinicid": 10,
                "providerid": 77,
                "providerArabicName": "د. خالد",
                "clinicArabicName": "عيادة الباطنة",
                "specialty": "باطنة",
            }
        )

    async def lookup_clinic_record(self, clinic_name):
        self.calls.append("lookup_clinic_record")
        return ProviderRecord.model_validate(
            {
                "clinicid": 10,
                "clinicArabicName": clinic_name,
            }
        )

    async def get_service_price(self, clinic_id, provider_id=None):
        self.calls.append("get_service_price")
        return ServicePriceResponse.model_validate(
            [
                {
                    "serviceArabicName": "كشف باطنة",
                    "price": 300,
                    "currency": "EGP",
                }
            ]
        )


def _conversation_state(intent="ask_price", clinic="عيادة الباطنة", doctor="دكتور خالد"):
    return ConversationState(
        entities=Entities(
            clinic=clinic,
            doctor=doctor,
            hospital=None,
            symptoms=[],
            specialty="باطنة",
        ),
        intent=intent,
        target_entity_type="clinic",
        last_user_question="test",
        needs_followup=False,
    )


def _decision(intent="ask_price"):
    return RouteDecision(
        mode=RouteMode.MCP,
        intent=intent,
        reason="test",
        tool_sequence=[],
        entities_snapshot={},
    )


def test_clinic_workflow_pricing_path_uses_mcp_data():
    fake_client = FakeMCPClient()
    workflow = ClinicWorkflowService(mcp_client=fake_client)
    fake_qa = FakeQAEngine()

    result = asyncio.run(
        workflow.run(
            decision=_decision("ask_price"),
            state=_conversation_state(),
            question="عايز أعرف سعر الكشف",
            qa_engine=fake_qa,
            chat_history=[],
        )
    )

    assert result.qa_response["answer"] == "تم تلخيص بيانات MCP."
    assert "get_service_price" in fake_client.calls
    assert fake_qa.last_contexts
    assert "الخدمات" in fake_qa.last_contexts[0].page_content


def test_clinic_workflow_requires_clinic_information():
    workflow = ClinicWorkflowService(mcp_client=FakeMCPClient())
    fake_qa = FakeQAEngine()
    state = ConversationState(
        entities=Entities(),
        intent="ask_price",
        target_entity_type="unknown",
        last_user_question="سعر الكشف كام؟",
        needs_followup=False,
    )

    with pytest.raises(MCPWorkflowError) as exc:
        asyncio.run(
            workflow.run(
                decision=_decision("ask_price"),
                state=state,
                question="سعر الكشف كام؟",
                qa_engine=fake_qa,
                chat_history=[],
            )
        )

    assert exc.value.reason == "missing_clinic"

