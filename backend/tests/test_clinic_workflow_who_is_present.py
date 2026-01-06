import asyncio
from datetime import datetime

import pytest

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import (
    ClinicMatchResponse,
    ClinicMatchResult,
    HybridMatchStatus,
    ProviderListPayload,
    ProviderRecord,
    ProviderScheduleResponse,
    ScheduleSlot,
    ServicePriceResponse,
)
from app.services.clinic_workflow import MCPWorkflowError
from app.services.clinic_workflow import ClinicWorkflowService


class FakeQAEngine:
    model = "fake-model"

    def __init__(self, now_dt: datetime):
        self._now_dt = now_dt

    def build_time_context(self, question, now_dt=None):
        return {"is_arabic": True, "tz_name": "Africa/Cairo", "date_hint": "stub", "time_context_message": "stub", "now_dt": self._now_dt}

    async def answer_question(self, *, question, contexts, time_context, chat_history=None):
        return {
            "answer": "ok-who",
            "sources": [c.metadata.get("source", "mcp") for c in contexts],
            "context_count": len(contexts),
            "model_used": self.model,
            "tokens_used": 0,
        }


class FakeMCPClient:
    def __init__(self, *, has_schedule_for_provider_ids: set[int], expected_day_id: int):
        self.calls: list[tuple[str, dict]] = []
        self._has_schedule_for_provider_ids = has_schedule_for_provider_ids
        self._expected_day_id = expected_day_id

    async def get_clinic_provider_list(self):
        self.calls.append(("get_clinic_provider_list", {}))
        return ProviderListPayload(
            providers=[
                ProviderRecord.model_validate(
                    {"clinicid": 2001, "clinicArabicName": "الاسنان", "providerid": 11, "DoctorNameA": "دكتور اسنان ١"}
                ),
                ProviderRecord.model_validate(
                    {"clinicid": 2001, "clinicArabicName": "الاسنان", "providerid": 22, "DoctorNameA": "دكتور اسنان ٢"}
                ),
            ]
        )

    async def get_clinic_provider_schedule(self, clinic_id, provider_id=None, day_id=None):
        self.calls.append(("get_clinic_provider_schedule", {"clinic_id": clinic_id, "provider_id": provider_id, "day_id": day_id}))
        assert day_id == self._expected_day_id
        if provider_id in self._has_schedule_for_provider_ids:
            return ProviderScheduleResponse(
                slots=[
                    ScheduleSlot(
                        clinic_id=clinic_id,
                        provider_id=provider_id,
                        day_id=day_id,
                        day_name="Tuesday",
                        shift_start="13:00",
                        shift_end="15:00",
                    )
                ]
            )
        return ProviderScheduleResponse(slots=[])

    async def get_service_price(self, clinic_id, provider_id=None):
        self.calls.append(("get_service_price", {"clinic_id": clinic_id, "provider_id": provider_id}))
        return ServicePriceResponse.model_validate(
            [{"serviceArabicName": "كشف", "price": 100, "currency": "EGP"}]
        )

    async def match_clinic_hybrid(self, **kwargs):
        raise AssertionError("Test uses explicit clinic_id; no matching should be needed.")

    async def match_doctor_hybrid(self, **kwargs):
        raise AssertionError("No doctor matching should be needed for who_is_present clinic query.")


def _decision(intent="check_availability"):
    return RouteDecision(mode=RouteMode.MCP, intent=intent, reason="test", tool_sequence=[], entities_snapshot={})


def test_who_is_present_today_joins_provider_list_with_schedule_for_single_day_only():
    # Tuesday 2026-01-06 → day_id should be 4 (Saturday=1)
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))
    client = FakeMCPClient(has_schedule_for_provider_ids={11}, expected_day_id=4)
    workflow = ClinicWorkflowService(mcp_client=client)

    state = ConversationState(
        entities=Entities(clinic="الاسنان", clinic_id=2001, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين موجود النهارده فى عيادة الاسنان؟",
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

    assert res.qa_response["answer"] == "ok-who"
    # Must include a dedicated presence context (not just schedule alone)
    assert "mcp.who_is_present" in res.qa_response["sources"]
    # Ensure we did not fetch all 7 days; one call per provider for the requested day only.
    schedule_calls = [c for c in client.calls if c[0] == "get_clinic_provider_schedule"]
    assert len(schedule_calls) == 2


def test_who_is_present_tomorrow_uses_tomorrow_day_id():
    # Tuesday 2026-01-06 with "بكرة" → Wednesday day_id = 5
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))
    client = FakeMCPClient(has_schedule_for_provider_ids={22}, expected_day_id=5)
    workflow = ClinicWorkflowService(mcp_client=client)

    state = ConversationState(
        entities=Entities(clinic="الاسنان", clinic_id=2001, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين موجود بكرة فى عيادة الاسنان؟",
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

    assert res.qa_response["answer"] == "ok-who"
    assert "mcp.who_is_present" in res.qa_response["sources"]


def test_who_is_present_phrase_variant_min_elly_mogod_triggers_handler():
    # "مين اللي موجود" should behave like "مين موجود"
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))
    client = FakeMCPClient(has_schedule_for_provider_ids={11}, expected_day_id=4)
    workflow = ClinicWorkflowService(mcp_client=client)

    state = ConversationState(
        entities=Entities(clinic="الاسنان", clinic_id=2001, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين اللي موجود النهارده فى عيادة الاسنان؟",
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
    assert res.qa_response["answer"] == "ok-who"
    assert "mcp.who_is_present" in res.qa_response["sources"]


def test_who_is_present_explicit_arabic_weekday_uses_mapping():
    # Query specifies a weekday explicitly; should map to day_id without relying on "today".
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))  # Tuesday
    client = FakeMCPClient(has_schedule_for_provider_ids={11}, expected_day_id=6)  # Thursday day_id=6
    workflow = ClinicWorkflowService(mcp_client=client)

    state = ConversationState(
        entities=Entities(clinic="الاسنان", clinic_id=2001, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين موجود الخميس فى عيادة الاسنان؟",
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
    assert res.qa_response["answer"] == "ok-who"
    assert "mcp.who_is_present" in res.qa_response["sources"]


def test_who_is_present_ghadan_uses_tomorrow():
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))  # Tuesday
    client = FakeMCPClient(has_schedule_for_provider_ids={22}, expected_day_id=5)  # Wednesday
    workflow = ClinicWorkflowService(mcp_client=client)

    state = ConversationState(
        entities=Entities(clinic="الاسنان", clinic_id=2001, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين موجود غدا فى عيادة الاسنان؟",
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
    assert res.qa_response["answer"] == "ok-who"
    assert "mcp.who_is_present" in res.qa_response["sources"]


def test_who_is_present_plus_price_runs_both_contexts():
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))
    client = FakeMCPClient(has_schedule_for_provider_ids={11}, expected_day_id=4)
    workflow = ClinicWorkflowService(mcp_client=client)

    state = ConversationState(
        entities=Entities(clinic="الاسنان", clinic_id=2001, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين موجود النهارده فى عيادة الاسنان و سعر الكشف كام؟",
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
    assert res.qa_response["answer"] == "ok-who"
    assert "mcp.who_is_present" in res.qa_response["sources"]
    assert "mcp.service_price" in res.qa_response["sources"]
    assert any(c[0] == "get_service_price" for c in client.calls)


class FakeMCPClientClinicAmbiguous:
    async def match_clinic_hybrid(self, **kwargs):
        return ClinicMatchResponse(
            status=HybridMatchStatus.AMBIGUOUS_NEED_MORE_INFO,
            message="ambiguous",
            query_tokens=["جراحه"],
            best_match=None,
            candidates=[
                ClinicMatchResult(clinic_id="1", clinic_name="جراحه", score=0.8, token_overlap=0.8, fuzzy_name_score=0.9, order_score=0.8, matched_tokens=["جراحه"]),
                ClinicMatchResult(clinic_id="2", clinic_name="جراحه تجميل", score=0.75, token_overlap=0.7, fuzzy_name_score=0.85, order_score=0.75, matched_tokens=["جراحه"]),
            ],
        )

    async def get_clinic_provider_list(self):
        raise AssertionError("Should not list providers before clinic is resolved.")

    async def get_clinic_provider_schedule(self, clinic_id, provider_id=None, day_id=None):
        raise AssertionError("Should not fetch schedule before clinic is resolved.")

    async def get_service_price(self, clinic_id, provider_id=None):
        raise AssertionError("Not needed")

    async def match_doctor_hybrid(self, **kwargs):
        raise AssertionError("Not needed")


def test_who_is_present_ambiguous_clinic_bubbles_up_as_clinic_ambiguous():
    qa = FakeQAEngine(now_dt=datetime(2026, 1, 6, 12, 0, 0))
    workflow = ClinicWorkflowService(mcp_client=FakeMCPClientClinicAmbiguous())

    state = ConversationState(
        entities=Entities(clinic="الجراحة", clinic_id=None, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مين موجود النهارده فى عيادة الجراحة؟",
        needs_followup=False,
    )

    with pytest.raises(MCPWorkflowError) as exc:
        asyncio.run(
            workflow.run(
                decision=_decision("check_availability"),
                state=state,
                question=state.last_user_question,
                qa_engine=qa,
                chat_history=[],
            )
        )
    assert exc.value.reason == "clinic_ambiguous"


