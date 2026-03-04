import asyncio

import pytest

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import (
    ClinicMatchResponse,
    ClinicMatchResult,
    DoctorMatchResult,
    HybridMatchResponse,
    HybridMatchStatus,
    ServicePriceResponse,
)
from app.services.clinic_workflow import ClinicWorkflowService, MCPWorkflowError


class FakeQAEngine:
    model = "fake-model"

    def build_time_context(self, question, now_dt=None):
        return {"is_arabic": True, "tz_name": "Africa/Cairo", "date_hint": "اليوم", "time_context_message": "stub"}

    async def answer_question(self, *, question, contexts, time_context, chat_history=None, user_gender=None):
        return {"answer": "ok", "sources": [], "context_count": len(contexts), "model_used": self.model, "tokens_used": 0}


class FakeMCPClientClinicResolution:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def match_clinic_hybrid(self, *, query: str, top_k: int = 5, min_score: float = 0.65):
        self.calls.append(("match_clinic_hybrid", {"query": query}))
        return ClinicMatchResponse(
            status=HybridMatchStatus.UNAMBIGUOUS_MATCH,
            message="ok",
            query_tokens=[],
            best_match=ClinicMatchResult(
                clinic_id="10",
                clinic_name="نسا وتوليد",
                score=0.9,
                token_overlap=1.0,
                fuzzy_name_score=0.9,
                order_score=0.9,
                matched_tokens=[],
            ),
            candidates=[],
        )

    async def match_doctor_hybrid(self, *, query: str, clinic_id: str | None = None, clinic_name=None, **kwargs):
        self.calls.append(("match_doctor_hybrid", {"query": query, "clinic_id": clinic_id, "clinic_name": clinic_name}))
        assert clinic_id == "10"
        assert clinic_name is None
        return HybridMatchResponse(
            status=HybridMatchStatus.UNAMBIGUOUS_MATCH,
            message="ok",
            query_tokens=[],
            best_match=DoctorMatchResult(
                provider_id="77",
                clinic_id="10",
                clinic_name="نسا وتوليد",
                name_ar=query,
                name_en="",
                score=0.9,
                token_overlap=1.0,
                fuzzy_name_score=0.9,
                position_score=0.9,
                matched_by_first_name=False,
                matched_tokens=[],
            ),
            candidates=[],
        )

    async def get_service_price(self, clinic_id, provider_id=None):
        self.calls.append(("get_service_price", {"clinic_id": clinic_id, "provider_id": provider_id}))
        assert clinic_id == 10
        return ServicePriceResponse.model_validate(
            [{"serviceArabicName": "كشف", "price": 300, "currency": "EGP"}]
        )


def _decision(intent="ask_price"):
    return RouteDecision(mode=RouteMode.MCP, intent=intent, reason="test", tool_sequence=[], entities_snapshot={})


def test_workflow_resolves_clinic_id_before_doctor_matching():
    workflow = ClinicWorkflowService(mcp_client=FakeMCPClientClinicResolution())
    state = ConversationState(
        entities=Entities(doctor="بيمن عدل", clinic="النساؤ وتوليد", hospital=None, symptoms=[], specialty=None),
        intent="ask_price",
        target_entity_type="doctor",
        last_user_question="سعر كشف بيمن عدل عيادة النساؤ وتوليد",
        needs_followup=False,
    )

    res = asyncio.run(
        workflow.run(
            decision=_decision("ask_price"),
            state=state,
            question=state.last_user_question,
            qa_engine=FakeQAEngine(),
            chat_history=[],
        )
    )
    assert res.qa_response["answer"] == "ok"


def test_workflow_clinic_ambiguous_raises_error():
    class _AmbiguousClient(FakeMCPClientClinicResolution):
        async def match_clinic_hybrid(self, *, query: str, top_k: int = 5, min_score: float = 0.65):
            return ClinicMatchResponse(
                status=HybridMatchStatus.AMBIGUOUS_NEED_MORE_INFO,
                message="يوجد أكثر من عيادة.",
                query_tokens=[],
                best_match=None,
                candidates=[
                    ClinicMatchResult(
                        clinic_id="10",
                        clinic_name="عيادة نسا وتوليد",
                        score=0.7,
                        token_overlap=0.7,
                        fuzzy_name_score=0.7,
                        order_score=0.7,
                        matched_tokens=[],
                    ),
                    ClinicMatchResult(
                        clinic_id="11",
                        clinic_name="عيادة نسا",
                        score=0.69,
                        token_overlap=0.69,
                        fuzzy_name_score=0.69,
                        order_score=0.69,
                        matched_tokens=[],
                    ),
                ],
            )

    workflow = ClinicWorkflowService(mcp_client=_AmbiguousClient())
    state = ConversationState(
        entities=Entities(doctor="بيمن", clinic="نسا", hospital=None, symptoms=[], specialty=None),
        intent="ask_price",
        target_entity_type="doctor",
        last_user_question="سعر كشف دكتور بيمن في عيادة نسا",
        needs_followup=False,
    )

    with pytest.raises(MCPWorkflowError) as exc:
        asyncio.run(
            workflow.run(
                decision=_decision("ask_price"),
                state=state,
                question=state.last_user_question,
                qa_engine=FakeQAEngine(),
                chat_history=[],
            )
        )

    assert exc.value.reason == "clinic_ambiguous"
    assert exc.value.data and exc.value.data.get("candidates")


def test_workflow_doctor_not_in_requested_clinic_raises_mismatch():
    class _MismatchClient(FakeMCPClientClinicResolution):
        async def match_doctor_hybrid(self, *, query: str, clinic_id: str | None = None, clinic_name=None, **kwargs):
            # Simulate: NO_MATCH when constrained by clinic, but match globally.
            if clinic_id is not None:
                return HybridMatchResponse(
                    status=HybridMatchStatus.NO_MATCH,
                    message="لم يتم العثور على دكتور بهذا الاسم.",
                    query_tokens=[],
                    best_match=None,
                    candidates=[],
                )
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

    workflow = ClinicWorkflowService(mcp_client=_MismatchClient())
    state = ConversationState(
        entities=Entities(doctor="بيمن عادل عزيز", clinic="عيادة نسا وتوليد", hospital=None, symptoms=[], specialty=None),
        intent="check_availability",
        target_entity_type="doctor",
        last_user_question="مواعيد بيمن عادل عزيز عيادة نسا وتوليد",
        needs_followup=False,
    )

    with pytest.raises(MCPWorkflowError) as exc:
        asyncio.run(
            workflow.run(
                decision=_decision("check_availability"),
                state=state,
                question=state.last_user_question,
                qa_engine=FakeQAEngine(),
                chat_history=[],
            )
        )

    assert exc.value.reason == "provider_clinic_mismatch"
    assert exc.value.data and exc.value.data.get("candidates")


