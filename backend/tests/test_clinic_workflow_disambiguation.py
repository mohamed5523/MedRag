import asyncio

import pytest

from app.core.intent_router import RouteDecision, RouteMode
from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import DoctorMatchResult, HybridMatchResponse, HybridMatchStatus
from app.services.clinic_workflow import ClinicWorkflowService, MCPWorkflowError


class FakeQAEngine:
    model = "fake-model"

    def build_time_context(self, question, now_dt=None):
        return {"is_arabic": True, "tz_name": "Africa/Cairo", "date_hint": "اليوم", "time_context_message": "stub"}

    async def answer_question(self, *, question, contexts, time_context, chat_history=None, user_gender=None):
        return {"answer": "ok", "sources": [], "context_count": len(contexts), "model_used": self.model, "tokens_used": 0}


class FakeMCPClientAmbiguous:
    async def match_doctor_hybrid(self, **kwargs):
        query = kwargs.get("query", "")
        return HybridMatchResponse(
            status=HybridMatchStatus.AMBIGUOUS_NEED_MORE_INFO,
            message="يوجد أكثر من دكتور بنفس الاسم.",
            query_tokens=[],
            best_match=None,
            candidates=[
                DoctorMatchResult(
                    provider_id="1",
                    clinic_id="10",
                    clinic_name="عيادة الباطنة",
                    name_ar=query or "بيمن عادل عزيز بساده",
                    name_en="",
                    score=0.7,
                    token_overlap=0.7,
                    fuzzy_name_score=0.7,
                    position_score=0.7,
                    matched_by_first_name=True,
                    matched_tokens=[],
                ),
                DoctorMatchResult(
                    provider_id="2",
                    clinic_id="10",
                    clinic_name="عيادة الباطنة",
                    name_ar="بيمن عادل عزيز بساده 2",
                    name_en="",
                    score=0.65,
                    token_overlap=0.65,
                    fuzzy_name_score=0.65,
                    position_score=0.65,
                    matched_by_first_name=True,
                    matched_tokens=[],
                ),
            ],
        )


def _decision(intent="ask_price"):
    return RouteDecision(mode=RouteMode.MCP, intent=intent, reason="test", tool_sequence=[], entities_snapshot={})


def test_workflow_ambiguous_match_raises_error_with_candidates_payload():
    workflow = ClinicWorkflowService(mcp_client=FakeMCPClientAmbiguous())
    state = ConversationState(
        entities=Entities(doctor="بيمن", clinic=None, hospital=None, symptoms=[], specialty=None),
        intent="ask_price",
        target_entity_type="doctor",
        last_user_question="سعر كشف دكتور بيمن",
        needs_followup=False,
    )

    with pytest.raises(MCPWorkflowError) as exc:
        asyncio.run(
            workflow.run(
                decision=_decision("ask_price"),
                state=state,
                question="سعر كشف دكتور بيمن",
                qa_engine=FakeQAEngine(),
                chat_history=[],
            )
        )

    assert exc.value.reason == "provider_ambiguous"
    assert hasattr(exc.value, "data")
    assert exc.value.data is not None
    assert "candidates" in exc.value.data
    assert len(exc.value.data["candidates"]) == 2

