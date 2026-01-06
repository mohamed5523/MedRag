import asyncio

from app.core.state_manager import ConversationState, Entities
from app.integrations.mcp_client import (
    ClinicMatchResponse,
    ClinicMatchResult,
    HybridMatchStatus,
    ProviderScheduleResponse,
    ScheduleSlot,
)
from app.services.clinic_workflow import ClinicWorkflowService


class FakeMCPClientCanonicalClinic:
    async def match_clinic_hybrid(self, **kwargs):
        # Simulate typo correction: "المسا" -> "نسا"
        return ClinicMatchResponse(
            status=HybridMatchStatus.UNAMBIGUOUS_MATCH,
            message="ok",
            query_tokens=["مسا", "توليد"],
            best_match=ClinicMatchResult(
                clinic_id="2001",
                clinic_name="نسا وتوليد",
                score=0.9,
                token_overlap=0.9,
                fuzzy_name_score=0.9,
                order_score=0.9,
                matched_tokens=["مسا", "توليد"],
            ),
            candidates=[],
        )

    async def get_clinic_provider_schedule(self, clinic_id, provider_id=None, day_id=None):
        # Minimal non-empty schedule so _handle_schedule doesn't error
        return ProviderScheduleResponse(
            slots=[
                ScheduleSlot(
                    clinic_id=clinic_id,
                    provider_id=provider_id,
                    day_id=day_id,
                    day_name="Tuesday",
                    shift_start="09:00",
                    shift_end="11:00",
                )
            ]
        )

    async def get_clinic_provider_list(self):
        raise AssertionError("Not needed for this test.")

    async def get_service_price(self, clinic_id, provider_id=None):
        raise AssertionError("Not needed for this test.")

    async def match_doctor_hybrid(self, **kwargs):
        raise AssertionError("Not needed for this test.")


def test_clinic_only_schedule_context_uses_canonical_clinic_name_not_user_typo():
    workflow = ClinicWorkflowService(mcp_client=FakeMCPClientCanonicalClinic())
    state = ConversationState(
        entities=Entities(clinic="عيادة المسا و التوليد", clinic_id=None, doctor=None, provider_id=None),
        intent="check_availability",
        target_entity_type="clinic",
        last_user_question="مواعيد عيادة المسا و التوليد",
        needs_followup=False,
    )

    docs, _ = asyncio.run(workflow._handle_schedule(state, [], state.last_user_question))
    assert docs and docs[0].metadata.get("source") == "mcp.provider_schedule"
    assert "العيادة:" in docs[0].page_content
    assert "نسا وتوليد" in docs[0].page_content
    assert "المسا" not in docs[0].page_content


