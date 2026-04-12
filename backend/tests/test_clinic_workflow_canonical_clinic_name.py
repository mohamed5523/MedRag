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

    async def get_clinic_provider_schedule(self, clinic_id: int, date_from: str, date_to: str, provider_id=None):
        # Minimal non-empty schedule so _handle_schedule doesn't error
        return ProviderScheduleResponse(
            slots=[
                ScheduleSlot(
                    clinic_id=clinic_id,
                    provider_id=8001,
                    day_id=getattr(self, "_expected_day_id", 1),
                    day_name="Tuesday",
                    shift_start="09:00",
                    shift_end="11:00",
                )
            ]
        )

    async def get_clinic_provider_list(self):
        # _handle_schedule now calls this to find providers for the matched clinic.
        # Return a record for clinic_id=2001 (the canonical ID returned by match_clinic_hybrid)
        # so the workflow can proceed to fetch schedules. The actual test assertion is
        # about the canonical clinic name appearing in the output doc.
        from app.integrations.mcp_client import ProviderListPayload, ProviderRecord
        return ProviderListPayload(providers=[
            ProviderRecord(
                clinic_id=2001,
                clinic_name_ar="نسا وتوليد",
                provider_id=8001,
                provider_name_ar="دكتور اختبار",
            )
        ])

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
    # BUG FIX: _handle_schedule now uses "mcp.clinic_schedule_all" source for clinic-only
    # queries (no specific doctor), replacing the old "mcp.provider_schedule" key.
    assert docs and docs[0].metadata.get("source") in {"mcp.provider_schedule", "mcp.clinic_schedule_all"}
    assert "نسا وتوليد" in docs[0].page_content
    assert "المسا" not in docs[0].page_content


