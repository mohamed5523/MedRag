from app.core.conversation_controller import (
    apply_pending_action_resolution,
    format_provider_disambiguation_prompt,
)


def test_format_provider_disambiguation_prompt_single_candidate_is_confirmation() -> None:
    prompt = format_provider_disambiguation_prompt([{"name_ar": "بيمن عادل عزيز بساده"}])
    assert "هل تقصد دكتور" in prompt
    assert "بيمن عادل عزيز بساده" in prompt
    # Accept both Latin and Arabic digit confirmation
    assert "1" in prompt
    assert "١" in prompt


def test_format_provider_disambiguation_prompt_multi_candidate_is_numbered_list() -> None:
    prompt = format_provider_disambiguation_prompt(
        [{"name_ar": "أحمد علي"}, {"name_ar": "أحمد محمد"}]
    )
    assert "فيه أكتر من دكتور" in prompt
    assert "1 أحمد علي" in prompt
    assert "2 أحمد محمد" in prompt


def test_apply_pending_action_resolution_provider_disambiguation_materializes_query() -> None:
    resolution = {
        "intent": "check_availability",
        "selected": {"name_ar": "بيمن عادل عزيز بساده", "clinic_name": "عيادة 1"},
    }
    forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query = (
        apply_pending_action_resolution("provider_disambiguation", resolution, "بيمن عادل عزيز بساده")
    )

    assert forced_intent == "check_availability"
    assert forced_doctor == "بيمن عادل عزيز بساده"
    assert forced_clinic == "عيادة 1"
    assert forced_specialty is None
    assert "مواعيد" in state_input_query
    assert "بيمن عادل عزيز بساده" in state_input_query


def test_apply_pending_action_resolution_symptom_triage_forces_list_doctors() -> None:
    resolution = {
        "specialty": "باطنة",
        "combined_query": "بطني بتوجعني النهاردة اروح لمين؟",
    }
    forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query = (
        apply_pending_action_resolution("symptom_triage", resolution, "اه بطني بتوجعني")
    )

    assert forced_intent == "list_doctors"
    assert forced_specialty == "باطنة"
    assert forced_doctor is None
    assert forced_clinic is None
    assert "بطني بتوجعني" in state_input_query


