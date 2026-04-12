from app.core.conversation_controller import (
    apply_pending_action_resolution,
    apply_context_switch_rules,
    format_clinic_disambiguation_prompt,
    format_provider_disambiguation_prompt,
)
from app.core.state_manager import Entities


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
    assert "1 - أحمد علي" in prompt
    assert "2 - أحمد محمد" in prompt


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


def test_apply_pending_action_resolution_provider_clinic_mismatch_materializes_query() -> None:
    resolution = {
        "intent": "check_availability",
        "selected": {"name_ar": "بيمن عادل عزيز بساده", "clinic_name": "جراحه"},
    }
    forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query = (
        apply_pending_action_resolution("provider_clinic_mismatch", resolution, "بيمن")
    )

    assert forced_intent == "check_availability"
    assert forced_doctor == "بيمن عادل عزيز بساده"
    assert forced_clinic == "جراحه"
    assert forced_specialty is None
    assert "مواعيد" in state_input_query
    assert "جراحه" in state_input_query


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


def test_format_clinic_disambiguation_prompt_multi_candidate_is_numbered_list() -> None:
    prompt = format_clinic_disambiguation_prompt(
        [{"clinic_name": "عيادة نسا وتوليد"}, {"clinic_name": "عيادة نسا"}]
    )
    assert "فيه أكتر من عيادة" in prompt
    assert "1 - عيادة نسا وتوليد" in prompt
    assert "2 - عيادة نسا" in prompt


def test_apply_pending_action_resolution_clinic_disambiguation_materializes_query() -> None:
    resolution = {
        "intent": "check_availability",
        "selected": {"clinic_name": "عيادة نسا وتوليد", "clinic_id": 10},
    }
    forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query = (
        apply_pending_action_resolution("clinic_disambiguation", resolution, "مواعيد")
    )
    assert forced_intent == "check_availability"
    assert forced_doctor is None
    assert forced_clinic == "عيادة نسا وتوليد"
    assert forced_specialty is None
    assert "مواعيد" in state_input_query
    assert "عيادة نسا وتوليد" in state_input_query


def test_apply_pending_action_resolution_prefers_original_question_to_preserve_multi_intent() -> None:
    # User originally asked a multi-intent question (schedule + price),
    # then disambiguated the clinic by picking an option number.
    resolution = {
        "intent": "check_availability",
        "selected": {"clinic_name": "جراحه", "clinic_id": 1097},
        "original_question": "مين دكتور النهارده متاح في عيادة الجراحة و سعر الكشف",
    }
    forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query = (
        apply_pending_action_resolution("clinic_disambiguation", resolution, "1")
    )
    assert forced_intent == "check_availability"
    assert forced_doctor is None
    assert forced_clinic == "جراحه"
    assert forced_specialty is None
    assert state_input_query == resolution["original_question"]


def test_apply_context_switch_rules_overwrites_stale_clinic_and_clears_clinic_id() -> None:
    entities = Entities(clinic="عيادة المسا و التوليد", clinic_id=999, doctor=None, provider_id=None)
    apply_context_switch_rules("مين موجود فى عيادة الجراحه النهارده؟", entities)
    assert entities.clinic is not None
    assert "عيادة" in entities.clinic
    assert "الجراح" in entities.clinic  # tolerate normalization (ه/ة)
    assert entities.clinic_id is None


def test_apply_context_switch_rules_overwrites_stale_clinic_when_new_query_mentions_new_clinic() -> None:
    entities = Entities(clinic="عيادة المسا و التوليد", clinic_id=999, doctor=None, provider_id=None)
    apply_context_switch_rules("مين موجود فى عيادة الاسنان النهارده؟", entities)
    assert entities.clinic is not None
    assert "اسنان" in entities.clinic
    assert entities.clinic_id is None


