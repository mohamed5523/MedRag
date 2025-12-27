import pytest


def test_parse_candidate_selection_accepts_arabic_and_latin_digits():
    from app.core.conversation_controller import parse_candidate_selection

    candidates = ["A", "B", "C"]

    assert parse_candidate_selection("2", candidates) == 1
    assert parse_candidate_selection("٢", candidates) == 1
    assert parse_candidate_selection("اختار ٣", candidates) == 2


def test_parse_candidate_selection_fuzzy_matches_partial_name():
    from app.core.conversation_controller import parse_candidate_selection

    candidates = [
        "فادى فوزى فرج دوس",
        "فادى فوزى سعيد مقار",
    ]

    # user can reply with a partial tail of the name
    assert parse_candidate_selection("سعيد مقار", candidates) == 1


def test_infer_specialty_from_symptoms_abdominal_pain():
    from app.core.conversation_controller import infer_specialty_from_symptoms

    query = "بطني بتوجعني النهاردة اروح لمين؟"
    assert infer_specialty_from_symptoms(query) == "باطنة"


def test_is_symptom_triage_request_detects_go_to_who_for_symptoms():
    from app.core.conversation_controller import is_symptom_triage_request

    assert is_symptom_triage_request("بطني بتوجعني النهاردة اروح لمين؟") is True
    assert is_symptom_triage_request("عايز أعرف سعر الكشف") is False


def test_resolve_pending_action_preserves_original_intent():
    from app.core.conversation_controller import resolve_pending_action

    pending = {
        "type": "provider_disambiguation",
        "intent": "ask_price",
        "turns_remaining": 2,
        "candidates": [
            {"name_ar": "فادى فوزى فرج دوس", "provider_id": 1, "clinic_id": 10},
            {"name_ar": "فادى فوزى سعيد مقار", "provider_id": 2, "clinic_id": 10},
        ],
    }

    resolution = resolve_pending_action("2", pending)
    assert resolution is not None
    assert resolution["intent"] == "ask_price"
    assert resolution["selected"]["name_ar"] == "فادى فوزى سعيد مقار"


def test_resolve_pending_action_symptom_triage_infers_specialty_from_followup():
    from app.core.conversation_controller import resolve_pending_action

    pending = {
        "type": "symptom_triage",
        "intent": "list_doctors",
        "turns_remaining": 1,
        "original_question": "انا تعبان ومحتاج اروح لمين؟",
    }

    resolution = resolve_pending_action("بطني بتوجعني", pending)
    assert resolution is not None
    assert resolution["intent"] == "list_doctors"
    assert resolution["specialty"] == "باطنة"

