import pytest

from app.services.clinic_workflow import _infer_phrase_after_keywords


def test_infer_phrase_after_clinic_ignores_doctor_marker():
    """
    Regression: 'عيادة دكتور <name>' should NOT be inferred as a clinic name.
    Users often mean "the doctor's clinic", and clinic resolution should not fail early.
    """
    text = "مواعيد عيادة دكتور بيمن عادل عزيز و سعر كشفه"
    inferred = _infer_phrase_after_keywords(text, keywords=["عيادة", "عياده", "clinic"], max_words=4)
    assert inferred is None


