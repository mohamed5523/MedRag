"""Tests that _infer_doctor_from_text correctly rejects specialty/clinic words.

Covers the hallucination bug where "دكتور أطفال حالياً" caused the system
to extract "أطفال حالياً" as a doctor name and then fuzzy-match fake doctor names.
"""

import pytest

from app.services.clinic_workflow import _infer_doctor_from_text, _is_plausible_doctor_name


class TestIsPlausibleDoctorNameRejectsSpecialties:
    """_is_plausible_doctor_name must reject specialty words."""

    @pytest.mark.parametrize(
        "candidate",
        [
            "أطفال حالياً",
            "أطفال متاح",
            "أطفال النهارده",
            "اطفال حاليا",
            "باطنة متاح",
            "عظام النهارده",
            "جراحة اليوم",
            "جلدية موجود",
            "أسنان متاح",
            "عيون",
            "نسا وتوليد",
            "أعصاب متاح",
            "قلب النهارده",
            "صدر متاح",
            "مسالك بوليه",
        ],
    )
    def test_rejects_specialty_words(self, candidate):
        assert _is_plausible_doctor_name(candidate) is False

    @pytest.mark.parametrize(
        "candidate",
        [
            "أحمد محمد",
            "ابانوب ميلاد",
            "محمد عبدالله",
            "مينا شوقي",
            "ماريا جورج",
        ],
    )
    def test_accepts_real_names(self, candidate):
        assert _is_plausible_doctor_name(candidate) is True


class TestInferDoctorRejectsSpecialtyPhrases:
    """_infer_doctor_from_text must return None for specialty-based queries."""

    @pytest.mark.parametrize(
        "text",
        [
            "محتاج دكتور أطفال حالياً",
            "مين دكتور اطفال متاح حاليا",
            "عايز دكتور باطنة النهارده",
            "أنا محتاج دكتور عظام",
            "فين دكتور أسنان متاح",
            "دكتور جراحة موجود",
            "دكتور جلدية متاح النهارده",
        ],
    )
    def test_returns_none_for_specialty_queries(self, text):
        result = _infer_doctor_from_text(text)
        assert result is None, f"Expected None for '{text}', got '{result}'"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("دكتور أحمد محمد", "أحمد محمد"),
            ("عايز دكتور ابانوب ميلاد", "ابانوب ميلاد"),
        ],
    )
    def test_still_extracts_real_names(self, text, expected):
        result = _infer_doctor_from_text(text)
        assert result == expected, f"Expected '{expected}' for '{text}', got '{result}'"
