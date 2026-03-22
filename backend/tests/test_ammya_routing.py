"""Tests for Ammya query routing fixes in intent_router and clinic_workflow."""

import pytest

from app.core.intent_router import _is_ammya_clinic_query
from app.services.clinic_workflow import _is_who_is_present_query


class TestAmmyaClinicQueryGuard:
    """Verifies the _is_ammya_clinic_query function correctly identifies common Ammya patterns."""

    @pytest.mark.parametrize(
        "query",
        [
            "مين دكتور أطفال متاح حاليا",
            "مين دكتورة باطنة موجودة",
            "محتاج دكتور أسنان",
            "عايز دكتور عظام",
            "عاوز دكتور رمد",
            "فين دكتور قلب",
            "دكتور جلدية متاح",
            "محتاج أطفال",
            "عايز نسا وتوليد",
            "مواعيد دكاترة اطفال",
            "مواعيد أطباء جراحة",
        ],
    )
    def test_ammya_patterns_match(self, query):
        assert _is_ammya_clinic_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "أين تقع المستشفى",
            "ما هي تخصصات المستشفى",
            "علاج الضغط",
            "أعراض السكر",
        ],
    )
    def test_non_clinic_queries_fail(self, query):
        assert _is_ammya_clinic_query(query) is False


class TestIsWhoIsPresentQuery:
    """Verifies the broadened _is_who_is_present_query handles Ammya patterns."""

    @pytest.mark.parametrize(
        "query",
        [
            "مين موجود",
            "مين متاح حاليا",
            "مين اللي موجودين",
            "مين دكتور أطفال متاح",
            "مين دكتور باطنة موجود",
            "محتاج دكتور عظام",
            "عايز دكتور أسنان حاليا",
            "عاوز دكتور رمد النهارده",
        ],
    )
    def test_who_is_present_patterns_match(self, query):
        assert _is_who_is_present_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "فين المستشفى",
            "احجز موعد مع دكتور أحمد",
            "سعر كشف الباطنة",
        ],
    )
    def test_other_queries_fail(self, query):
        assert _is_who_is_present_query(query) is False
