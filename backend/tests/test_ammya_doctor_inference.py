import pytest
from app.services.clinic_workflow import _infer_doctor_from_text, _is_plausible_doctor_name

@pytest.mark.parametrize("text,expected", [
    ("دكتور جاى الضهر", None),
    ("دكتور جاى", None),
    ("دكتور الضهر", None),
    ("دكتور دلوقتي", None),
    ("دكتور بكره", None),
    ("مواعيد دكتور جاى الضهر", None),
    ("دكتور اطفال", None), # specialty (usually caught by CLINIC_SPECIALTY_WORDS)
    ("دكتور اطفال متاح", None),
    ("دكتور ابانوب", "ابانوب"),
    ("دكتور احمد محمد", "احمد محمد"),
    ("دكتور جاى من السفر", None),
    ("دكتور رايح المستشفى", None),
    ("دكتور موجود", None),
    ("دكتور متاح دلوقتي", None),
])
def test_infer_doctor_ammya_rejections(text, expected):
    from app.services.clinic_workflow import _infer_doctor_from_text
    assert _infer_doctor_from_text(text) == expected

def test_is_plausible_doctor_name_ammya():
    from app.services.clinic_workflow import _is_plausible_doctor_name
    # One word check
    assert _is_plausible_doctor_name("جاى") is False
    assert _is_plausible_doctor_name("الضهر") is False
    # Multi-token check
    assert _is_plausible_doctor_name("جاى الضهر") is False
    # Name check
    assert _is_plausible_doctor_name("ابانوب") is True
