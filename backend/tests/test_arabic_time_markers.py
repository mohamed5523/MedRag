from app.core.qa_engine import QAEngine
from app.integrations.mcp_client import ProviderScheduleResponse, ScheduleSlot
from app.services.clinic_workflow import _format_schedule


def test_qa_engine_normalizes_time_markers_to_tanween_words():
    engine = QAEngine(model="fake-model")
    text = "المواعيد من 09:30 AM لحد 07.30م وكمان صباحا وصباحً ومساءا"
    out = engine._normalize_arabic_ampm(text)

    assert "09:30 صباحًا" in out
    assert "07.30 مساءً" in out
    assert "صباحًا" in out
    assert "مساءً" in out
    assert "AM" not in out
    assert "PM" not in out


def test_format_schedule_normalizes_shift_times_to_tanween_words():
    response = ProviderScheduleResponse(
        slots=[
            ScheduleSlot(
                day_id=1,
                day_name="السبت",
                shift_start="09:00 AM",
                shift_end="11.00م",
            )
        ]
    )

    rendered = _format_schedule(response, provider_entry=None)
    assert "09:00 صباحًا" in rendered
    assert "11.00 مساءً" in rendered


