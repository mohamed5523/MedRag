"""
Unit tests for app.core.qa_engine.QAEngine

Covers:
- Arabic detection logic
- AM/PM normalization
- Time context building
- Graceful degradation with no API key
- Mocked LLM response shape
"""
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Force no API key to test fallback paths cleanly
os.environ.setdefault("OPENAI_API_KEY", "")


def _make_doc(text: str, source: str = "TestDoc") -> MagicMock:
    """Create a fake langchain Document for testing."""
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"source": source}
    return doc


# ---------------------------------------------------------------------------
# Arabic detection
# ---------------------------------------------------------------------------

class TestArabicDetection:
    """Tests for QAEngine._is_arabic_query."""

    def setup_method(self):
        """Create QAEngine instance with no API key (graceful degradation)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            from app.core.qa_engine import QAEngine
            self.engine = QAEngine()

    def test_pure_arabic_detected(self):
        """Arabic text with >50% Arabic chars should return True."""
        assert self.engine._is_arabic_query("مرحبا كيف حالك") is True

    def test_english_not_arabic(self):
        """Pure English text should return False."""
        assert self.engine._is_arabic_query("Hello world, how are you?") is False

    def test_empty_string_not_arabic(self):
        """Empty string should return False safely (no division by zero)."""
        assert self.engine._is_arabic_query("") is False

    def test_mixed_mostly_arabic(self):
        """Text that is >50% Arabic chars should return True."""
        assert self.engine._is_arabic_query("أهلاً hello") is True

    def test_mixed_mostly_english(self):
        """Text that is <50% Arabic chars should return False."""
        assert self.engine._is_arabic_query("hello world مرحبا") is False

    def test_numbers_only_not_arabic(self):
        """Numeric-only string (no alpha chars) should return False."""
        assert self.engine._is_arabic_query("12345 67890") is False


# ---------------------------------------------------------------------------
# AM/PM normalization
# ---------------------------------------------------------------------------

class TestNormalizeArabicAmpm:
    """Tests for QAEngine._normalize_arabic_ampm."""

    def setup_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            from app.core.qa_engine import QAEngine
            self.engine = QAEngine()

    def test_english_am_replaced(self):
        """English 'AM' should be replaced by 'صباحًا'."""
        result = self.engine._normalize_arabic_ampm("9:30 AM")
        assert "صباحًا" in result
        assert "AM" not in result

    def test_english_pm_replaced(self):
        """English 'PM' should be replaced by 'مساءً'."""
        result = self.engine._normalize_arabic_ampm("3:00 PM")
        assert "مساءً" in result
        assert "PM" not in result

    def test_arabic_single_letter_sah_after_time(self):
        """Arabic ص after time should become صباحًا."""
        result = self.engine._normalize_arabic_ampm("09:30ص")
        assert "صباحًا" in result

    def test_arabic_single_letter_msa_after_time(self):
        """Arabic م after time should become مساءً."""
        result = self.engine._normalize_arabic_ampm("09:30م")
        assert "مساءً" in result

    def test_empty_string_passthrough(self):
        """Empty string should return empty string without error."""
        assert self.engine._normalize_arabic_ampm("") == ""

    def test_no_time_markers_unchanged(self):
        """Text with no time markers should be returned as-is."""
        text = "مرحباً بك في المستشفى"
        assert self.engine._normalize_arabic_ampm(text) == text

    def test_am_case_insensitive(self):
        """Lowercase 'am' should also be replaced."""
        result = self.engine._normalize_arabic_ampm("10:00 am")
        assert "صباحًا" in result

    def test_pm_dotted_variant(self):
        """'P.M.' should be replaced by مساءً."""
        result = self.engine._normalize_arabic_ampm("5:00 P.M.")
        assert "مساءً" in result


# ---------------------------------------------------------------------------
# Time context building
# ---------------------------------------------------------------------------

class TestBuildTimeContext:
    """Tests for QAEngine.build_time_context."""

    def setup_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            from app.core.qa_engine import QAEngine
            self.engine = QAEngine()

    def test_returns_required_keys(self):
        """Context dict must contain all required keys."""
        ctx = self.engine.build_time_context("hello")
        required = {"now_dt", "tz_name", "is_arabic", "date_hint", "time_hint", "now_iso", "time_context_message"}
        assert required.issubset(ctx.keys())

    def test_arabic_query_arabic_date_hint(self):
        """Arabic query should produce Arabic-script date_hint."""
        ctx = self.engine.build_time_context("مرحبا")
        # Should contain Arabic weekday name
        assert any(day in ctx["date_hint"] for day in ["السبت", "الأحد", "الاتنين", "التلات", "الأربع", "الخميس", "الجمعة"])

    def test_english_query_english_date_hint(self):
        """English query should produce English date_hint."""
        ctx = self.engine.build_time_context("Hello")
        assert "Today is" in ctx["date_hint"]

    def test_custom_datetime_used(self):
        """Providing now_dt should base context on that datetime."""
        fixed_dt = datetime(2025, 6, 15, 10, 30)
        ctx = self.engine.build_time_context("hello", now_dt=fixed_dt)
        assert "2025" in ctx["now_iso"]

    def test_am_time_suffix_before_noon(self):
        """Morning time should produce صباحًا suffix in Arabic contexts."""
        dt_morning = datetime(2025, 1, 1, 9, 0)
        ctx = self.engine.build_time_context("مرحبا", now_dt=dt_morning)
        assert "صباحًا" in ctx["time_hint"]

    def test_pm_time_suffix_after_noon(self):
        """Afternoon time should produce مساءً suffix in Arabic contexts."""
        dt_afternoon = datetime(2025, 1, 1, 15, 0)
        ctx = self.engine.build_time_context("مرحبا", now_dt=dt_afternoon)
        assert "مساءً" in ctx["time_hint"]

    def test_time_context_message_contains_tz(self):
        """time_context_message should reference the timezone."""
        ctx = self.engine.build_time_context("hello")
        assert ctx["tz_name"] in ctx["time_context_message"]


# ---------------------------------------------------------------------------
# Graceful degradation with no API key
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Ensure QAEngine degrades gracefully without an API key."""

    def setup_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            from app.core.qa_engine import QAEngine
            self.engine = QAEngine()

    @pytest.mark.asyncio
    async def test_answer_question_returns_error_dict_no_key(self):
        """answer_question with no API key returns a dict with error key."""
        result = await self.engine.answer_question(
            question="ما هو موعد العيادة؟",
            contexts=[_make_doc("العيادة تفتح الساعة 9")],
        )
        assert "error" in result
        assert result["answer"] != ""  # Must return a message, not crash

    @pytest.mark.asyncio
    async def test_answer_hybrid_returns_error_dict_no_key(self):
        """answer_with_hybrid_context with no API key returns error dict."""
        result = await self.engine.answer_with_hybrid_context(
            question="بكم الكشف؟",
            mcp_context="سعر الكشف 200 جنيه",
            rag_contexts=[],
        )
        assert "error" in result

    def test_is_available_false_no_key(self):
        """is_available() returns False when client is None."""
        assert self.engine.is_available() is False

    def test_get_model_info_no_key(self):
        """get_model_info should report api_configured=False."""
        info = self.engine.get_model_info()
        assert info["api_configured"] is False


# ---------------------------------------------------------------------------
# Mocked LLM response shape
# ---------------------------------------------------------------------------

class TestMockedLLMResponse:
    """Use mocked OpenAI client to verify response shape handling."""

    def _make_engine_with_mock(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"}):
            from app.core.qa_engine import QAEngine
            engine = QAEngine()
        return engine

    @pytest.mark.asyncio
    async def test_answer_question_returns_correct_shape(self):
        """Mocked LLM call should produce dict with 'answer', 'sources', 'model_used'."""
        engine = self._make_engine_with_mock()

        # Build mock response
        mock_message = MagicMock()
        mock_message.content = "العيادة تفتح الساعة 9 صباحًا."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_usage = MagicMock()
        mock_usage.total_tokens = 42
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine.answer_question(
            question="متى تفتح العيادة؟",
            contexts=[_make_doc("العيادة تفتح الساعة 9")],
        )

        assert "answer" in result
        assert "sources" in result
        assert "model_used" in result
        assert result["answer"] == "العيادة تفتح الساعة 9 صباحًا."
        assert result["tokens_used"] == 42

    @pytest.mark.asyncio
    async def test_answer_question_arabic_normalizes_ampm(self):
        """Arabic response containing AM/PM should be post-processed."""
        engine = self._make_engine_with_mock()

        mock_message = MagicMock()
        mock_message.content = "العيادة تفتح الساعة 9:00 AM"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine.answer_question(
            question="العيادة بتفتح امتى؟",
            contexts=[],
        )
        assert "AM" not in result["answer"]
        assert "صباحًا" in result["answer"]

    @pytest.mark.asyncio
    async def test_answer_question_handles_exception(self):
        """If LLM raises, answer_question should return dict with 'error' key."""
        engine = self._make_engine_with_mock()
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        result = await engine.answer_question(
            question="hello",
            contexts=[],
        )
        assert "error" in result
        assert "answer" in result
