"""
Integration tests for the WhatsApp webhook (app.api.whatsapp)

Uses FastAPI TestClient with all external APIs mocked.

Covers:
- GET webhook verification handshake (valid + invalid token)
- POST text message with mocked LLM response
- Duplicate message deduplication logic (Redis)
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


ENV_OVERRIDES = {
    "TTS_PROVIDER": "openai",
    "OPENAI_API_KEY": "sk-fake",
    "WHATSAPP_TOKEN": "fake-wa-token",
    "WHATSAPP_PHONE_NUMBER_ID": "fake-phone-id",
    "WHATSAPP_VERIFY_TOKEN": "my-test-verify-token",
    "ASR_PROVIDER": "groq",
    "GROQ_API_KEY": "fake-groq",
}

# Sample WhatsApp webhook payload for a text message
def _make_wa_payload(message_id: str = "msg123", body: str = "مرحبا", from_number: str = "201234567890") -> dict:
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": message_id,
                        "from": from_number,
                        "type": "text",
                        "text": {"body": body}
                    }]
                }
            }]
        }]
    }


@pytest.fixture(scope="module")
def whatsapp_client():
    """Build a TestClient for the WhatsApp router with all externals mocked."""
    import importlib
    with patch.dict(__import__("os").environ, ENV_OVERRIDES):
        import app.core.tts_settings as ts_mod
        importlib.reload(ts_mod)

        # Patch all heavy dependencies before import
        with patch("app.api.whatsapp.QAEngine") as mock_qa_cls, \
             patch("app.api.whatsapp.ClinicWorkflowService") as mock_clinic_cls, \
             patch("app.api.whatsapp.TextToSpeech") as mock_tts_cls, \
             patch("app.api.whatsapp.redis_client") as mock_redis, \
             patch("app.api.whatsapp.short_term_memory") as mock_memory, \
             patch("app.api.whatsapp.session_manager") as mock_session, \
             patch("app.api.whatsapp.state_manager") as mock_state_mgr, \
             patch("app.api.whatsapp.VectorStore") as mock_vs:

            # Configure QAEngine mock
            mock_qa = MagicMock()
            mock_qa.is_available.return_value = True
            mock_qa.build_time_context.return_value = {
                "is_arabic": True,
                "date_hint": "اليوم",
                "time_hint": "9:00 صباحًا",
                "tz_name": "Africa/Cairo",
                "now_iso": "2025-01-01T09:00:00",
                "now_dt": MagicMock(),
                "time_context_message": "Now: 9:00",
            }
            mock_qa.rewrite_query_with_date_hint.return_value = ("rewritten query", {})
            mock_qa.answer_question = AsyncMock(return_value={"answer": "الخدمة متاحة"})
            mock_qa_cls.return_value = mock_qa

            # Configure Redis mock
            mock_redis.exists.return_value = False
            mock_redis.set.return_value = True

            # Configure memory mock
            mock_memory.add_message.return_value = None
            mock_memory.get_messages.return_value = []
            mock_memory.get_state.return_value = None
            mock_memory.get_pending_action.return_value = None
            mock_memory.save_state.return_value = None

            # Configure state manager mock
            mock_new_state = MagicMock()
            mock_new_state.intent = "general_inquiry"
            mock_new_state.target_entity_type = "unknown"
            mock_new_state.entities = MagicMock(
                doctor=None, clinic=None, hospital=None, specialty=None,
                clinic_id=None, provider_id=None
            )
            mock_new_state.last_user_question = "مرحبا"
            mock_new_state.entities.model_dump.return_value = {}
            mock_state_mgr.extract_state.return_value = mock_new_state
            mock_state_mgr.rewrite_query.return_value = "مرحبا"

            # Mock VectorStore
            mock_vs_instance = MagicMock()
            mock_vs_instance.retrieve.return_value = []
            mock_vs.return_value = mock_vs_instance

            # TTS not needed for these tests
            mock_tts_cls.side_effect = Exception("TTS not configured")

            from app.api.whatsapp import router
            from fastapi import FastAPI
            app = FastAPI()
            app.include_router(router)

            yield TestClient(app)


class TestWebhookVerification:
    """Tests for GET /webhook/whatsapp (Meta verification handshake)."""

    def test_valid_token_returns_challenge(self, whatsapp_client):
        """Valid verify_token should return the challenge string with 200."""
        response = whatsapp_client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "my-test-verify-token",
                "hub.challenge": "abc123challenge"
            }
        )
        assert response.status_code == 200
        assert response.text == "abc123challenge"

    def test_invalid_token_returns_403(self, whatsapp_client):
        """Wrong verify_token should return 403."""
        response = whatsapp_client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "WRONG_TOKEN",
                "hub.challenge": "abc123"
            }
        )
        assert response.status_code == 403

    def test_missing_mode_returns_403(self, whatsapp_client):
        """Missing hub.mode should return 403 (mode != 'subscribe')."""
        response = whatsapp_client.get(
            "/webhook/whatsapp",
            params={
                "hub.verify_token": "my-test-verify-token",
                "hub.challenge": "abc123"
            }
        )
        assert response.status_code == 403


class TestWebhookPostMessage:
    """Tests for POST /webhook/whatsapp (incoming messages)."""

    def test_text_message_returns_200_quickly(self, whatsapp_client):
        """POST with a valid text message should return 200 immediately (background task)."""
        payload = _make_wa_payload(message_id="unique-msg-1", body="مرحبا")
        response = whatsapp_client.post(
            "/webhook/whatsapp",
            json=payload
        )
        assert response.status_code == 200

    def test_status_callback_acknowledged(self, whatsapp_client):
        """POSTing a status callback should return 200 without processing."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "statuses": [{"id": "msg1", "status": "delivered"}]
                    }
                }]
            }]
        }
        response = whatsapp_client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200

    def test_empty_entry_acknowledged(self, whatsapp_client):
        """POST with no entries should be acknowledged gracefully."""
        response = whatsapp_client.post("/webhook/whatsapp", json={"entry": []})
        assert response.status_code == 200


class TestMessageDeduplication:
    """Tests for duplicate message deduplication logic."""

    def test_duplicate_message_returns_200_and_skips(self):
        """A message with the same ID seen twice should be ignored on second receipt."""
        import importlib
        with patch.dict(__import__("os").environ, ENV_OVERRIDES):
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)

            with patch("app.api.whatsapp.QAEngine") as mock_qa_cls, \
                 patch("app.api.whatsapp.ClinicWorkflowService"), \
                 patch("app.api.whatsapp.TextToSpeech", side_effect=Exception), \
                 patch("app.api.whatsapp.redis_client") as mock_redis, \
                 patch("app.api.whatsapp.short_term_memory") as mock_memory, \
                 patch("app.api.whatsapp.state_manager") as mock_state_mgr, \
                 patch("app.api.whatsapp.session_manager"), \
                 patch("app.api.whatsapp.VectorStore"):

                # Configure mocks
                mock_redis.exists.return_value = True  # Already seen!
                mock_redis.set.return_value = True
                mock_memory.get_messages.return_value = []
                mock_memory.get_pending_action.return_value = None

                mock_qa = MagicMock()
                mock_qa_cls.return_value = mock_qa

                from app.api.whatsapp import router
                from fastapi import FastAPI
                from fastapi.testclient import TestClient

                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)

                payload = _make_wa_payload(message_id="dup-msg-id")
                response = client.post("/webhook/whatsapp", json=payload)

                # Should still return 200 (not an error)
                assert response.status_code == 200
                assert "Duplicate" in response.text
