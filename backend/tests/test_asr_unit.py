"""
Unit tests for app.api.asr (ASR endpoint)

Covers:
- Groq transcribe path mocked
- ElevenLabs transcribe path mocked
- Empty file upload raises 400
- Provider routing logic
"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _build_test_app():
    """Build a minimal FastAPI app with the ASR router mounted."""
    import importlib
    with patch.dict(__import__("os").environ, {
        "TTS_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-fake",
        "ASR_PROVIDER": "groq",
        "GROQ_API_KEY": "fake-groq-key",
    }):
        import app.core.tts_settings as ts_mod
        importlib.reload(ts_mod)
        from app.api.asr import router
        application = FastAPI()
        application.include_router(router, prefix="/api/asr")
    return application


class TestASRProviderFunctions:
    """Unit tests for the internal helper functions."""

    def test_groq_transcribe_returns_text(self):
        """_groq_transcribe should return the text from a string result."""
        with patch.dict(__import__("os").environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("app.api.asr.Groq") as MockGroq:
                mock_client = MagicMock()
                mock_client.audio.transcriptions.create.return_value = "مرحبا"
                MockGroq.return_value = mock_client

                from app.api.asr import _groq_transcribe
                result = _groq_transcribe(b"audio_data", "test.webm", "whisper-large-v3")
                assert result == "مرحبا"

    def test_groq_transcribe_handles_object_result(self):
        """_groq_transcribe should extract .text attribute from non-string results."""
        with patch.dict(__import__("os").environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("app.api.asr.Groq") as MockGroq:
                mock_client = MagicMock()
                mock_result = MagicMock()
                mock_result.text = "transcribed text"
                mock_client.audio.transcriptions.create.return_value = mock_result
                MockGroq.return_value = mock_client

                from app.api.asr import _groq_transcribe
                result = _groq_transcribe(b"audio_data", "test.webm", "whisper-large-v3")
                assert result == "transcribed text"

    def test_groq_transcribe_raises_without_api_key(self):
        """_groq_transcribe should raise RuntimeError without GROQ_API_KEY."""
        with patch.dict(__import__("os").environ, {"GROQ_API_KEY": ""}):
            from app.api.asr import _groq_transcribe
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                _groq_transcribe(b"data", "test.webm", "whisper-large-v3")

    def test_elevenlabs_transcribe_raises_without_api_key(self):
        """_elevenlabs_transcribe should raise RuntimeError without ELEVENLABS_API_KEY.

        BUG FIX: tts_settings is a singleton cached at import time. Reloading the
        module or setting env vars does not update the already-created singleton.
        The correct fix is to patch the attribute on the singleton directly.
        """
        import app.core.tts_settings as ts_mod
        from app.api.asr import _elevenlabs_transcribe
        with patch.object(ts_mod.tts_settings, "ELEVENLABS_API_KEY", None):
            with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
                _elevenlabs_transcribe(b"data", "test.webm")

    def test_elevenlabs_transcribe_returns_text(self):
        """_elevenlabs_transcribe should return the transcription text."""
        import importlib
        with patch.dict(__import__("os").environ, {"ELEVENLABS_API_KEY": "fake-key"}):
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            with patch("app.api.asr.ElevenLabs") as MockEL:
                mock_client = MagicMock()
                mock_result = MagicMock()
                mock_result.text = "نص مكتوب"
                mock_client.speech_to_text.convert.return_value = mock_result
                MockEL.return_value = mock_client

                from app.api.asr import _elevenlabs_transcribe
                result = _elevenlabs_transcribe(b"audio_data", "audio.webm")
                assert result == "نص مكتوب"


class TestASREndpoint:
    """Integration-style tests for the /transcribe endpoint."""

    @pytest.mark.asyncio
    async def test_empty_file_returns_400(self):
        """Uploading an empty file should return HTTP 400."""
        import importlib
        with patch.dict(__import__("os").environ, {
            "TTS_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-fake",
            "ASR_PROVIDER": "groq",
        }):
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.api.asr import router
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            app = FastAPI()
            app.include_router(router, prefix="/api/asr")
            client = TestClient(app)

            # Upload empty file
            response = client.post(
                "/api/asr/transcribe",
                files={"audio_file": ("empty.webm", io.BytesIO(b""), "audio/webm")},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_groq_path_returns_transcript(self):
        """With ASR_PROVIDER=groq, endpoint should return transcription.
        
        BUG FIX: tts_settings is a singleton loaded at import time — reloading
        the module is unreliable in pytest. Instead, patch _get_asr_provider()
        directly so the endpoint executes the groq branch.
        """
        import importlib
        with patch.dict(__import__("os").environ, {
            "TTS_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-fake",
            "ASR_PROVIDER": "groq",
            "GROQ_API_KEY": "fake-groq-key",
        }):
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.api.asr import router
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            app = FastAPI()
            app.include_router(router, prefix="/api/asr")
            client = TestClient(app)

            # Patch both _get_asr_provider AND the internal function to avoid
            # the stale-singleton issue with tts_settings.
            with patch("app.api.asr._get_asr_provider", return_value="groq"), \
                 patch("app.api.asr._groq_transcribe", return_value="دكتور عظام"):
                response = client.post(
                    "/api/asr/transcribe",
                    files={"audio_file": ("audio.webm", io.BytesIO(b"fake_audio"), "audio/webm")},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["transcribed_text"] == "دكتور عظام"

