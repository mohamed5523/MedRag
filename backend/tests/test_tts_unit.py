"""
Unit tests for app.core.text_to_speech.TextToSpeech

Covers:
- OpenAI provider path (mocked)
- ElevenLabs provider path (mocked)
- Empty text raises ValueError
- Text too long raises ValueError
- Health check returns correct structure
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


def _make_tts(provider: str = "openai", extra_env: dict = None):
    """Create a TextToSpeech instance with the given provider and environment."""
    env = {"TTS_PROVIDER": provider, "OPENAI_API_KEY": "sk-fake"}
    if provider == "elevenlabs":
        env["ELEVENLABS_API_KEY"] = "fake-el-key"
        env["ELEVENLABS_VOICE_ID"] = "fake-voice-id"
    if extra_env:
        env.update(extra_env)
    with patch.dict(os.environ, env, clear=False):
        # Reload settings to pick up new env
        import importlib
        import app.core.tts_settings as ts_mod
        importlib.reload(ts_mod)
        from app.core.text_to_speech import TextToSpeech
        tts = TextToSpeech.__new__(TextToSpeech)
        tts.provider = provider
        tts._client = None
        if provider == "openai":
            from openai import AsyncOpenAI
            tts.openai_client = AsyncMock(spec=AsyncOpenAI)
        else:
            tts.openai_client = None
    return tts


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests for input validation before synthesis."""

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        """Empty string must raise ValueError before hitting any provider."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech
            tts = TextToSpeech.__new__(TextToSpeech)
            tts.provider = "openai"
            tts._client = None
            tts.openai_client = None

        with pytest.raises(ValueError, match="empty"):
            await tts.synthesize("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self):
        """Whitespace-only string should also raise ValueError."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech
            tts = TextToSpeech.__new__(TextToSpeech)
            tts.provider = "openai"
            tts._client = None
            tts.openai_client = None

        with pytest.raises(ValueError, match="empty"):
            await tts.synthesize("   ")

    @pytest.mark.asyncio
    async def test_text_too_long_raises(self):
        """Text longer than 5000 chars must raise ValueError."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech
            tts = TextToSpeech.__new__(TextToSpeech)
            tts.provider = "openai"
            tts._client = None
            tts.openai_client = None

        with pytest.raises(ValueError, match="5000"):
            await tts.synthesize("x" * 5001)


# ---------------------------------------------------------------------------
# OpenAI provider (mocked)
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    """Tests for TTS via OpenAI with the SDK mocked."""

    @pytest.mark.asyncio
    async def test_openai_synthesize_returns_bytes(self):
        """OpenAI path should call the SDK and return bytes."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake", "OPENAI_TTS_VOICE": "nova", "OPENAI_TTS_MODEL": "tts-1"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech

            with patch.object(TextToSpeech, "_validate_env_vars"):
                tts = TextToSpeech.__new__(TextToSpeech)
                tts.provider = "openai"
                tts._client = None

                mock_openai_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.aread = AsyncMock(return_value=b"fake_audio_data")
                mock_openai_client.audio.speech.create = AsyncMock(return_value=mock_response)
                tts.openai_client = mock_openai_client

                with patch("app.core.text_to_speech.tts_settings") as mock_settings:
                    mock_settings.TTS_PROVIDER = "openai"
                    mock_settings.OPENAI_TTS_VOICE = "nova"
                    mock_settings.OPENAI_TTS_MODEL = "tts-1"
                    mock_settings.OPENAI_TTS_AUDIO_FORMAT = "mp3"

                    with patch("app.core.tts_normalization.normalize_arabic_for_tts", return_value="مرحبا"):
                        result = await tts._openai_synthesize("مرحبا", "nova")

                assert isinstance(result, bytes)
                assert result == b"fake_audio_data"

    @pytest.mark.asyncio
    async def test_openai_synthesize_raises_tts_error_on_failure(self):
        """OpenAI path should wrap SDK exceptions in TextToSpeechError."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech
            from app.core.tts_exceptions import TextToSpeechError

            with patch.object(TextToSpeech, "_validate_env_vars"):
                tts = TextToSpeech.__new__(TextToSpeech)
                tts.provider = "openai"
                tts._client = None

                mock_openai_client = AsyncMock()
                mock_openai_client.audio.speech.create = AsyncMock(side_effect=RuntimeError("network error"))
                tts.openai_client = mock_openai_client

                with patch("app.core.tts_normalization.normalize_arabic_for_tts", return_value="hello"):
                    with patch("app.core.text_to_speech.tts_settings") as mock_settings:
                        mock_settings.OPENAI_TTS_MODEL = "tts-1"
                        mock_settings.OPENAI_TTS_VOICE = "nova"
                        mock_settings.OPENAI_TTS_AUDIO_FORMAT = "mp3"

                        with pytest.raises(TextToSpeechError):
                            await tts._openai_synthesize("hello", "nova")


# ---------------------------------------------------------------------------
# ElevenLabs provider (mocked)
# ---------------------------------------------------------------------------

class TestElevenLabsProvider:
    """Tests for TTS via ElevenLabs with SDK mocked."""

    @pytest.mark.asyncio
    async def test_elevenlabs_synthesize_returns_bytes(self):
        """ElevenLabs path should call SDK and return bytes."""
        with patch.dict(os.environ, {
            "TTS_PROVIDER": "elevenlabs",
            "ELEVENLABS_API_KEY": "fake-key",
            "ELEVENLABS_VOICE_ID": "fake-voice",
        }):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech
            from app.core.tts_exceptions import TextToSpeechError

            with patch.object(TextToSpeech, "_validate_env_vars"):
                tts = TextToSpeech.__new__(TextToSpeech)
                tts.provider = "elevenlabs"
                tts.openai_client = None

                # Mock the ElevenLabs client
                mock_el_client = MagicMock()
                mock_el_client.text_to_speech.convert = MagicMock(
                    return_value=iter([b"chunk1", b"chunk2"])
                )
                tts._client = mock_el_client

                with patch("app.core.text_to_speech.tts_settings") as mock_settings:
                    mock_settings.TTS_PROVIDER = "elevenlabs"
                    mock_settings.ELEVENLABS_STABILITY = 0.5
                    mock_settings.ELEVENLABS_SIMILARITY_BOOST = 0.75
                    mock_settings.ELEVENLABS_STYLE = 0.4
                    mock_settings.ELEVENLABS_MODEL = "eleven_multilingual_v2"
                    mock_settings.ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
                    mock_settings.ELEVENLABS_VOICE_ID = "fake-voice"

                    with patch("app.core.tts_normalization.normalize_arabic_for_tts", return_value="مرحبا"):
                        result = await tts.synthesize("مرحبا", voice_id="fake-voice")

                assert isinstance(result, bytes)
                assert result == b"chunk1chunk2"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Tests for TextToSpeech.health_check."""

    @pytest.mark.asyncio
    async def test_health_check_unavailable_returns_unhealthy(self):
        """Health check should return 'unhealthy' when provider not configured."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech

            with patch.object(TextToSpeech, "_validate_env_vars"):
                tts = TextToSpeech.__new__(TextToSpeech)
                tts.provider = "openai"
                tts._client = None
                tts.openai_client = None

                with patch.object(tts, "is_available", return_value=False):
                    result = await tts.health_check()

                assert result["status"] == "unhealthy"
                assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_available_returns_healthy(self):
        """Health check should return 'healthy' when provider is configured."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}):
            import importlib
            import app.core.tts_settings as ts_mod
            importlib.reload(ts_mod)
            from app.core.text_to_speech import TextToSpeech

            with patch.object(TextToSpeech, "_validate_env_vars"):
                tts = TextToSpeech.__new__(TextToSpeech)
                tts.provider = "openai"
                tts._client = None
                tts.openai_client = None

                with patch.object(tts, "is_available", return_value=True):
                    with patch.object(tts, "get_available_voices", new_callable=AsyncMock, return_value=[{"id": "nova", "name": "Nova"}]):
                        result = await tts.health_check()

                assert result["status"] == "healthy"
                assert "provider" in result
