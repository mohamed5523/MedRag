import logging
import os
from typing import Optional

from elevenlabs import ElevenLabs

from .tts_exceptions import TextToSpeechError
from .tts_settings import tts_settings

logger = logging.getLogger(__name__)


class TextToSpeech:
    """A class to handle text-to-speech conversion using ElevenLabs."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(self):
        """Initialize the TextToSpeech class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[ElevenLabs] = None

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    @property
    def client(self) -> ElevenLabs:
        """Get or create ElevenLabs client instance using singleton pattern."""
        if self._client is None:
            self._client = ElevenLabs(api_key=tts_settings.ELEVENLABS_API_KEY)
        return self._client

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        """Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID to use (defaults to configured voice)

        Returns:
            bytes: Audio data in MP3 format

        Raises:
            ValueError: If the input text is empty or too long
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:  # ElevenLabs typical limit
            raise ValueError("Input text exceeds maximum length of 5000 characters")

        try:
            # Use provided voice_id or fall back to configured voice
            selected_voice_id = voice_id or tts_settings.ELEVENLABS_VOICE_ID

            logger.info(f"Synthesizing speech for text: {text[:50]}... with voice: {selected_voice_id}")

            # Generate audio using ElevenLabs API
            # Note: LLM already handles Arabic dialect and tashkeel via prompt engineering
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=selected_voice_id,
                model_id=tts_settings.ELEVENLABS_MODEL,
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            logger.info(f"Successfully generated audio: {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {str(e)}")
            raise TextToSpeechError(
                f"Text-to-speech conversion failed: {str(e)}"
            ) from e

    def is_available(self) -> bool:
        """Check if TTS service is available."""
        try:
            return self.client is not None
        except Exception:
            return False

    async def get_available_voices(self) -> list[dict]:
        """Get list of available voices with Arabic support."""
        try:
            # Return voices that work well with Arabic
            return [
                {
                    "id": tts_settings.ELEVENLABS_VOICE_ID,
                    "name": "Kemitt",
                    "description": "Female Arabic voice",
                    "language": "Arabic",
                },
                {
                    "id": "IES4nrmZdUBHByLBde0P",
                    "name": "Haytham",
                    "description": "Male Arabic voice",
                    "language": "Arabic",
                },
                {
                    "id": "ocqVw6LVSdCxCra4XhMH",
                    "name": "Abdullah",
                    "description": "Currently configured default voice",
                    "language": "Arabic",
                },
                # Multilingual voices (good for Arabic)
                {
                    "id": "pNInz6obpgDQGcFmaJgB",
                    "name": "Adam",
                    "description": "Deep male voice - Good for Arabic",
                    "language": "Multi-language",
                },
                {
                    "id": "EXAVITQu4vr4xnSDxMaL",
                    "name": "Bella",
                    "description": "Warm female voice - Good for Arabic",
                    "language": "Multi-language",
                },
                {
                    "id": "ThT5KcBeYPX3keUQqHPh",
                    "name": "Dorothy",
                    "description": "Pleasant female voice",
                    "language": "Multi-language",
                },
                {
                    "id": "TxGEqnHWrfWFTfGW9XjX",
                    "name": "Josh",
                    "description": "Young male voice",
                    "language": "Multi-language",
                },
                {
                    "id": "VR6AewLTigWG4xSOukaG",
                    "name": "Arnold",
                    "description": "Crisp male voice",
                    "language": "Multi-language",
                },
                {
                    "id": "21m00Tcm4TlvDq8ikWAM",
                    "name": "Rachel",
                    "description": "Calm female voice",
                    "language": "Multi-language",
                },
                
            ]
        except Exception:
            return []

    async def health_check(self) -> dict:
        """Check the health status of the TTS service."""
        if not self.is_available():
            return {"status": "unhealthy", "error": "ElevenLabs API key not configured"}

        return {
            "status": "healthy",
            "provider": "elevenlabs",
            "available_voices": len(await self.get_available_voices()),
        }


