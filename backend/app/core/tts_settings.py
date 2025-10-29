import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file_encoding="utf-8")

    # Provider selection: "openai" (default), "azure", or "elevenlabs"
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "openai").lower()

    # OpenAI TTS
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "nova")
    OPENAI_TTS_AUDIO_FORMAT: str = os.getenv("OPENAI_TTS_AUDIO_FORMAT", "mp3")
    
    # ElevenLabs
    ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID: str | None = os.getenv("ELEVENLABS_VOICE_ID")
    ELEVENLABS_MODEL: str = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

    # Azure Neural TTS (Cognitive Services Speech)
    AZURE_TTS_API_KEY: str | None = os.getenv("AZURE_TTS_API_KEY")
    AZURE_TTS_ENDPOINT: str | None = os.getenv("AZURE_TTS_ENDPOINT")  # e.g. https://eastus.tts.speech.microsoft.com or full /cognitiveservices/v1
    AZURE_TTS_VOICE: str = os.getenv("AZURE_TTS_VOICE", "ar-EG-SalmaNeural")
    AZURE_TTS_OUTPUT_FORMAT: str = os.getenv(
        "AZURE_TTS_OUTPUT_FORMAT",
        "audio-24khz-48kbitrate-mono-mp3",
    )


tts_settings = TTSSettings()