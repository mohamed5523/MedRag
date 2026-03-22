import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file_encoding="utf-8")

    # Provider selection: "openai" (default), "azure", or "elevenlabs" (legacy)
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "openai").lower()

    # OpenAI TTS
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "nova")
    OPENAI_TTS_AUDIO_FORMAT: str = os.getenv("OPENAI_TTS_AUDIO_FORMAT", "mp3")

    # ElevenLabs (legacy — kept for backward compatibility)
    ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID: str | None = os.getenv("ELEVENLABS_VOICE_ID")
    ELEVENLABS_MODEL: str = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    ELEVENLABS_STABILITY: float = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
    ELEVENLABS_SIMILARITY_BOOST: float = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
    ELEVENLABS_STYLE: float = float(os.getenv("ELEVENLABS_STYLE", "0.4"))
    ELEVENLABS_OUTPUT_FORMAT: str = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

    # ASR (Speech-to-Text) provider: "groq" (default) or "elevenlabs" (legacy)
    ASR_PROVIDER: str = os.getenv("ASR_PROVIDER", "groq").lower()
    GROQ_WHISPER_MODEL: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    # Legacy ElevenLabs ASR model (kept for backward compatibility)
    ELEVENLABS_ASR_MODEL: str = os.getenv("ELEVENLABS_ASR_MODEL", "scribe_v1")

    # Azure Neural TTS (Cognitive Services Speech)
    AZURE_TTS_API_KEY: str | None = os.getenv("AZURE_TTS_API_KEY")
    AZURE_TTS_REGION: str | None = os.getenv("AZURE_TTS_REGION", "eastus")  # Azure region
    AZURE_TTS_ENDPOINT: str | None = os.getenv("AZURE_TTS_ENDPOINT")  # e.g. https://eastus.tts.speech.microsoft.com or full /cognitiveservices/v1
    AZURE_TTS_VOICE: str = os.getenv("AZURE_TTS_VOICE", "ar-EG-SalmaNeural")
    AZURE_TTS_OUTPUT_FORMAT: str = os.getenv(
        "AZURE_TTS_OUTPUT_FORMAT",
        "audio-24khz-48kbitrate-mono-mp3",
    )


tts_settings = TTSSettings()