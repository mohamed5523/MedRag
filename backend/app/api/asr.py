import asyncio
import io
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from opentelemetry import trace

from ..core.tts_settings import tts_settings

try:
    from elevenlabs import ElevenLabs
except Exception:  # pragma: no cover
    ElevenLabs = None  # type: ignore

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

load_dotenv()

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.asr")

router = APIRouter()


def _get_asr_provider() -> str:
    """Return the configured ASR provider (elevenlabs or groq)."""
    return tts_settings.ASR_PROVIDER


# ── ElevenLabs Scribe transcription ───────────────────────────────────────────

def _elevenlabs_transcribe(audio_data: bytes, filename: str) -> str:
    """Transcribe audio bytes using ElevenLabs Scribe (synchronous SDK call)."""
    if ElevenLabs is None:
        raise RuntimeError("elevenlabs SDK not installed")

    api_key = tts_settings.ELEVENLABS_API_KEY
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not configured")

    client = ElevenLabs(api_key=api_key)
    buf = io.BytesIO(audio_data)
    buf.name = filename

    result = client.speech_to_text.convert(
        file=buf,
        model_id=tts_settings.ELEVENLABS_ASR_MODEL,  # e.g. "scribe_v1"
        language_code="ar",  # optimize for Arabic
    )

    # SpeechToTextConvertResponse has a .text attribute
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return (text or "").strip()


# ── Groq Whisper transcription (fallback) ─────────────────────────────────────

def _groq_transcribe(audio_data: bytes, filename: str, model: str) -> str:
    """Transcribe audio bytes using Groq Whisper."""
    if Groq is None:
        raise RuntimeError("groq SDK not installed")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    client = Groq(api_key=api_key)
    buf = io.BytesIO(audio_data)
    buf.name = filename

    result = client.audio.transcriptions.create(
        model=model,
        file=buf,  # type: ignore[arg-type]
        response_format="text",
    )

    if isinstance(result, str):
        return result.strip()

    text = getattr(result, "text", None) or (
        result.get("text") if isinstance(result, dict) else None
    )
    return (text or "").strip()


# ── API endpoint ──────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = "whisper-large-v3",
    save: bool = False,
    session_id: str | None = None,
):
    """
    Transcribe an uploaded audio file.

    Uses ElevenLabs Scribe (default) or Groq Whisper based on ASR_PROVIDER setting.
    Accepts common browser-recorded formats like WebM/Opus and WAV.
    Returns a JSON object with `transcribed_text`.
    """
    provider = _get_asr_provider()

    try:
        with tracer.start_as_current_span("read_input") as span:
            data = await audio_file.read()
            span.set_attribute("filename", audio_file.filename or "unknown")
            span.set_attribute("size", len(data) if data else 0)
            if not data:
                raise HTTPException(status_code=400, detail="Empty audio file")

        # Optionally save input audio to disk under uploads/voice
        if save:
            with tracer.start_as_current_span("save_input") as span:
                uploads_dir = Path("uploads")
                voice_dir = uploads_dir / "voice"
                voice_dir.mkdir(parents=True, exist_ok=True)
                base_name = audio_file.filename or "recording.webm"
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                sid = session_id or "anon"
                safe_name = f"{sid}_{ts}_{base_name}"
                target_path = voice_dir / safe_name
                span.set_attribute("path", str(target_path))
                try:
                    with open(target_path, "wb") as f:
                        f.write(data)
                    logger.info("Saved input audio to %s", target_path)
                except Exception as ex:
                    logger.warning("Failed to save input audio: %s", ex)

        filename = audio_file.filename or "audio.webm"

        with tracer.start_as_current_span("asr.transcribe") as span:
            span.set_attribute("provider", provider)

            if provider == "elevenlabs":
                span.set_attribute("model", tts_settings.ELEVENLABS_ASR_MODEL)
                logger.info("[ASR] Transcribing with ElevenLabs Scribe (model=%s)", tts_settings.ELEVENLABS_ASR_MODEL)
                text = await asyncio.to_thread(_elevenlabs_transcribe, data, filename)
                used_model = tts_settings.ELEVENLABS_ASR_MODEL
            else:
                # Groq Whisper fallback
                span.set_attribute("model", model)
                logger.info("[ASR] Transcribing with Groq Whisper (model=%s)", model)
                text = await asyncio.to_thread(_groq_transcribe, data, filename, model)
                used_model = model

        logger.info("[ASR] Transcription result (%s): %s", provider, text[:100] if text else "(empty)")
        return JSONResponse({"transcribed_text": text, "model": used_model})

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
