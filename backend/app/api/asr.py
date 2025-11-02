import io
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from opentelemetry import trace

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

load_dotenv()

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.asr")

router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = "whisper-large-v3",
    save: bool = False,
    session_id: str | None = None,
):
    """
    Transcribe an uploaded audio file using Groq Whisper.

    Accepts common browser-recorded formats like WebM/Opus and WAV.
    Returns a JSON object with `transcribed_text`.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ASR service unavailable: GROQ_API_KEY not configured",
        )

    if Groq is None:
        raise HTTPException(status_code=500, detail="Groq SDK not installed")

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

        client = Groq(api_key=api_key)

        # Prepare a file-like object for the SDK
        buf = io.BytesIO(data)
        buf.name = audio_file.filename or "audio.webm"

        with tracer.start_as_current_span("groq.transcribe") as span:
            span.set_attribute("model", model)
            # Ask for plain text to simplify handling
            result = client.audio.transcriptions.create(
                model=model,
                file=buf,  # type: ignore[arg-type]
                response_format="text",
            )

        # The SDK may return a plain string for response_format="text"
        if isinstance(result, str):
            text = result
        else:
            # Fallbacks for object/dict-like responses
            text = getattr(result, "text", None) or (
                result.get("text") if isinstance(result, dict) else None
            )
            if text is None:
                text = str(result)

        return JSONResponse({"transcribed_text": text, "model": model})

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


