import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from opentelemetry import trace

from ..core.text_to_speech import TextToSpeech
from ..core.tts_exceptions import TextToSpeechError
from ..models.schemas import (
    TTSHealthResponse,
    TTSRequest,
    TTSResponse,
    VoiceListResponse,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.tts.api")

router = APIRouter()

# Initialize TTS service (will be None if credentials not configured)
try:
    tts_service = TextToSpeech()
    logger.info("TTS service initialized successfully")
except Exception as e:
    logger.warning(f"TTS service initialization failed: {e}")
    tts_service = None


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using the server-configured TTS provider.

    Returns metadata about the generated audio with base64 data included.
    """
    if not tts_service:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable or misconfigured",
        )

    try:
        with tracer.start_as_current_span("tts.synthesize_api") as span:
            span.set_attribute("text.length", len(request.text))
            audio_data = await tts_service.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                provider=None,  # enforce server-side provider
            )
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            return TTSResponse(
                success=True,
                audio_data=audio_base64,
                audio_size=len(audio_data),
                voice_used=request.voice_id or "default",
                text_length=len(request.text),
            )
    except TextToSpeechError as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected TTS error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audio")
async def get_audio_stream(text: str, voice_id: Optional[str] = None, provider: Optional[str] = None):
    """
    Get audio file stream for the given text.

    Returns:
        Audio file (MP3 format) as streaming response
    """
    if not tts_service:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable or misconfigured",
        )

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter is required")

    try:
        with tracer.start_as_current_span("tts.audio_stream_api") as span:
            span.set_attribute("text.length", len(text))
            audio_data = await tts_service.synthesize(text=text, voice_id=voice_id, provider=None)
            return Response(
                content=audio_data,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "inline; filename=speech.mp3",
                    "Cache-Control": "public, max-age=3600",
                },
            )
    except TextToSpeechError as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected TTS error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/voices", response_model=VoiceListResponse)
async def get_available_voices():
    """Get list of available voices."""
    if not tts_service:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable or misconfigured",
        )

    try:
        voices = await tts_service.get_available_voices()
        return VoiceListResponse(voices=voices, count=len(voices))
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available voices")


@router.get("/health", response_model=TTSHealthResponse)
async def health_check():
    """Check TTS service health."""
    if not tts_service:
        return TTSHealthResponse(status="unhealthy", provider="unknown", error="TTS service not initialized")

    try:
        health = await tts_service.health_check()
        return TTSHealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return TTSHealthResponse(status="unhealthy", provider="unknown", error=str(e))


