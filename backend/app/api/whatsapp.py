import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from ..core.qa_engine import QAEngine
from ..core.text_to_speech import TextToSpeech
from ..core.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


# WhatsApp API credentials (provided via environment)
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")


# Initialize core services (reuse existing QA/VectorStore/TTS)
vector_store = VectorStore()
qa_engine = QAEngine()
try:
    tts_service = TextToSpeech()
    logger.info("WhatsApp TTS service initialized successfully")
except Exception as e:
    logger.warning(f"TTS service not available for WhatsApp replies: {e}")
    tts_service = None


@router.get("/webhook/whatsapp/health")
async def whatsapp_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "has_token": bool(WHATSAPP_TOKEN),
        "has_phone_id": bool(WHATSAPP_PHONE_NUMBER_ID),
    }


@router.api_route("/webhook/whatsapp", methods=["GET", "POST"])
async def whatsapp_handler(request: Request) -> Response:
    """Handle WhatsApp webhook verification (GET) and messages (POST)."""

    # Verification handshake
    if request.method == "GET":
        params = request.query_params
        mode = params.get("hub.mode")
        verify_token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")

        if mode == "subscribe" and verify_token == WHATSAPP_VERIFY_TOKEN:
            return Response(content=challenge or "", status_code=200)
        return Response(content="Verification token mismatch", status_code=403)

    # Message handling
    try:
        body = await request.json()
        change = (
            body.get("entry", [{}])[0]
            .get("changes", [{}])[0]
            .get("value", {})
        )

        # Status callbacks — acknowledge
        if "statuses" in change:
            return Response(content="Status received", status_code=200)

        messages = change.get("messages")
        if not messages:
            return Response(content="No messages", status_code=200)

        message = messages[0]
        from_number = message.get("from")
        if not from_number:
            return Response(content="Missing sender", status_code=400)

        # Normalize user input
        user_text: Optional[str] = None
        msg_type = message.get("type")

        if msg_type == "text":
            user_text = message.get("text", {}).get("body", "")

        elif msg_type == "audio":
            audio_id = message.get("audio", {}).get("id")
            if audio_id:
                audio_bytes = await _download_media(audio_id)
                user_text = await _transcribe_audio_bytes(audio_bytes)
            else:
                user_text = ""

        elif msg_type == "image":
            # Use caption as input for now (vision analysis can be added later)
            user_text = message.get("image", {}).get("caption", "")

        else:
            # Unsupported types — acknowledge without processing
            return Response(content="Unsupported message type", status_code=200)

        user_text = (user_text or "").strip()
        if not user_text:
            await _send_text_message(from_number, "I received your message but could not parse any text.")
            return Response(content="Empty content handled", status_code=200)

        # Generate answer using existing RAG pipeline
        if not qa_engine.is_available():
            await _send_text_message(
                from_number,
                "AI service unavailable. Please ensure OPENAI_API_KEY is configured.",
            )
            return Response(content="Service unavailable", status_code=200)

        relevant_docs = vector_store.retrieve(query=user_text, top_k=5)
        if not relevant_docs:
            answer = (
                "I don't have any relevant information in my knowledge base to answer your question. "
                "Please try rephrasing your question or ensure that relevant documents have been uploaded."
            )
        else:
            result = qa_engine.answer_question(user_text, relevant_docs)
            answer = result.get("answer") or ""

        # DEMO: send both text and audio replies
        # 1) Send text
        await _send_text_message(from_number, answer or "(no answer generated)")

        # 2) Send audio (if TTS is available)
        if tts_service and answer:
            try:
                audio_bytes = await tts_service.synthesize(answer)
                await _send_audio_message(from_number, audio_bytes)
            except Exception as e:
                logger.warning(f"Failed to generate/send audio reply: {e}")

        return Response(content="Message processed", status_code=200)

    except Exception as e:  # pragma: no cover
        logger.exception("WhatsApp webhook error")
        return Response(content=f"Internal server error: {e}", status_code=500)


async def _download_media(media_id: str) -> bytes:
    """Download media bytes from WhatsApp Graph API using media ID."""
    if not WHATSAPP_TOKEN:
        raise HTTPException(status_code=503, detail="WHATSAPP_TOKEN not configured")

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient() as client:
        meta_resp = await client.get(f"https://graph.facebook.com/v21.0/{media_id}", headers=headers)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        url = meta.get("url")
        if not url:
            raise HTTPException(status_code=500, detail="Media URL not found")

        data_resp = await client.get(url, headers=headers)
        data_resp.raise_for_status()
        return data_resp.content


async def _upload_media(media_bytes: bytes, mime_type: str) -> str:
    """Upload media to WhatsApp and return the media ID."""
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID):
        raise HTTPException(status_code=503, detail="WhatsApp credentials not configured")

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("media.bin", BytesIO(media_bytes), mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files,
            data=data,
        )
        resp.raise_for_status()
        result = resp.json()

    media_id = result.get("id")
    if not media_id:
        raise HTTPException(status_code=500, detail="Failed to upload media")
    return media_id


async def _send_text_message(to_number: str, body: str) -> None:
    """Send a text message via WhatsApp."""
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID):
        raise HTTPException(status_code=503, detail="WhatsApp credentials not configured")

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    json_data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body},
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=json_data,
        )
        # Log but do not raise to avoid blocking flows
        if resp.status_code >= 300:
            logger.warning("Failed to send text message: %s - %s", resp.status_code, resp.text)


async def _send_audio_message(to_number: str, audio_bytes: bytes) -> None:
    """Send an audio message via WhatsApp by uploading, then referencing media ID."""
    media_id = await _upload_media(audio_bytes, "audio/mpeg")

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    json_data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "audio",
        "audio": {"id": media_id},
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=json_data,
        )
        if resp.status_code >= 300:
            logger.warning("Failed to send audio message: %s - %s", resp.status_code, resp.text)


async def _transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe audio using Groq Whisper similar to ASR module (plain text)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""  # Gracefully return empty; upstream will handle

    try:
        from groq import Groq  # lazy import
    except Exception:
        return ""

    try:
        client = Groq(api_key=api_key)
        buf = BytesIO(audio_bytes)
        buf.name = "audio.webm"  # WhatsApp often uses Opus/WebM; Groq handles various inputs
        result = client.audio.transcriptions.create(
            model=os.getenv("ASR_MODEL", "whisper-large-v3"),
            file=buf,  # type: ignore[arg-type]
            response_format="text",
        )
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        return text or ""
    except Exception as e:  # pragma: no cover
        logger.warning(f"Audio transcription failed: {e}")
        return ""


