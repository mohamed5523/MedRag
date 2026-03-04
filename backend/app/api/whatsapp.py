import base64
import logging
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore[assignment]

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from ..core.conversation_controller import (
    apply_pending_action_resolution,
    apply_context_switch_rules,
    format_clinic_disambiguation_prompt,
    format_provider_disambiguation_prompt,
    infer_specialty_from_symptoms,
    is_symptom_triage_request,
    resolve_pending_action,
    should_abandon_pending_action,
)
from ..core.conversation_memory import short_term_memory
from ..core.intent_router import RouteMode, route_conversation
from ..core.qa_engine import QAEngine
from ..core.redis_client import redis_client
from ..core.session_manager import session_manager
from ..core.state_manager import state_manager
from ..core.text_to_speech import TextToSpeech
from ..core.vector_store import VectorStore
from ..services.clinic_workflow import ClinicWorkflowService, MCPWorkflowError
from ..integrations.mcp_client import MCPClientError

logger = logging.getLogger(__name__)

router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
ENABLE_PENDING_ACTION = os.getenv("ENABLE_PENDING_ACTION", "true").strip().lower() not in {"0", "false", "no"}
ENABLE_SYMPTOM_TRIAGE = os.getenv("ENABLE_SYMPTOM_TRIAGE", "true").strip().lower() not in {"0", "false", "no"}

# Initialize core services
_vector_store: VectorStore | None = None

def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

qa_engine = QAEngine()
clinic_workflow = ClinicWorkflowService()

try:
    tts_service = TextToSpeech()
    logger.info("WhatsApp TTS service initialized successfully")
except Exception as e:
    logger.warning(f"TTS service not available for WhatsApp replies: {e}")
    tts_service = None

# Helper functions for disambiguation prompts
def _format_provider_disambiguation_prompt(candidates: list[dict]) -> str:
    return format_provider_disambiguation_prompt(candidates)

def _format_clinic_disambiguation_prompt(candidates: list[dict]) -> str:
    return format_clinic_disambiguation_prompt(candidates)

def _format_provider_clinic_mismatch_prompt(
    candidates: list[dict],
    *,
    requested_clinic: str | None = None,
) -> str:
    """Build a prompt for provider-clinic mismatch."""
    requested = (requested_clinic or "").strip()
    items = []
    for c in candidates[:5]:
        if not isinstance(c, dict): continue
        name = (str(c.get("name_ar") or c.get("name_en") or "")).strip()
        clinic = (str(c.get("clinic_name") or "")).strip()
        if name: items.append((name, clinic))

    if not items:
        return "الاسم موجود لكن مش في العيادة اللي اتذكرت. ممكن تكتب اسم العيادة أو اسم الدكتور بالكامل؟"

    if len(items) == 1:
        name, clinic = items[0]
        if requested and clinic:
            return f"الاسم موجود، لكن مش في {requested}. الدكتور {name} موجود في عيادة {clinic}. اكتب 1 للتأكيد."
        return f"الدكتور {name} موجود في عيادة {clinic}. اكتب 1 للتأكيد."

    parts = []
    if requested: parts.append(f"ملقتش الدكتور في {requested}.")
    parts.append("لقيت دكاترة بنفس الاسم لكن في عيادات مختلفة. اختار رقم:")
    for i, (name, clinic) in enumerate(items, start=1):
        suffix = f" — {clinic}" if clinic else ""
        parts.append(f"{i} {name}{suffix}")
    parts.append("اكتب رقم الاختيار")
    return " ".join(parts).strip()

def _apply_pending_action_resolution_helper(
    pending_action_type: str | None,
    resolution: dict,
    request_query: str,
):
    return apply_pending_action_resolution(pending_action_type, resolution, request_query)


@router.get("/webhook/whatsapp/health")
async def whatsapp_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "has_token": bool(WHATSAPP_TOKEN),
        "has_phone_id": bool(WHATSAPP_PHONE_NUMBER_ID),
        "mcp_enabled": True
    }


async def process_whatsapp_message(message: Dict[str, Any], from_number: str) -> None:
    """
    Process WhatsApp message in background to avoid timeouts.
    Includes parsing, intent detection, MCP execution, and response generation.
    """
    try:
        # Session ID based on phone number (prefixed for namespace)
        session_id = f"wa:{from_number}"

        # Compute localized 'now'
        tz_name = os.getenv("DEFAULT_TZ", "Africa/Cairo")
        now_local = datetime.now(ZoneInfo(tz_name)) if ZoneInfo else datetime.now()

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
            user_text = message.get("image", {}).get("caption", "")
        else:
            logger.warning(f"Unsupported message type: {msg_type}")
            return

        user_text = (user_text or "").strip()
        if not user_text:
            await _send_text_message(from_number, "I received a message but couldn't understand it.")
            return

        # ---------------------------------------------------------
        # Core Chat Logic (Mirrors chat.py)
        # ---------------------------------------------------------

        # Add user message to memory
        short_term_memory.add_message(session_id, "user", user_text)

        # Retrieve history
        history = short_term_memory.get_messages(session_id, limit=5)
        history_dicts = [{"role": m.role, "content": m.content} for m in history]
        history_payload = [{"role": m.role, "content": m.content} for m in history if m.content]

        previous_state = short_term_memory.get_state(session_id)

        # --- Pending Action (Disambiguation) ---
        pending_action = None
        forced_intent = None
        forced_doctor = None
        forced_clinic = None
        forced_clinic_id = None
        forced_provider_id = None
        reset_doctor = False
        forced_specialty = None
        state_input_query = user_text

        if ENABLE_PENDING_ACTION:
            pending_action = short_term_memory.get_pending_action(session_id)
            if pending_action:
                pending_type = str(pending_action.get("type") or "").strip() or None
                resolution = resolve_pending_action(user_text, pending_action)
                
                if resolution:
                    selected = resolution.get("selected") if isinstance(resolution, dict) else None
                    if isinstance(selected, dict):
                        if pending_type == "clinic_disambiguation":
                            try:
                                forced_clinic_id = int(selected.get("clinic_id"))
                            except: pass
                            forced_clinic = (selected.get("clinic_name") or forced_clinic or "").strip() or None
                            reset_doctor = True
                        elif pending_type in {"provider_disambiguation", "provider_clinic_mismatch"}:
                            try:
                                forced_provider_id = int(selected.get("provider_id"))
                            except: pass
                            try:
                                forced_clinic_id = int(selected.get("clinic_id"))
                            except: pass

                    short_term_memory.clear_pending_action(session_id)
                    (
                        forced_intent,
                        forced_doctor,
                        forced_clinic,
                        forced_specialty,
                        state_input_query,
                    ) = _apply_pending_action_resolution_helper(pending_type, resolution, user_text)
                    pending_action = None
                else:
                     if should_abandon_pending_action(user_text, pending_action):
                         short_term_memory.clear_pending_action(session_id)
                         pending_action = None
                     else:
                         # Re-prompt
                         turns_remaining = int(pending_action.get("turns_remaining") or 2) - 1
                         pending_action["turns_remaining"] = turns_remaining
                         if turns_remaining <= 0:
                             short_term_memory.clear_pending_action(session_id)
                             prompt = "ممكن توضح الاسم أكتر؟"
                         else:
                             short_term_memory.save_pending_action(session_id, pending_action)
                             if pending_action.get("type") == "provider_disambiguation":
                                 prompt = _format_provider_disambiguation_prompt(list(pending_action.get("candidates") or []))
                             elif pending_action.get("type") == "provider_clinic_mismatch":
                                 prompt = _format_provider_clinic_mismatch_prompt(
                                     list(pending_action.get("candidates") or []),
                                     requested_clinic=str(pending_action.get("requested_clinic") or "").strip() or None,
                                 )
                             elif pending_action.get("type") == "clinic_disambiguation":
                                 prompt = _format_clinic_disambiguation_prompt(list(pending_action.get("candidates") or []))
                             else:
                                 prompt = "قولي التفاصيل إيه؟"
                        
                         await _send_response(from_number, prompt)
                         return

        # --- State Extraction ---
        new_state = state_manager.extract_state(state_input_query, history_dicts, previous_state)
        short_term_memory.save_state(session_id, new_state)

        # Apply forced entities
        if forced_intent: new_state.intent = forced_intent
        if forced_doctor: 
            new_state.entities.doctor = forced_doctor
            new_state.target_entity_type = "doctor"
        if forced_clinic: 
            new_state.entities.clinic = forced_clinic
            if new_state.target_entity_type == "unknown": new_state.target_entity_type = "clinic"
        if forced_clinic_id is not None: new_state.entities.clinic_id = forced_clinic_id
        if forced_provider_id is not None: new_state.entities.provider_id = forced_provider_id
        if reset_doctor:
            new_state.entities.doctor = None
            new_state.entities.provider_id = None
            if new_state.target_entity_type == "doctor": new_state.target_entity_type = "clinic"
        if forced_specialty: new_state.entities.specialty = forced_specialty

        apply_context_switch_rules(user_text, new_state.entities)

        # --- Symptom Triage ---
        if ENABLE_SYMPTOM_TRIAGE and not pending_action and is_symptom_triage_request(user_text):
            specialty = infer_specialty_from_symptoms(user_text)
            if not specialty:
                # Clarify
                triage_pending = {
                    "type": "symptom_triage",
                    "intent": "list_doctors",
                    "turns_remaining": 2,
                    "original_question": user_text,
                }
                short_term_memory.save_pending_action(session_id, triage_pending)
                prompt = "ممكن توصف الألم فين بالظبط؟"
                await _send_response(from_number, prompt)
                return
            
            new_state.intent = "list_doctors"
            new_state.entities.specialty = specialty
            new_state.entities.doctor = None
            new_state.target_entity_type = "clinic"
            short_term_memory.save_state(session_id, new_state)

        # --- Routing ---
        decision = route_conversation(new_state, user_text)
        
        # --- Execution ---
        if decision.mode == RouteMode.MCP:
            # MCP Flow
            try:
                result = await clinic_workflow.run(
                    decision=decision,
                    state=new_state,
                    question=state_input_query,
                    qa_engine=qa_engine,
                    chat_history=history_payload,
                    user_gender="male", # Default, can be improved to detect
                )
                result_payload = result.qa_response
                answer = result_payload.get("answer")
            except (MCPWorkflowError, MCPClientError) as e:
                # Handle Fallback / Disambiguation
                reason = getattr(e, "reason", "mcp_error")
                data = getattr(e, "data", {})
                
                if ENABLE_PENDING_ACTION and reason in {
                    "provider_ambiguous", "provider_low_confidence", 
                    "provider_clinic_mismatch", "clinic_ambiguous"
                } and data.get("candidates"):
                     # Disambiguation needed
                     candidates = data.get("candidates")
                     pending_type = "clinic_disambiguation" if reason == "clinic_ambiguous" else \
                                    "provider_clinic_mismatch" if reason == "provider_clinic_mismatch" else \
                                    "provider_disambiguation"
                     
                     pending = {
                         "type": pending_type,
                         "intent": decision.intent,
                         "turns_remaining": 2,
                         "original_question": state_input_query,
                         "candidates": candidates
                     }
                     if pending_type == "provider_clinic_mismatch":
                         pending["requested_clinic"] = str(data.get("requested_clinic") or "").strip()

                     short_term_memory.save_pending_action(session_id, pending)
                     
                     if pending_type == "clinic_disambiguation":
                         prompt = _format_clinic_disambiguation_prompt(candidates)
                     elif pending_type == "provider_clinic_mismatch":
                         prompt = _format_provider_clinic_mismatch_prompt(candidates, requested_clinic=pending.get("requested_clinic"))
                     else:
                         prompt = _format_provider_disambiguation_prompt(candidates)
                         
                     await _send_response(from_number, prompt)
                     return
                
                # General Failure
                logger.error(f"MCP Error: {e}")
                answer = "معلش، فيه مشكلة في الاتصال بنظام العيادات حاليًا. ممكن تجرب تاني بعد شوية؟"

        else:
            # RAG / General Flow
            if qa_engine.is_available():
                # Rewrite query for RAG
                time_context = qa_engine.build_time_context(user_text)
                rewrite_input_q = state_manager.rewrite_query(new_state)
                rewritten_q, _ = qa_engine.rewrite_query_with_date_hint(rewrite_input_q, time_context)
                
                relevant_docs = _get_vector_store().retrieve(query=rewritten_q, top_k=5)
                
                result = await qa_engine.answer_question(
                    question=rewritten_q,
                    contexts=relevant_docs,
                    time_context=time_context,
                    chat_history=history_payload,
                    user_gender="male"
                )
                answer = result.get("answer")
            else:
                answer = "AI service unavailable."

        # Send Final Answer
        await _send_response(from_number, answer or "(no answer)")
        
        # Save Assistant Reply
        short_term_memory.add_message(session_id, "assistant", answer or "")
        
    except Exception as e:
        logger.exception(f"Error processing WhatsApp message from {from_number}: {e}")
        # Optionally send generic error message to user, but avoiding loop is priority


@router.api_route("/webhook/whatsapp", methods=["GET", "POST"])
async def whatsapp_handler(request: Request, background_tasks: BackgroundTasks) -> Response:
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
        entries = body.get("entry", [])
        if not entries:
            return Response(content="No entries", status_code=200)

        changes = entries[0].get("changes", [])
        if not changes:
            return Response(content="No changes", status_code=200)

        value = changes[0].get("value", {})
        
        # Status callbacks — acknowledge
        if "statuses" in value:
            return Response(content="Status received", status_code=200)

        messages = value.get("messages")
        if not messages:
            return Response(content="No messages", status_code=200)

        message = messages[0]
        from_number = message.get("from")
        if not from_number:
            return Response(content="Missing sender", status_code=400)

        # Deduplication check
        message_id = message.get("id")
        if message_id:
            dedupe_key = f"wa:msg:{message_id}"
            if redis_client.exists(dedupe_key):
                logger.info(f"Duplicate WhatsApp message ignored: {message_id}")
                return Response(content="Duplicate ignored", status_code=200)
            
            # Mark as processed with 24h TTL
            redis_client.set(dedupe_key, "1", ex=86400)

        # Offload processing to background task using FastAPI BackgroundTasks
        background_tasks.add_task(process_whatsapp_message, message, from_number)
        
        # Immediately acknowledge receipt to WhatsApp to prevent timeout/retries
        return Response(content="Message queued", status_code=200)

    except Exception as e:
        # BUG FIX: Removed duplicate except Exception block that was unreachable dead code.
        # The original file had two identical `except Exception` clauses on the same try block.
        # Only the first handler ever executes; the second was silently ignored by Python.
        logger.exception("WhatsApp webhook error")
        return Response(content=f"Internal server error: {e}", status_code=500)


async def _send_response(to_number: str, text: str):
    """Send text and optionally audio."""
    await _send_text_message(to_number, text)
    if tts_service and text:
        try:
            audio_bytes = await tts_service.synthesize(text)
            await _send_audio_message(to_number, audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed: {e}")


# --- Low Level Helpers ---

async def _download_media(media_id: str) -> bytes:
    if not WHATSAPP_TOKEN: raise HTTPException(503, "Token missing")
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://graph.facebook.com/v21.0/{media_id}", headers=headers)
        r.raise_for_status()
        url = r.json().get("url")
        if not url: raise HTTPException(500, "Media URL missing")
        r_data = await client.get(url, headers=headers)
        r_data.raise_for_status()
        return r_data.content

async def _upload_media(media_bytes: bytes, mime_type: str) -> str:
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID): raise HTTPException(503, "Creds missing")
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("media.bin", BytesIO(media_bytes), mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers, files=files, data=data
        )
        r.raise_for_status()
        return r.json().get("id")

async def _send_text_message(to_number: str, body: str) -> None:
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID): return
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    json_data = {"messaging_product": "whatsapp", "to": to_number, "type": "text", "text": {"body": body}}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers, json=json_data
        )
        if response.status_code != 200:
            logger.error(f"WhatsApp text message send failed: {response.status_code} - {response.text}")

async def _send_audio_message(to_number: str, audio_bytes: bytes) -> None:
    try:
        media_id = await _upload_media(audio_bytes, "audio/mpeg")
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
        json_data = {"messaging_product": "whatsapp", "to": to_number, "type": "audio", "audio": {"id": media_id}}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
                headers=headers, json=json_data
            )
            if response.status_code != 200:
                logger.error(f"WhatsApp audio message send failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.warning(f"Audio send failed: {e}")

async def _transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe audio using ElevenLabs Scribe (default) or Groq Whisper based on ASR_PROVIDER."""
    from ..core.tts_settings import tts_settings

    provider = tts_settings.ASR_PROVIDER

    if provider == "elevenlabs":
        api_key = tts_settings.ELEVENLABS_API_KEY
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not configured, skipping transcription")
            return ""
        try:
            from elevenlabs import ElevenLabs as EL
            import io as _io

            client = EL(api_key=api_key)
            buf = _io.BytesIO(audio_bytes)
            buf.name = "audio.webm"
            result = client.speech_to_text.convert(
                file=buf,
                model_id=tts_settings.ELEVENLABS_ASR_MODEL,
                language_code="ar",
            )
            text = getattr(result, "text", None)
            if text is None and isinstance(result, dict):
                text = result.get("text")
            return (text or "").strip()
        except Exception as e:
            logger.warning(f"ElevenLabs transcription failed: {e}")
            return ""

    # Groq Whisper fallback
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
    except Exception as e:
        logger.warning(f"Audio transcription failed: {e}")
        return "" 
