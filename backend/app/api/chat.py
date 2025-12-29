import base64
import logging
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from opentelemetry import trace

from ..core.conversation_controller import (
    apply_pending_action_resolution,
    format_provider_disambiguation_prompt,
    infer_specialty_from_symptoms,
    is_symptom_triage_request,
    resolve_pending_action,
)
from ..core.conversation_memory import short_term_memory
from ..core.intent_router import RouteMode, route_conversation
from ..core.qa_engine import QAEngine
from ..core.session_manager import session_manager
from ..core.state_manager import state_manager
from ..core.text_to_speech import TextToSpeech
from ..core.vector_store import VectorStore
from ..integrations.mcp_client import MCPClientError
from ..models.schemas import ChatRequest, ChatResponse, ChatResponseWithAudio
from ..services.clinic_workflow import ClinicWorkflowService, MCPWorkflowError

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.chat")
ARABIC_HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA_ARABIC", "0.75"))
ENABLE_PENDING_ACTION = os.getenv("ENABLE_PENDING_ACTION", "true").strip().lower() not in {"0", "false", "no"}
ENABLE_SYMPTOM_TRIAGE = os.getenv("ENABLE_SYMPTOM_TRIAGE", "true").strip().lower() not in {"0", "false", "no"}

router = APIRouter()

# Initialize components (lazy-init vector store to avoid blocking startup on HF model download)
_vector_store: Optional[VectorStore] = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

qa_engine = QAEngine()
try:
    tts_service = TextToSpeech()
    logger.info("TTS service initialized successfully")
except Exception as e:
    logger.warning(f"TTS service not available: {e}")
    tts_service = None

clinic_workflow = ClinicWorkflowService()

def _materialize_intent_query(intent: str, *, doctor_name: str | None = None, clinic_name: str | None = None) -> str:
    """Create a full query string from a sticky intent + resolved entities."""
    doctor = (doctor_name or "").strip()
    clinic = (clinic_name or "").strip()
    if intent == "ask_price":
        if doctor and clinic:
            return f"سعر الكشف عند الدكتور {doctor} في {clinic}"
        if doctor:
            return f"سعر الكشف عند الدكتور {doctor}"
        if clinic:
            return f"سعر الكشف في {clinic}"
        return "سعر الكشف"
    if intent in {"check_availability", "book_appointment"}:
        if doctor and clinic:
            return f"مواعيد الدكتور {doctor} في {clinic}"
        if doctor:
            return f"مواعيد الدكتور {doctor}"
        if clinic:
            return f"مواعيد {clinic}"
        return "المواعيد"
    if intent == "list_doctors":
        if clinic:
            return f"اسماء الأطباء في {clinic}"
        return "اسماء الأطباء"
    # fallback
    if doctor:
        return f"{intent} عند الدكتور {doctor}"
    return intent or "استفسار"


def _format_provider_disambiguation_prompt(candidates: list[dict]) -> str:
    return format_provider_disambiguation_prompt(candidates)


def _apply_pending_action_resolution(
    pending_action_type: str | None,
    resolution: dict,
    request_query: str,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str,
]:
    """
    Convert a resolved pending_action into forced intent/entity overrides and a stable state_input_query.

    This is a small helper so we can unit-test the mapping logic and avoid bugs where pending_action
    is cleared/None but later accessed.
    """
    return apply_pending_action_resolution(pending_action_type, resolution, request_query)


async def _maybe_generate_audio(text: str) -> tuple[Optional[str], Optional[int], bool]:
    """Synthesize audio if a TTS backend is available."""
    if not tts_service or not text:
        return None, None, False

    with tracer.start_as_current_span("synthesize_tts"):
        try:
            audio_bytes = await tts_service.synthesize(text)
            audio_data = base64.b64encode(audio_bytes).decode("utf-8")
            return audio_data, len(audio_bytes), True
        except Exception as exc:  # pragma: no cover - network/service failure
            logger.warning(f"TTS generation failed: {exc}")
            return None, None, False

@router.post("/query", response_model=ChatResponse)
async def query_documents(request: ChatRequest, x_session_id: Optional[str] = Header(None)):
    """
    Process a user query and return an AI-generated answer based on the document collection.
    All operations are traced with detailed spans visible in Phoenix.
    """
    # Create root span for the entire request
    with tracer.start_as_current_span("chat.query") as root_span:
        root_span.set_attribute("http.method", "POST")
        root_span.set_attribute("http.route", "/api/chat/query")
        root_span.set_attribute("query.text", request.query)
        root_span.set_attribute("query.length", len(request.query))
        root_span.set_attribute("max_results", request.max_results or 5)
        root_span.add_event("request.received", {"timestamp": datetime.now().isoformat()})
        
        try:
            start_time = time.time()
            session_id = session_manager.get_or_create(x_session_id)
            # Add user message to Redis
            short_term_memory.add_message(session_id, "user", request.query)

            # Retrieve conversation history
            history = short_term_memory.get_messages(session_id, limit=5)
            history_ctx = short_term_memory.get_formatted_context(session_id, last_n=5)
            root_span.set_attribute("conversation.length", len(history))
            root_span.set_attribute("conversation.has_context", bool(history_ctx))
            # Log a concise view of the history and the context string used for rewriting
            try:
                history_preview = [
                    {
                        "role": msg.role,
                        "content": (msg.content[:200] + "...") if len(msg.content) > 200 else msg.content,
                        "ts": msg.timestamp,
                    }
                    for msg in history
                ]
                logger.info(
                    "Session %s history (%d msgs): %s",
                    session_id[:16] + "...",
                    len(history_preview),
                    history_preview,
                )
                if history_ctx:
                    logger.info(
                        "Session %s history_ctx used for rewrite (len=%d): %s",
                        session_id[:16] + "...",
                        len(history_ctx),
                        history_ctx[:1000] + ("..." if len(history_ctx) > 1000 else ""),
                    )
            except Exception as _log_err:
                logger.debug("Unable to log history preview: %s", _log_err)
            
            # Extract and update conversation state
            previous_state = short_term_memory.get_state(session_id)
            
            # Prepare history for state extraction (reverse order to be chronological if needed, but the extractor expects list of dicts)
            # ShortTermMemoryStore.get_messages returns chronological order (oldest first)?
            # Let's check: get_messages says "Get messages oldest-first"
            # So 'history' is oldest -> newest. That's good.
            history_dicts = [{"role": m.role, "content": m.content} for m in history]
            
            with tracer.start_as_current_span("extract_state") as span:
                new_state = state_manager.extract_state(request.query, history_dicts, previous_state)

                short_term_memory.save_state(session_id, new_state)
                
                rewritten_query_from_state = state_manager.rewrite_query(new_state)
                
                span.set_attribute("state.intent", new_state.intent)
                span.set_attribute("state.entities.doctor", str(new_state.entities.doctor))
                span.set_attribute("query.rewritten_state", rewritten_query_from_state)
                
                logger.info(f"🧠 State Update: Intent={new_state.intent}, Entities={new_state.entities}")

            # Rewrite query to include explicit date hint for retrieval
            # We use the state-rewritten query as the base for the retrieval query
            query_with_context = rewritten_query_from_state
            retrieval_query = rewritten_query_from_state
            time_context = qa_engine.build_time_context(request.query)
            
            with tracer.start_as_current_span("rewrite_query") as span:
                # If we have context, prepend it in a compact form for retrieval and generation
                # Note: With state management, we might rely less on raw history concatenation,
                # but keeping it for date context or other subtleties is fine.
                # However, mixing the "state rewritten" query with raw history might be confusing.
                # Let's just use the state rewritten query + date hint for retrieval.
                
                rewritten_query, time_context = qa_engine.rewrite_query_with_date_hint(
                    query_with_context, time_context=time_context
                )
                span.set_attribute("query.rewritten", rewritten_query[:200])
                span.set_attribute("date_hint", time_context["date_hint"])
                span.set_attribute("timezone", time_context["tz_name"])
                span.set_attribute("query.retrieval.length", len(retrieval_query))
            # Log the final rewritten query that will influence retrieval and LLM prompting
            try:
                logger.info(
                    "Session %s rewritten_query (len=%d): %s",
                    session_id[:16] + "...",
                    len(rewritten_query),
                    rewritten_query[:1000] + ("..." if len(rewritten_query) > 1000 else ""),
                )
            except Exception as _log_err:
                logger.debug("Unable to log rewritten query: %s", _log_err)

            # Validate query
            with tracer.start_as_current_span("validate_input") as span:
                if not request.query.strip():
                    span.record_exception(ValueError("Empty query"))
                    raise HTTPException(status_code=400, detail="Query cannot be empty")
                span.add_event("validation.passed")
            
            # Check if QA engine is available
            with tracer.start_as_current_span("check_availability") as span:
                is_available = qa_engine.is_available()
                span.set_attribute("qa_engine.available", is_available)
                span.set_attribute("qa_engine.model", qa_engine.model)
                if not is_available:
                    span.record_exception(Exception("QA engine unavailable"))
                    raise HTTPException(
                        status_code=503, 
                        detail="AI service unavailable. Please ensure OPENAI_API_KEY is configured."
                    )
                span.add_event("availability.check.passed")
            
            # Retrieve relevant documents
            retrieve_start = time.time()
            with tracer.start_as_current_span("retrieve_documents") as span:
                span.set_attribute("query.length", len(request.query))
                span.set_attribute("top_k", request.max_results or 5)
                span.add_event("retrieval.started", {"timestamp": datetime.now().isoformat()})
                
                alpha_override = ARABIC_HYBRID_ALPHA if time_context.get("is_arabic") else None
                relevant_docs = _get_vector_store().retrieve(
                    query=retrieval_query,
                    top_k=request.max_results or 5,
                    alpha_override=alpha_override
                )
                
                retrieve_time = time.time() - retrieve_start
                span.set_attribute("results.count", len(relevant_docs))
                span.set_attribute("retrieval.duration_ms", retrieve_time * 1000)
                span.add_event("retrieval.completed", {
                    "duration_ms": retrieve_time * 1000,
                    "documents_found": len(relevant_docs)
                })
                
                if relevant_docs:
                    # Add source information
                    sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
                    span.set_attribute("sources.count", len(set(sources)))
            
            if not relevant_docs:
                root_span.set_attribute("response.no_documents", True)
                response_text = (
                    "I don't have any relevant information in my knowledge base to answer your question. Please try rephrasing your question or ensure that relevant documents have been uploaded."
                )
                # Save assistant response for continuity
                short_term_memory.add_message(session_id, "assistant", response_text)
                return ChatResponse(
                    answer=response_text,
                    sources=[],
                    context_count=0,
                    model_used=qa_engine.model,
                    tokens_used=0
                )
            
            # Generate answer using QA engine with time context and prior chat history
            answer_start = time.time()
            with tracer.start_as_current_span("generate_answer") as span:
                span.set_attribute("context.count", len(relevant_docs))
                span.set_attribute("model", qa_engine.model)
                span.set_attribute("timezone", time_context["tz_name"])
                # Prepare prior turns for LLM (oldest-first)
                history_payload = [
                    {"role": m.role, "content": m.content}
                    for m in history
                    if m.content
                ]
                span.add_event("answer.generation.started", {"timestamp": datetime.now().isoformat()})
                
                result = await qa_engine.answer_question(
                    question=rewritten_query,
                    contexts=relevant_docs,
                    time_context=time_context,
                    chat_history=history_payload,
                )

                answer_time = time.time() - answer_start
                span.set_attribute("answer.length", len(result.get("answer", "")))
                span.set_attribute("tokens_used", result.get("tokens_used", 0))
                span.set_attribute("sources.count", len(result.get("sources", [])))
                span.set_attribute("generation.duration_ms", answer_time * 1000)
                span.add_event("answer.generation.completed", {
                    "duration_ms": answer_time * 1000,
                    "tokens_used": result.get("tokens_used", 0)
                })
            
            # Save assistant reply for multi-turn continuity
            short_term_memory.add_message(session_id, "assistant", result.get("answer", ""))
            
            processing_time = time.time() - start_time
            root_span.set_attribute("total.duration_ms", processing_time * 1000)
            root_span.set_attribute("response.answer_length", len(result.get("answer", "")))
            root_span.add_event("request.completed", {
                "duration_ms": processing_time * 1000,
                "timestamp": datetime.now().isoformat()
            })
            
            # Force flush spans to Phoenix immediately (BatchSpanProcessor buffers by default)
            # This ensures traces appear in Phoenix UI without waiting for batch timeout
            try:
                current_span = trace.get_current_span()
                if current_span:
                    # Get the tracer provider and force flush
                    provider = trace.get_tracer_provider()
                    if hasattr(provider, 'force_flush'):
                        provider.force_flush(timeout_millis=1000)  # Wait up to 1 second
            except Exception as flush_error:
                logger.debug(f"Could not force flush spans: {flush_error}")
            
            logger.info(f"Processed query in {processing_time:.2f}s: {request.query[:50]}...")
            
            return ChatResponse(**result)
            
        except HTTPException:
            root_span.record_exception(Exception("HTTP Exception"))
            raise
        except Exception as e:
            root_span.record_exception(e)
            logger.error(f"Error processing query '{request.query}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/query-with-voice", response_model=ChatResponseWithAudio)
async def query_with_voice_response(request: ChatRequest, x_session_id: Optional[str] = Header(None)):
    """
    Process a user query and return AI-generated answer with automatic voice synthesis.
    Mirrors /query logic so multi-turn context, Redis history, and Phoenix tracing stay in sync.
    """
    with tracer.start_as_current_span("chat.query_with_voice") as root_span:
        root_span.set_attribute("http.method", "POST")
        root_span.set_attribute("http.route", "/api/chat/query-with-voice")
        root_span.set_attribute("query.text", request.query)
        root_span.set_attribute("query.length", len(request.query))
        root_span.set_attribute("max_results", request.max_results or 5)
        root_span.add_event("request.received", {"timestamp": datetime.now().isoformat()})

        try:
            start_time = time.time()
            session_id = session_manager.get_or_create(x_session_id)
            root_span.set_attribute("session.id", session_id[:16] + "...")

            # Add user message to short-term memory
            short_term_memory.add_message(session_id, "user", request.query)

            # Retrieve recent conversation history
            history = short_term_memory.get_messages(session_id, limit=5)
            history_ctx = short_term_memory.get_formatted_context(session_id, last_n=5)
            root_span.set_attribute("conversation.length", len(history))
            root_span.set_attribute("conversation.has_context", bool(history_ctx))

            # Log helpful previews for debugging observability
            try:
                history_preview = [
                    {
                        "role": msg.role,
                        "content": (msg.content[:200] + "...") if len(msg.content) > 200 else msg.content,
                        "ts": msg.timestamp,
                    }
                    for msg in history
                ]
                logger.info(
                    "Session %s history (%d msgs): %s",
                    session_id[:16] + "...",
                    len(history_preview),
                    history_preview,
                )
                if history_ctx:
                    logger.info(
                        "Session %s voice history_ctx (len=%d): %s",
                        session_id[:16] + "...",
                        len(history_ctx),
                        history_ctx[:1000] + ("..." if len(history_ctx) > 1000 else ""),
                    )
            except Exception as _log_err:
                logger.debug("Unable to log voice history preview: %s", _log_err)

            # Extract and update conversation state (Voice)
            previous_state = short_term_memory.get_state(session_id)
            history_dicts = [{"role": m.role, "content": m.content} for m in history]

            history_payload = [
                {"role": m.role, "content": m.content}
                for m in history
                if m.content
            ]

            # ------------------------------------------------------------------
            # Conversation Controller: pending_action (disambiguation / triage)
            # ------------------------------------------------------------------
            pending_action = None
            forced_intent: Optional[str] = None
            forced_doctor: Optional[str] = None
            forced_clinic: Optional[str] = None
            forced_specialty: Optional[str] = None
            state_input_query = request.query

            if ENABLE_PENDING_ACTION:
                pending_action = short_term_memory.get_pending_action(session_id)
                if pending_action:
                    root_span.set_attribute("pending_action.type", pending_action.get("type", ""))
                    root_span.set_attribute("pending_action.intent", pending_action.get("intent", ""))
                    pending_action_type = (str(pending_action.get("type") or "").strip() or None)

                    resolution = resolve_pending_action(request.query, pending_action)
                    if resolution:
                        short_term_memory.clear_pending_action(session_id)
                        root_span.set_attribute("pending_action.resolved", True)
                        (
                            forced_intent,
                            forced_doctor,
                            forced_clinic,
                            forced_specialty,
                            state_input_query,
                        ) = _apply_pending_action_resolution(
                            pending_action_type, resolution, request.query
                        )
                        
                        # Important: once resolved and cleared from storage, treat this request as no longer having
                        # a pending action so downstream logic (e.g. symptom triage detection) can run.
                        pending_action = None
                    else:
                        # Not resolved: re-prompt without re-running ambiguous MCP logic
                        turns_remaining = int(pending_action.get("turns_remaining") or 2) - 1
                        pending_action["turns_remaining"] = turns_remaining
                        if turns_remaining <= 0:
                            short_term_memory.clear_pending_action(session_id)
                            prompt = "ممكن تكتب الاسم كامل عشان أقدر أحدد الدكتور أو العيادة صح؟"
                        else:
                            short_term_memory.save_pending_action(session_id, pending_action)
                            if pending_action.get("type") == "provider_disambiguation":
                                prompt = _format_provider_disambiguation_prompt(
                                    list(pending_action.get("candidates") or [])
                                )
                            else:
                                prompt = "قولي الألم فين بالظبط وهل فيه سخونية أو قيء؟"

                        short_term_memory.add_message(session_id, "assistant", prompt)
                        audio_data, audio_size, has_audio = await _maybe_generate_audio(prompt)
                        return ChatResponseWithAudio(
                            answer=prompt,
                            sources=[],
                            context_count=0,
                            model_used=qa_engine.model,
                            tokens_used=0,
                            audio_data=audio_data,
                            audio_size=audio_size,
                            has_audio=has_audio,
                            error="pending_action",
                        )
            
            with tracer.start_as_current_span("extract_state") as span:
                new_state = state_manager.extract_state(state_input_query, history_dicts, previous_state)

                short_term_memory.save_state(session_id, new_state)
                
                rewritten_query_from_state = state_manager.rewrite_query(new_state)
                
                span.set_attribute("state.intent", new_state.intent)
                span.set_attribute("query.rewritten_state", rewritten_query_from_state)

            # Apply forced entities/intent from pending_action resolution (post-extract)
            if forced_intent:
                new_state.intent = forced_intent
            if forced_doctor:
                new_state.entities.doctor = forced_doctor
                new_state.target_entity_type = "doctor"
            if forced_clinic:
                new_state.entities.clinic = forced_clinic
                if new_state.target_entity_type == "unknown":
                    new_state.target_entity_type = "clinic"
            if forced_specialty:
                new_state.entities.specialty = forced_specialty

            # Symptom triage (Option A): route "اروح لمين" symptom queries to MCP list_doctors
            if ENABLE_SYMPTOM_TRIAGE and not pending_action and is_symptom_triage_request(request.query):
                specialty = infer_specialty_from_symptoms(request.query)
                if not specialty:
                    # Ask one clarifying question and keep a pending_action so next turn doesn't fall to RAG
                    triage_pending = {
                        "type": "symptom_triage",
                        "intent": "list_doctors",
                        "turns_remaining": 2,
                        "original_question": request.query,
                    }
                    short_term_memory.save_pending_action(session_id, triage_pending)
                    prompt = "تمام قولي الألم فين بالظبط وهل فيه سخونية أو قيء؟"
                    short_term_memory.add_message(session_id, "assistant", prompt)
                    audio_data, audio_size, has_audio = await _maybe_generate_audio(prompt)
                    return ChatResponseWithAudio(
                        answer=prompt,
                        sources=[],
                        context_count=0,
                        model_used=qa_engine.model,
                        tokens_used=0,
                        audio_data=audio_data,
                        audio_size=audio_size,
                        has_audio=has_audio,
                        error="symptom_triage_clarify",
                    )

                # Force list_doctors using inferred specialty
                new_state.intent = "list_doctors"
                new_state.entities.specialty = specialty
                new_state.entities.doctor = None  # avoid doctor-specific workflows
                new_state.target_entity_type = "clinic"

                # Update rewrite baseline so retrieval/generation sees consistent query
                rewritten_query_from_state = state_manager.rewrite_query(new_state)
                short_term_memory.save_state(session_id, new_state)

            decision = route_conversation(new_state, request.query)
            root_span.set_attribute("routing.intent", decision.intent)
            root_span.set_attribute("routing.mode", decision.mode.value)

            # Build time context and optionally prepend conversation history
            query_with_context = rewritten_query_from_state
            retrieval_query = rewritten_query_from_state
            time_context = qa_engine.build_time_context(request.query)
            
            with tracer.start_as_current_span("rewrite_query") as span:
                rewritten_query, time_context = qa_engine.rewrite_query_with_date_hint(
                    query_with_context, time_context=time_context
                )
                span.set_attribute("query.rewritten", rewritten_query[:200])
                span.set_attribute("date_hint", time_context["date_hint"])
                span.set_attribute("timezone", time_context["tz_name"])
                span.set_attribute("query.retrieval.length", len(retrieval_query))
            try:
                logger.info(
                    "Session %s voice rewritten_query (len=%d): %s",
                    session_id[:16] + "...",
                    len(rewritten_query),
                    rewritten_query[:1000] + ("..." if len(rewritten_query) > 1000 else ""),
                )
            except Exception as _log_err:
                logger.debug("Unable to log voice rewritten query: %s", _log_err)

            # Validate query
            with tracer.start_as_current_span("validate_input") as span:
                if not request.query.strip():
                    span.record_exception(ValueError("Empty query"))
                    raise HTTPException(status_code=400, detail="Query cannot be empty")
                span.add_event("validation.passed")

            # Ensure QA engine is online
            with tracer.start_as_current_span("check_availability") as span:
                is_available = qa_engine.is_available()
                span.set_attribute("qa_engine.available", is_available)
                span.set_attribute("qa_engine.model", qa_engine.model)
                if not is_available:
                    span.record_exception(Exception("QA engine unavailable"))
                    raise HTTPException(
                        status_code=503,
                        detail="AI service unavailable. Please ensure OPENAI_API_KEY is configured.",
                    )
                span.add_event("availability.check.passed")

            if decision.mode == RouteMode.MCP:
                try:
                    workflow_result = await clinic_workflow.run(
                        decision=decision,
                        state=new_state,
                        question=state_input_query,
                        qa_engine=qa_engine,
                        chat_history=history_payload,
                    )
                except (MCPWorkflowError, MCPClientError) as workflow_error:
                    fallback_reason = getattr(workflow_error, "reason", "mcp_client_error")

                    # Convert ambiguous/low-confidence provider match into a numbered disambiguation flow
                    if (
                        ENABLE_PENDING_ACTION
                        and isinstance(workflow_error, MCPWorkflowError)
                        and fallback_reason in {"provider_ambiguous", "provider_low_confidence"}
                        and getattr(workflow_error, "data", None)
                        and (workflow_error.data.get("candidates") or [])
                    ):
                        candidates = list(workflow_error.data.get("candidates") or [])
                        pending = {
                            "type": "provider_disambiguation",
                            "intent": decision.intent,
                            "turns_remaining": 2,
                            "original_question": state_input_query,
                            "candidates": candidates,
                        }
                        short_term_memory.save_pending_action(session_id, pending)
                        prompt = _format_provider_disambiguation_prompt(candidates)
                        short_term_memory.add_message(session_id, "assistant", prompt)
                        audio_data, audio_size, has_audio = await _maybe_generate_audio(prompt)
                        return ChatResponseWithAudio(
                            answer=prompt,
                            sources=[],
                            context_count=0,
                            model_used=qa_engine.model,
                            tokens_used=0,
                            audio_data=audio_data,
                            audio_size=audio_size,
                            has_audio=has_audio,
                            error=fallback_reason,
                        )

                    root_span.add_event(
                        "routing.mcp.failed",
                        {
                            "reason": fallback_reason,
                            "message": str(workflow_error),
                        },
                    )
                    root_span.set_attribute("routing.mode", "mcp.error")
                    logger.info("MCP workflow failed: %s", workflow_error)

                    error_message = str(workflow_error) or "معذرةً، حصل خطأ وأنا بحاول أجيب البيانات من نظام العيادات."
                    short_term_memory.add_message(session_id, "assistant", error_message)
                    audio_data, audio_size, has_audio = await _maybe_generate_audio(error_message)
                    processing_time = time.time() - start_time
                    root_span.set_attribute("total.duration_ms", processing_time * 1000)
                    root_span.set_attribute("response.answer_length", len(error_message))
                    root_span.set_attribute("response.has_audio", has_audio)
                    if audio_size is not None:
                        root_span.set_attribute("response.audio_size", audio_size)
                    root_span.add_event(
                        "request.completed",
                        {
                            "duration_ms": processing_time * 1000,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    return ChatResponseWithAudio(
                        answer=error_message,
                        sources=[],
                        context_count=0,
                        model_used=qa_engine.model,
                        tokens_used=None,
                        audio_data=audio_data,
                        audio_size=audio_size,
                        has_audio=has_audio,
                        error=fallback_reason,
                    )
                else:
                    # MCP succeeded - pure MCP-only path (no RAG enrichment)
                    root_span.add_event(
                        "routing.mcp.success",
                        {
                            "tool_count": len(workflow_result.tool_audit),
                        },
                    )
                    root_span.set_attribute("routing.mode", "mcp_only")

                    result_payload = workflow_result.qa_response
                    short_term_memory.add_message(session_id, "assistant", result_payload.get("answer", ""))
                    audio_data, audio_size, has_audio = await _maybe_generate_audio(result_payload.get("answer", ""))
                    processing_time = time.time() - start_time
                    root_span.set_attribute("total.duration_ms", processing_time * 1000)
                    root_span.set_attribute("response.answer_length", len(result_payload.get("answer", "")))
                    root_span.set_attribute("response.has_audio", has_audio)
                    if audio_size is not None:
                        root_span.set_attribute("response.audio_size", audio_size)

                    root_span.add_event(
                        "request.completed",
                        {
                            "duration_ms": processing_time * 1000,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    try:
                        current_span = trace.get_current_span()
                        if current_span:
                            provider = trace.get_tracer_provider()
                            if hasattr(provider, "force_flush"):
                                provider.force_flush(timeout_millis=1000)
                    except Exception as flush_error:
                        logger.debug(f"Could not force flush voice spans: {flush_error}")

                    return ChatResponseWithAudio(
                        **result_payload,
                        audio_data=audio_data,
                        audio_size=audio_size,
                        has_audio=has_audio,
                    )

            # Retrieve relevant documents (RAG-only path)
            retrieve_start = time.time()
            with tracer.start_as_current_span("retrieve_documents") as span:
                span.set_attribute("query.length", len(request.query))
                span.set_attribute("top_k", request.max_results or 5)
                span.add_event("retrieval.started", {"timestamp": datetime.now().isoformat()})

                alpha_override = ARABIC_HYBRID_ALPHA if time_context.get("is_arabic") else None
                relevant_docs = _get_vector_store().retrieve(
                    query=retrieval_query,
                    top_k=request.max_results or 5,
                    alpha_override=alpha_override
                )

                retrieve_time = time.time() - retrieve_start
                span.set_attribute("results.count", len(relevant_docs))
                span.set_attribute("retrieval.duration_ms", retrieve_time * 1000)
                span.add_event("retrieval.completed", {
                    "duration_ms": retrieve_time * 1000,
                    "documents_found": len(relevant_docs)
                })
                if relevant_docs:
                    sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
                    span.set_attribute("sources.count", len(set(sources)))

            if not relevant_docs:
                root_span.set_attribute("response.no_documents", True)
                response_text = (
                    "I don't have any relevant information in my knowledge base to answer your question. "
                    "Please try rephrasing your question or ensure that relevant documents have been uploaded."
                )

                short_term_memory.add_message(session_id, "assistant", response_text)

                audio_data, audio_size, has_audio = await _maybe_generate_audio(response_text)

                processing_time = time.time() - start_time
                root_span.set_attribute("total.duration_ms", processing_time * 1000)
                root_span.add_event("request.completed", {
                    "duration_ms": processing_time * 1000,
                    "timestamp": datetime.now().isoformat()
                })

                return ChatResponseWithAudio(
                    answer=response_text,
                    sources=[],
                    context_count=0,
                    model_used=qa_engine.model,
                    tokens_used=0,
                    audio_data=audio_data,
                    audio_size=audio_size,
                    has_audio=has_audio,
                )

            # Generate answer with QA engine
            answer_start = time.time()
            with tracer.start_as_current_span("generate_answer") as span:
                span.set_attribute("context.count", len(relevant_docs))
                span.set_attribute("model", qa_engine.model)
                span.set_attribute("timezone", time_context["tz_name"])
                span.add_event("answer.generation.started", {"timestamp": datetime.now().isoformat()})

                result = await qa_engine.answer_question(
                    question=rewritten_query,
                    contexts=relevant_docs,
                    time_context=time_context,
                    chat_history=history_payload,
                )

                answer_time = time.time() - answer_start
                span.set_attribute("answer.length", len(result.get("answer", "")))
                span.set_attribute("tokens_used", result.get("tokens_used", 0))
                span.set_attribute("sources.count", len(result.get("sources", [])))
                span.set_attribute("generation.duration_ms", answer_time * 1000)
                span.add_event("answer.generation.completed", {
                    "duration_ms": answer_time * 1000,
                    "tokens_used": result.get("tokens_used", 0)
                })

            # Persist assistant reply for continuity
            short_term_memory.add_message(session_id, "assistant", result.get("answer", ""))

            # Synthesize audio if possible
            audio_data, audio_size, has_audio = await _maybe_generate_audio(result.get("answer", ""))

            processing_time = time.time() - start_time
            root_span.set_attribute("total.duration_ms", processing_time * 1000)
            root_span.set_attribute("response.answer_length", len(result.get("answer", "")))
            root_span.set_attribute("response.has_audio", has_audio)
            if audio_size is not None:
                root_span.set_attribute("response.audio_size", audio_size)
            root_span.add_event("request.completed", {
                "duration_ms": processing_time * 1000,
                "timestamp": datetime.now().isoformat()
            })

            # Force flush spans so Phoenix shows voice trace immediately
            try:
                current_span = trace.get_current_span()
                if current_span:
                    provider = trace.get_tracer_provider()
                    if hasattr(provider, "force_flush"):
                        provider.force_flush(timeout_millis=1000)
            except Exception as flush_error:
                logger.debug(f"Could not force flush voice spans: {flush_error}")

            logger.info(
                f"Processed voice query in {processing_time:.2f}s: {request.query[:50]}..."
            )

            return ChatResponseWithAudio(
                **result,
                audio_data=audio_data,
                audio_size=audio_size,
                has_audio=has_audio,
            )

        except HTTPException:
            root_span.record_exception(Exception("HTTP Exception"))
            raise
        except Exception as e:
            root_span.record_exception(e)
            logger.error(f"Error processing voice query '{request.query}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/health")
async def chat_health():
    """
    Check the health of the chat service components.
    """
    try:
        vector_stats = _get_vector_store().get_collection_stats()
        qa_info = qa_engine.get_model_info()
        
        return {
            "status": "healthy" if qa_engine.is_available() else "degraded",
            "vector_store": vector_stats,
            "qa_engine": qa_info,
            "available_documents": vector_stats.get("total_documents", 0)
        }
        
    except Exception as e:
        logger.error(f"Error checking chat health: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/test")
async def test_query():
    """
    Test endpoint to verify the chat system is working.
    """
    test_request = ChatRequest(
        query="What medical information is available?",
        max_results=3
    )
    
    try:
        response = await query_documents(test_request)
        return {
            "test_status": "success",
            "response": response
        }
    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e)
        }


# ──────────────────────────────────────────────────────────────
# Session management endpoints
# ──────────────────────────────────────────────────────────────

@router.post("/session/clear")
async def clear_session_history(x_session_id: str = Header(...)):
    """Clear conversation history for a session (messages + state + pending actions)."""
    with tracer.start_as_current_span("session.clear_history") as span:
        span.set_attribute("session.id", x_session_id[:16] + "...")
        # Ensure a session meta exists so clients can immediately reuse the same session id.
        session_manager.get_or_create(x_session_id)
        short_term_memory.clear_session(x_session_id)
        return {"message": "Session history cleared", "session_id": x_session_id}


@router.post("/session/end")
async def end_session(x_session_id: str = Header(...)):
    """Delete a session completely (meta + memory)."""
    with tracer.start_as_current_span("session.end") as span:
        span.set_attribute("session.id", x_session_id[:16] + "...")
        ok = session_manager.delete(x_session_id)
        short_term_memory.clear_session(x_session_id)
        if ok:
            return {"message": "Session ended", "session_id": x_session_id}
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/session/history")
async def get_session_history(x_session_id: str = Header(...), limit: int = 10):
    """Get conversation history for a session (oldest-first)."""
    with tracer.start_as_current_span("session.get_history") as span:
        span.set_attribute("session.id", x_session_id[:16] + "...")
        # Ensure the session exists (if client generated a new session id in the browser).
        session_manager.get_or_create(x_session_id)
        msgs = short_term_memory.get_messages(x_session_id, limit=limit)
        history = [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in msgs]
        return {"session_id": x_session_id, "history": history, "total_messages": len(history)}


@router.post("/session/new")
async def new_session(x_session_id: Optional[str] = Header(None)):
    """
    Create a brand-new session id and optionally end the previous session (if provided).
    Useful for "New chat" UI flows.
    """
    with tracer.start_as_current_span("session.new") as span:
        if x_session_id:
            span.set_attribute("session.prev_id", x_session_id[:16] + "...")
            try:
                session_manager.delete(x_session_id)
                short_term_memory.clear_session(x_session_id)
            except Exception:
                logger.debug("Failed to cleanup previous session during /session/new", exc_info=True)

        new_id = session_manager.create_session()
        span.set_attribute("session.new_id", new_id[:16] + "...")
        return {"session_id": new_id}