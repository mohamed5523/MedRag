import base64
import logging
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from opentelemetry import trace

from ..core.conversation_memory import short_term_memory
from ..core.qa_engine import QAEngine
from ..core.session_manager import session_manager
from ..core.text_to_speech import TextToSpeech
from ..core.vector_store import VectorStore
from ..models.schemas import ChatRequest, ChatResponse, ChatResponseWithAudio

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.chat")

router = APIRouter()

# Initialize components
vector_store = VectorStore()
qa_engine = QAEngine()
try:
    tts_service = TextToSpeech()
    logger.info("TTS service initialized successfully")
except Exception as e:
    logger.warning(f"TTS service not available: {e}")
    tts_service = None

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
            # Rewrite query to include explicit date hint for retrieval
            rewritten_query = request.query
            time_context = qa_engine.build_time_context(request.query)
            with tracer.start_as_current_span("rewrite_query") as span:
                # If we have context, prepend it in a compact form for retrieval and generation
                if history_ctx:
                    rewritten_query = f"{history_ctx}\n\nCurrent user message: {request.query}"
                rewritten_query, time_context = qa_engine.rewrite_query_with_date_hint(
                    rewritten_query, time_context=time_context
                )
                span.set_attribute("query.rewritten", rewritten_query[:200])
                span.set_attribute("date_hint", time_context["date_hint"])
                span.set_attribute("timezone", time_context["tz_name"])
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
                
                relevant_docs = vector_store.retrieve(
                    query=rewritten_query,
                    top_k=request.max_results or 5
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
                
                result = qa_engine.answer_question(
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
async def query_with_voice_response(request: ChatRequest):
    """
    Process a user query and return AI-generated answer with automatic voice synthesis.
    """
    try:
        start_time = time.time()

        # Validate query
        with tracer.start_as_current_span("validate_input"):
            if not request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty")

        rewritten_query = request.query
        time_context = qa_engine.build_time_context(request.query)
        with tracer.start_as_current_span("rewrite_query") as span:
            rewritten_query, time_context = qa_engine.rewrite_query_with_date_hint(
                request.query, time_context=time_context
            )
            span.set_attribute("query.rewritten", rewritten_query[:200])
            span.set_attribute("date_hint", time_context["date_hint"])
            span.set_attribute("timezone", time_context["tz_name"])

        # Check if QA engine is available
        with tracer.start_as_current_span("check_availability"):
            if not qa_engine.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="AI service unavailable. Please ensure OPENAI_API_KEY is configured.",
                )

        # Retrieve relevant documents
        with tracer.start_as_current_span("retrieve_documents") as span:
            span.set_attribute("query.length", len(request.query))
            relevant_docs = vector_store.retrieve(
                query=rewritten_query, top_k=request.max_results or 5
            )

        if not relevant_docs:
            response_text = (
                "I don't have any relevant information in my knowledge base to answer your question. "
                "Please try rephrasing your question or ensure that relevant documents have been uploaded."
            )

            audio_data = None
            audio_size = None
            has_audio = False

            if tts_service:
                try:
                    audio_bytes = await tts_service.synthesize(response_text)
                    audio_data = base64.b64encode(audio_bytes).decode("utf-8")
                    audio_size = len(audio_bytes)
                    has_audio = True
                except Exception as e:
                    logger.warning(f"TTS generation failed: {e}")

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

        # Compute Egypt-local 'now' and pass to QA engine to ground relative dates
        # Generate answer using QA engine with time context
        with tracer.start_as_current_span("generate_answer") as span:
            span.set_attribute("context.count", len(relevant_docs))
            span.set_attribute("timezone", time_context["tz_name"])
            result = qa_engine.answer_question(
                request.query,
                relevant_docs,
                time_context=time_context,
            )

        audio_data = None
        audio_size = None
        has_audio = False

        if tts_service and result.get("answer"):
            with tracer.start_as_current_span("synthesize_tts"):
                try:
                    audio_bytes = await tts_service.synthesize(result["answer"])
                    audio_data = base64.b64encode(audio_bytes).decode("utf-8")
                    audio_size = len(audio_bytes)
                    has_audio = True
                except Exception as e:
                    logger.warning(f"TTS generation failed: {e}")

        processing_time = time.time() - start_time
        logger.info(
            f"Processed query with voice in {processing_time:.2f}s: {request.query[:50]}..."
        )

        return ChatResponseWithAudio(
            **result, audio_data=audio_data, audio_size=audio_size, has_audio=has_audio
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/health")
async def chat_health():
    """
    Check the health of the chat service components.
    """
    try:
        vector_stats = vector_store.get_collection_stats()
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
    """Clear conversation history for a session."""
    with tracer.start_as_current_span("session.clear_history") as span:
        span.set_attribute("session.id", x_session_id[:16] + "...")
        deleted = short_term_memory.clear_session(x_session_id)
        if deleted > 0:
            return {"message": "Session history cleared", "session_id": x_session_id}
        raise HTTPException(status_code=404, detail="Session not found")


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
        if not session_manager.is_valid(x_session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = short_term_memory.get_messages(x_session_id, limit=limit)
        history = [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in msgs]
        return {"session_id": x_session_id, "history": history, "total_messages": len(history)}