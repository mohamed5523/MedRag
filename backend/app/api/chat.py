import base64
import logging
import os
import time
from datetime import datetime

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore[assignment]

from fastapi import APIRouter, HTTPException
from opentelemetry import trace

from ..core.qa_engine import QAEngine
from ..core.text_to_speech import TextToSpeech
from ..core.vector_store import VectorStore
from ..models.schemas import ChatRequest, ChatResponse, ChatResponseWithAudio

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
async def query_documents(request: ChatRequest):
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
                    query=request.query, 
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
                return ChatResponse(
                    answer="I don't have any relevant information in my knowledge base to answer your question. Please try rephrasing your question or ensure that relevant documents have been uploaded.",
                    sources=[],
                    context_count=0,
                    model_used=qa_engine.model,
                    tokens_used=0
                )
            
            # Compute Egypt-local 'now' and pass to QA engine to ground relative dates
            tz_name = os.getenv("DEFAULT_TZ", "Africa/Cairo")
            now_local = datetime.now(ZoneInfo(tz_name)) if ZoneInfo else datetime.now()

            # Generate answer using QA engine with time context
            answer_start = time.time()
            with tracer.start_as_current_span("generate_answer") as span:
                span.set_attribute("context.count", len(relevant_docs))
                span.set_attribute("model", qa_engine.model)
                span.set_attribute("timezone", tz_name)
                span.add_event("answer.generation.started", {"timestamp": datetime.now().isoformat()})
                
                result = qa_engine.answer_question(request.query, relevant_docs, now_dt=now_local)
                
                answer_time = time.time() - answer_start
                span.set_attribute("answer.length", len(result.get("answer", "")))
                span.set_attribute("tokens_used", result.get("tokens_used", 0))
                span.set_attribute("sources.count", len(result.get("sources", [])))
                span.set_attribute("generation.duration_ms", answer_time * 1000)
                span.add_event("answer.generation.completed", {
                    "duration_ms": answer_time * 1000,
                    "tokens_used": result.get("tokens_used", 0)
                })
            
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
                query=request.query, top_k=request.max_results or 5
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
        tz_name = os.getenv("DEFAULT_TZ", "Africa/Cairo")
        now_local = datetime.now(ZoneInfo(tz_name)) if ZoneInfo else datetime.now()

        # Generate answer using QA engine with time context
        with tracer.start_as_current_span("generate_answer") as span:
            span.set_attribute("context.count", len(relevant_docs))
            result = qa_engine.answer_question(request.query, relevant_docs, now_dt=now_local)

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