import base64
import logging
import os
import time
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore[assignment]

from fastapi import APIRouter, HTTPException

from ..core.qa_engine import QAEngine
from ..core.text_to_speech import TextToSpeech
from ..core.tts_exceptions import TextToSpeechError
from ..core.vector_store import VectorStore
from ..models.schemas import ChatRequest, ChatResponse, ChatResponseWithAudio

logger = logging.getLogger(__name__)

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
    """
    try:
        start_time = time.time()
        
        # Validate query
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if QA engine is available
        if not qa_engine.is_available():
            raise HTTPException(
                status_code=503, 
                detail="AI service unavailable. Please ensure OPENAI_API_KEY is configured."
            )
        
        # Retrieve relevant documents
        relevant_docs = vector_store.retrieve(
            query=request.query, 
            top_k=request.max_results or 5
        )
        
        if not relevant_docs:
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
        result = qa_engine.answer_question(request.query, relevant_docs, now_dt=now_local)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed query in {processing_time:.2f}s: {request.query[:50]}...")
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
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
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Check if QA engine is available
        if not qa_engine.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service unavailable. Please ensure OPENAI_API_KEY is configured.",
            )

        # Retrieve relevant documents
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
        result = qa_engine.answer_question(request.query, relevant_docs, now_dt=now_local)

        audio_data = None
        audio_size = None
        has_audio = False

        if tts_service and result.get("answer"):
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