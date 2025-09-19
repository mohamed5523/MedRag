from fastapi import APIRouter, HTTPException
import time
import logging

from ..core.vector_store import VectorStore
from ..core.qa_engine import QAEngine
from ..models.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
vector_store = VectorStore()
qa_engine = QAEngine()

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
        
        # Generate answer using QA engine
        result = qa_engine.answer_question(request.query, relevant_docs)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed query in {processing_time:.2f}s: {request.query[:50]}...")
        
        return ChatResponse(**result)
        
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