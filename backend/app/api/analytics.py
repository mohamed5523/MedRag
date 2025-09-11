from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime
from typing import Dict, List

from ..core.vector_store import VectorStore
from ..core.qa_engine import QAEngine
from ..models.schemas import AnalyticsResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
vector_store = VectorStore()
qa_engine = QAEngine()

# In-memory storage for analytics (in production, use a proper database)
query_log: List[Dict] = []
document_stats: Dict = {}

@router.get("/overview", response_model=AnalyticsResponse)
async def get_analytics_overview():
    """
    Get comprehensive analytics overview for the admin dashboard.
    """
    try:
        # Get vector store stats
        vector_stats = vector_store.get_collection_stats()
        
        # Calculate metrics from query log
        total_queries = len(query_log)
        successful_queries = len([q for q in query_log if q.get("success", False)])
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        # Calculate average response time
        response_times = [q.get("response_time", 0) for q in query_log if q.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Get popular queries (mock data for now)
        popular_queries = [
            {"query": "Available cardiologists", "count": 45},
            {"query": "Emergency room procedures", "count": 32},
            {"query": "Appointment scheduling", "count": 28},
            {"query": "Doctor specialties", "count": 21},
            {"query": "Hospital policies", "count": 18}
        ]
        
        return AnalyticsResponse(
            total_documents=vector_stats.get("total_documents", 0),
            total_queries=total_queries,
            avg_response_time=avg_response_time,
            success_rate=success_rate,
            popular_queries=popular_queries,
            document_stats=vector_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/queries")
async def get_query_logs(limit: int = 50):
    """
    Get recent query logs for monitoring.
    """
    try:
        # Return most recent queries
        recent_queries = query_log[-limit:] if len(query_log) > limit else query_log
        recent_queries.reverse()  # Most recent first
        
        return {
            "queries": recent_queries,
            "total": len(query_log),
            "showing": len(recent_queries)
        }
        
    except Exception as e:
        logger.error(f"Error getting query logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get query logs: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def get_system_health():
    """
    Get comprehensive system health information.
    """
    try:
        vector_stats = vector_store.get_collection_stats()
        qa_info = qa_engine.get_model_info()
        
        # Determine overall system status
        status = "healthy"
        if not qa_engine.is_available():
            status = "degraded"
        if vector_stats.get("total_documents", 0) == 0:
            status = "warning" if status == "healthy" else status
        
        return HealthResponse(
            status=status,
            service="medrag-api",
            vector_store_status=vector_stats,
            qa_engine_status=qa_info,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/log-query")
async def log_query(query_data: dict):
    """
    Log a query for analytics purposes.
    This would be called by the chat endpoint to track usage.
    """
    try:
        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query_data.get("query", ""),
            "response_time": query_data.get("response_time", 0),
            "success": query_data.get("success", False),
            "tokens_used": query_data.get("tokens_used", 0),
            "sources_count": query_data.get("sources_count", 0)
        }
        
        query_log.append(query_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(query_log) > 1000:
            query_log.pop(0)
        
        return {"message": "Query logged successfully"}
        
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to log query: {str(e)}")

@router.get("/stats")
async def get_detailed_stats():
    """
    Get detailed system statistics.
    """
    try:
        vector_stats = vector_store.get_collection_stats()
        
        # Query statistics
        query_stats = {
            "total_queries": len(query_log),
            "successful_queries": len([q for q in query_log if q.get("success", False)]),
            "failed_queries": len([q for q in query_log if not q.get("success", False)]),
            "avg_response_time": sum([q.get("response_time", 0) for q in query_log]) / len(query_log) if query_log else 0,
            "total_tokens_used": sum([q.get("tokens_used", 0) for q in query_log])
        }
        
        return {
            "vector_store": vector_stats,
            "queries": query_stats,
            "system": {
                "qa_engine_available": qa_engine.is_available(),
                "model": qa_engine.model
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")