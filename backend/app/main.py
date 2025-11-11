import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import analytics, asr, chat, documents, tts, whatsapp
from app.core.conversation_memory import ShortTermMemoryStore

# Import Redis-related components
from app.core.redis_client import RedisClient
from app.core.session_manager import SessionManager
from app.observability.phoenix import init_observability
from app.services.supabase_reindexer import reindex_all_supabase_documents

# Configure logging
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 🎯 GLOBAL REDIS INSTANCES (Created here, initialized on startup)
# ──────────────────────────────────────────────────────────────

# Redis client (lazy initialization)
redis_client = RedisClient()

# Session manager (uses redis_client)
session_manager = SessionManager()

# Conversation memory (uses redis_client)
short_term_memory = ShortTermMemoryStore(
    namespace=os.getenv("REDIS_NAMESPACE", "medrag"),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_MINUTES", "60")) * 60,
    max_messages=int(os.getenv("CONVERSATION_HISTORY_LENGTH", "20")),
    ttl_policy=os.getenv("TTL_POLICY", "sliding"),  # type: ignore
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup
    try:
        logger.info("🚀 Starting MedRAG Backend...")
        
        # Initialize Redis client (this establishes the actual connection)
        logger.info("🔌 Initializing Redis connection...")
        redis_client._initialize_client()
        
        # Verify Redis is working
        redis_client.client.ping()
        logger.info("✅ Redis connection verified")
        
        # Verify we can perform basic operations
        test_key = "medrag:startup:health"
        redis_client.set(test_key, "healthy", ex=30)
        value = redis_client.get(test_key)
        if value == "healthy":
            logger.info("✅ Redis write/read test passed")
        redis_client.delete(test_key)
        
        logger.info("🧠 Redis session management ready")
        logger.info(f"   Namespace: {short_term_memory.namespace}")
        logger.info(f"   TTL Policy: {short_term_memory.ttl_policy}")
        logger.info(f"   Max Messages: {short_term_memory.max_messages}")
        
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization issue: {e}")
        logger.info("   App will continue with FakeRedis for development")
    
    # Start background reindexing job (existing)
    logger.info("🔄 Starting background reindexing job...")
    asyncio.create_task(reindex_all_supabase_documents())
    
    logger.info("✅ Startup complete - Server ready for requests")
    
    yield  # App runs here
    
    # Shutdown
    logger.info("🛑 Shutting down MedRAG Backend...")
    
    # Close Redis connection if it's a real client
    try:
        if redis_client._client:  # Real Redis
            redis_client.client.close()
            logger.info("🔒 Redis connection closed")
    except Exception as e:
        logger.warning(f"⚠️ Error closing Redis: {e}")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="MedRAG API",
    description="Medical RAG system backend with document processing and AI chat",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize observability (Phoenix via OTLP)
init_observability(app)

# Make instances available to routers
# Option 1: Import directly from main (app.main.redis_client)
# Option 2: Re-export via app.core.__init__.py (recommended)


# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React dev server
        "http://localhost:3000",  # Alternative React dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8080",  # Vite dev server
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Mount static files for uploaded documents
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include API routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(asr.router, prefix="/api/asr", tags=["asr"])
app.include_router(tts.router, prefix="/api/tts", tags=["tts"])
app.include_router(whatsapp.router, tags=["whatsapp"])  # exposes /webhook/whatsapp


@app.get("/")
async def root():
    return {
        "message": "MedRAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "documents": "/api/documents",
            "chat": "/api/chat",
            "analytics": "/api/analytics",
            "asr": "/api/asr",
            "tts": "/api/tts"
        },
        "features": {
            "session_management": True,
            "redis_memory": True,
            "observability": True
        }
    }


@app.get("/health")
async def health_check():
    """Enhanced health check with Redis status."""
    redis_status = "healthy"
    
    try:
        # Check Redis connectivity
        if not redis_client._is_initialized:
            redis_status = "warning"
        else:
            redis_client.client.ping()
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "medrag-api",
        "redis": redis_status,
        "features": {
            "session_management": session_manager is not None,
            "conversation_memory": short_term_memory is not None,
            "observability": True
        }
    }


# Optional: Expose Redis instances for import elsewhere
# These can be imported directly from app.main in other modules
__all__ = ["app", "redis_client", "session_manager", "short_term_memory"]