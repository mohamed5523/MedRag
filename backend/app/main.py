import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import analytics, asr, chat, documents, tts, whatsapp

# Create FastAPI app
app = FastAPI(
    title="MedRAG API",
    description="Medical RAG system backend with document processing and AI chat",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React dev server
        "http://localhost:3000",  # Alternative React dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8080",  # Vite dev server (configured)
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
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medrag-api"}