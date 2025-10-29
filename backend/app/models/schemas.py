from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    tts_provider: Optional[str] = None  # Optional override for /query-with-voice

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    context_count: int
    model_used: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    chunks_created: int
    file_size: int
    processing_time: float
    error: Optional[str] = None

class DocumentInfo(BaseModel):
    id: int
    filename: str
    original_name: str
    file_size: int
    file_type: str
    status: str
    chunks_count: int
    upload_date: datetime
    processing_time: Optional[float] = None

class AnalyticsResponse(BaseModel):
    total_documents: int
    total_queries: int
    avg_response_time: float
    success_rate: float
    popular_queries: List[dict]
    document_stats: dict

class HealthResponse(BaseModel):
    status: str
    service: str
    vector_store_status: dict
    qa_engine_status: dict
    timestamp: datetime

# ----------------- TTS Schemas -----------------

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    provider: Optional[str] = None  # Ignored; provider is server-side configured (supports "openai" | "azure" | "elevenlabs")

class TTSResponse(BaseModel):
    success: bool
    audio_data: Optional[str] = None  # base64 encoded audio
    audio_size: int
    voice_used: str
    text_length: int
    error: Optional[str] = None


class TTSHealthResponse(BaseModel):
    status: str
    provider: str
    available_voices: Optional[int] = None
    error: Optional[str] = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: str


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]
    count: int


class ChatResponseWithAudio(BaseModel):
    answer: str
    sources: List[str]
    context_count: int
    model_used: str
    tokens_used: Optional[int] = None
    audio_data: Optional[str] = None
    audio_size: Optional[int] = None
    has_audio: bool = False
    error: Optional[str] = None