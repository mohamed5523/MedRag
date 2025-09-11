from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

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