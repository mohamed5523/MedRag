from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import uuid
import logging
from typing import List

from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore
from ..models.schemas import DocumentUploadResponse, DocumentInfo

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
document_processor = DocumentProcessor()
vector_store = VectorStore()

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a medical document.
    Supported formats: PDF, DOCX, TXT, MD
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not document_processor.is_supported_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported formats: {document_processor.get_supported_extensions()}"
            )
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file.filename} as {unique_filename}")
        
        # Process document
        try:
            raw_text = document_processor.load_document(file_path)
            
            if not raw_text.strip():
                raise ValueError("Document appears to be empty or unreadable")
            
            # Build vector index
            chunks_created = vector_store.build_index(raw_text, file.filename)
            
            processing_time = time.time() - start_time
            
            return DocumentUploadResponse(
                filename=unique_filename,
                status="processed",
                chunks_created=chunks_created,
                file_size=len(content),
                processing_time=processing_time
            )
            
        except Exception as e:
            # Clean up file on processing error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/list")
async def list_documents():
    """
    List all uploaded documents with their metadata.
    """
    try:
        documents = []
        
        # Get all files in upload directory
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                documents.append({
                    "filename": file_path.name,
                    "file_size": stat.st_size,
                    "upload_date": stat.st_mtime,
                    "file_type": file_path.suffix.upper().lstrip('.')
                })
        
        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x["upload_date"], reverse=True)
        
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and remove it from the vector store.
    """
    try:
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from vector store
        # Note: This requires knowing the original filename, which we'd need to store in metadata
        # For now, we'll just delete the file
        file_path.unlink()
        
        return {"message": f"Document {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/stats")
async def get_document_stats():
    """
    Get statistics about the document collection.
    """
    try:
        # Get vector store stats
        vector_stats = vector_store.get_collection_stats()
        
        # Get file system stats
        total_files = len(list(UPLOAD_DIR.iterdir()))
        total_size = sum(f.stat().st_size for f in UPLOAD_DIR.iterdir() if f.is_file())
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "vector_store": vector_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")