import logging
import time
import uuid
from pathlib import Path
from typing import List

from connect2supabase import supabase
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from opentelemetry import trace

from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore
from ..models.schemas import DocumentInfo, DocumentUploadResponse

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.documents")

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
        with tracer.start_as_current_span("validate_file"):
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
        with tracer.start_as_current_span("save_file") as span:
            content = await file.read()
            span.set_attribute("filename", file.filename)
            span.set_attribute("target", str(file_path))
            with open(file_path, "wb") as f:
                f.write(content)
        
        logger.info(f"Saved uploaded file: {file.filename} as {unique_filename}")
        
        # Process document
        try:
            with tracer.start_as_current_span("process_document"):
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

@router.post("/process-supabase")
async def process_supabase_document(payload: dict = Body(...)):
    """
    Download a document from Supabase Storage and index it into the vector DB.
    Expects: { id?: UUID, storage_path: str, filename?: str }
    """
    try:
        storage_path = payload.get("storage_path")
        if not storage_path:
            raise HTTPException(status_code=400, detail="storage_path is required")

        original_filename = payload.get("filename") or Path(storage_path).name
        doc_id = payload.get("id")

        # Download file bytes from Supabase Storage bucket 'documents'
        storage = supabase.storage.from_("documents")
        file_bytes = storage.download(storage_path)
        if not file_bytes:
            raise HTTPException(status_code=404, detail="File not found in storage")

        # Persist to local uploads for processing
        # Use a deterministic name to avoid collisions while keeping trace to storage_path
        local_name = f"supabase_{Path(storage_path).name}"
        file_path = UPLOAD_DIR / local_name
        with open(file_path, "wb") as f:
            # Some SDKs return a Response-like object; handle bytes or .content
            content = getattr(file_bytes, "content", file_bytes)
            f.write(content)

        # Extract text and index
        raw_text = document_processor.load_document(file_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Downloaded document is empty or unreadable")

        # Use storage_path as stable source key so we can delete by source later
        chunks_created = vector_store.build_index(raw_text, storage_path)

        # Mark as processed in Supabase if id or storage_path provided
        update_query = supabase.from_("documents").update({"processed": True})
        if doc_id:
            update_query = update_query.eq("id", doc_id)
        else:
            update_query = update_query.eq("storage_path", storage_path)
        update_query.execute()

        return {
            "status": "processed",
            "chunks_created": chunks_created,
            "storage_path": storage_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Supabase document {payload}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/delete-supabase")
async def delete_supabase_document(payload: dict = Body(...)):
    """
    Delete a Supabase-stored document: remove vectors (by storage_path), storage object, and DB row.
    Expects: { id?: UUID, storage_path?: str }
    """
    try:
        storage_path = payload.get("storage_path")
        doc_id = payload.get("id")
        if not storage_path and not doc_id:
            raise HTTPException(status_code=400, detail="id or storage_path is required")

        # Look up storage_path if only id provided
        if not storage_path and doc_id:
            resp = supabase.from_("documents").select("storage_path").eq("id", doc_id).single().execute()
            data = getattr(resp, 'data', None) or resp
            storage_path = (data or {}).get('storage_path')
        if not storage_path:
            raise HTTPException(status_code=404, detail="storage_path not found")

        # 1) Delete vectors from Chroma by source = storage_path
        try:
            vector_store.delete_documents_by_source(storage_path)
        except Exception as e:
            logger.warning(f"Vector deletion warning for {storage_path}: {e}")

        # 2) Delete file from Supabase Storage
        try:
            supabase.storage.from_("documents").remove([storage_path])
        except Exception as e:
            logger.warning(f"Storage deletion warning for {storage_path}: {e}")

        # 3) Delete DB row
        q = supabase.from_("documents").delete()
        if doc_id:
            q = q.eq("id", doc_id)
        else:
            q = q.eq("storage_path", storage_path)
        q.execute()

        return {"status": "deleted", "storage_path": storage_path}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting Supabase document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

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