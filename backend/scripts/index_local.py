#!/usr/bin/env python3
"""CLI helper to index local documents into Weaviate."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ensure backend package is importable
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    uploads_dir = Path(ROOT) / "uploads"
    if not uploads_dir.exists():
        logger.error(f"Uploads directory not found: {uploads_dir}")
        return

    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    files = [f for f in uploads_dir.iterdir() if f.is_file()]
    logger.info(f"Found {len(files)} files to index in {uploads_dir}")
    
    processed_count = 0
    for file_path in files:
        try:
            logger.info(f"Processing {file_path.name}...")
            # We use filename as source initially
            source_name = file_path.name
            
            # Delete old docs from this source first to avoid duplicates
            vector_store.delete_documents_by_source(source_name)
            
            raw_text = processor.load_document(file_path)
            if not raw_text.strip():
                logger.warning(f"Document {file_path.name} is empty, skipping.")
                continue
                
            vector_store.build_index(raw_text, source_name)
            processed_count += 1
            logger.info(f"✅ Successfully indexed {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to index {file_path.name}: {e}")

    vector_store.close()
    logger.info(f"🎉 Local indexing complete! {processed_count}/{len(files)} files indexed.")

if __name__ == "__main__":
    main()
