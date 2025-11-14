"""Utilities for re-indexing Supabase documents into Weaviate."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

from connect2supabase import supabase

from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def reindex_all_supabase_documents(delay_seconds: float = 5.0) -> None:
    """Re-index every document stored in Supabase into Weaviate."""

    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    logger.info("🔁 Starting Supabase → Weaviate reindex job")

    try:
        response = supabase.from_("documents").select(
            "id, storage_path, filename"
        ).execute()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Supabase client unavailable or query failed. Skipping reindex job. Details: %s",
            exc,
        )
        return

    documents: list[Dict[str, Any]] = list(getattr(response, "data", None) or [])
    if not documents:
        logger.info("No documents found in Supabase to reindex.")
        return

    uploads_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
    uploads_dir.mkdir(exist_ok=True)

    processor = DocumentProcessor()
    vector_store = VectorStore()
    total_documents = len(documents)

    processed_count = 0
    storage_client = None

    for doc in documents:
        storage_path = doc.get("storage_path")
        if not storage_path:
            logger.warning("Skipping Supabase record missing storage_path: %s", doc)
            continue

        filename = doc.get("filename") or Path(storage_path).name

        try:
            if storage_client is None:
                storage_client = supabase.storage.from_("documents")
            file_bytes = storage_client.download(storage_path)
        except Exception as exc:  # pragma: no cover - depends on Supabase state
            logger.error("Failed to download %s from Supabase Storage: %s", storage_path, exc)
            continue

        content = getattr(file_bytes, "content", file_bytes)
        local_path = uploads_dir / f"reindex_{Path(storage_path).name}"

        try:
            with open(local_path, "wb") as f:
                f.write(content)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to write temporary file for %s: %s", storage_path, exc)
            continue

        try:
            vector_store.delete_documents_by_source(storage_path)
            raw_text = processor.load_document(local_path)
            if not raw_text.strip():
                logger.warning("Document %s appears empty after processing; skipping.", filename)
                continue
            vector_store.build_index(raw_text, storage_path)
            processed_count += 1
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to index %s: %s", filename, exc)
            continue
        finally:
            try:
                local_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                logger.debug("Could not delete temporary file %s", local_path)

        try:
            supabase.from_("documents").update({"processed": True}).eq("id", doc["id"]).execute()
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to update processed flag for %s in Supabase: %s",
                storage_path,
                exc,
            )

    try:
        vector_store.close()
    except Exception:  # pragma: no cover
        logger.debug("Error closing vector store client", exc_info=True)

    logger.info(
        "✅ Supabase reindex complete: %s/%s documents ingested into Weaviate",
        processed_count,
        total_documents,
    )


