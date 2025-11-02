import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        from langchain.docstore.document import Document
        
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.vector_store")

class VectorStore:
    """
    Manages document indexing and retrieval using ChromaDB and HuggingFace embeddings.
    """
    
    def __init__(self, persist_dir: str = None, embed_model: str = "intfloat/multilingual-e5-base"):
        # Allow override via env var; default to ./data/chroma_db
        persist_root = os.getenv("CHROMA_DB_PATH") or persist_dir or "data/chroma_db"
        self.persist_dir = str(Path(persist_root).absolute())
        self.embed_model = embed_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.collection_name = "medrag_collection"
        
        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    def build_index(self, raw_text: str, source: str) -> int:
        """
        Split text into chunks, embed, and store in ChromaDB.
        Returns the number of chunks created.
        """
        try:
            with tracer.start_as_current_span("split_text") as span:
                chunks = self.text_splitter.split_text(raw_text)
                span.set_attribute("chunks.count", len(chunks))
                logger.info(f"Split document {source} into {len(chunks)} chunks")
            
            # Create documents with metadata
            docs = [
                Document(
                    page_content=f"passage: {chunk}", 
                    metadata={"source": source, "chunk_id": i}
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Get or create vector store
            with tracer.start_as_current_span("index_documents") as span:
                span.set_attribute("docs.count", len(docs))
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir,
                )
                
                # Add documents to vector store
                vector_store.add_documents(docs)
            logger.info(f"Successfully indexed {len(docs)} chunks from {source}")
            
            return len(docs)
            
        except Exception as e:
            logger.error(f"Error building index for {source}: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve top-k most relevant chunks from ChromaDB.
        """
        try:
            with tracer.start_as_current_span("similarity_search") as span:
                span.set_attribute("query.length", len(query))
                span.set_attribute("top_k", top_k)
                span.set_attribute("embed_model", self.embed_model)
                span.add_event("search.started", {"timestamp": datetime.now().isoformat()})
                
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir,
                )
                
                # Search with query prefix for better retrieval
                docs = vector_store.similarity_search(f"query: {query}", k=top_k)
                
                span.set_attribute("results.count", len(docs))
                if docs:
                    # Add source information
                    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
                    span.set_attribute("sources.count", len(set(sources)))
                    span.set_attribute("unique_sources", list(set(sources))[:5])  # Limit to 5
                
                span.add_event("search.completed", {
                    "documents_found": len(docs),
                    "timestamp": datetime.now().isoformat()
                })
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {str(e)}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
            with tracer.start_as_current_span("get_collection_stats"):
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir,
                )
                
                # Get collection info
                collection = vector_store._collection
                count = collection.count()
                
                return {
                    "total_documents": count,
                    "collection_name": self.collection_name,
                    "embed_model": self.embed_model
                }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "embed_model": self.embed_model,
                "error": str(e)
            }
    
    def delete_documents_by_source(self, source: str) -> bool:
        """Delete all documents from a specific source."""
        try:
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            
            # Get documents with the specified source
            docs = vector_store.get(where={"source": source})
            
            if docs and docs.get('ids'):
                vector_store.delete(ids=docs['ids'])
                logger.info(f"Deleted {len(docs['ids'])} documents from source: {source}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting documents from source {source}: {str(e)}")
            return False