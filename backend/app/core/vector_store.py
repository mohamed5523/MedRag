import logging
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages document indexing and retrieval using ChromaDB and HuggingFace embeddings.
    """
    
    def __init__(self, persist_dir: str = "data/chroma_db", embed_model: str = "intfloat/multilingual-e5-base"):
        self.persist_dir = str(Path(persist_dir).absolute())
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
            # Split text into chunks
            chunks = self.text_splitter.split_text(raw_text)
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
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            
            # Search with query prefix for better retrieval
            docs = vector_store.similarity_search(f"query: {query}", k=top_k)
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {str(e)}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
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