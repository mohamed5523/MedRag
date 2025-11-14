import logging
import os
from datetime import datetime
from typing import List, Optional

import weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from opentelemetry import trace
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        from langchain.docstore.document import Document

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.vector_store")

class VectorStore:
    """
    Manages document indexing and retrieval using Weaviate with hybrid search and HuggingFace embeddings.
    """

    def __init__(self, weaviate_url: str = None, embed_model: str = "intfloat/multilingual-e5-base", hybrid_beta: float = 0.3):
        # Allow override via env var; default to http://localhost:8081
        self.weaviate_url = os.getenv("WEAVIATE_URL") or weaviate_url or "http://localhost:8081"
        self.embed_model = embed_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.collection_name = "MedragCollection"
        self.hybrid_beta = float(os.getenv("HYBRID_ALPHA", hybrid_beta))

        # Initialize Weaviate client
        self.client = weaviate.connect_to_local(
            host=self.weaviate_url.replace("http://", "").split(":")[0],
            port=int(self.weaviate_url.split(":")[-1]) if ":" in self.weaviate_url else 8081
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the Weaviate collection exists with proper configuration."""
        try:
            if not self.client.collections.exists(self.collection_name):
                self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="page_content", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="chunk_id", data_type=DataType.INT)
                    ]
                )
                # collection_config = Configure.collection(
                #     properties=[
                #         Property(name="page_content", data_type=DataType.TEXT),
                #         Property(name="source", data_type=DataType.TEXT),
                #         Property(name="chunk_id", data_type=DataType.INT),
                #     ],
                #     vector_config=[
                #         NamedVectorConfig(
                #             name="default",
                #             vectorizer=Configure.Vectorizer.none(),
                #             vector_index=Configure.VectorIndex.hnsw(),
                #         )
                #     ],
                # )

                # self.client.collections.create(name=self.collection_name, config=collection_config)
                logger.info(f"Created Weaviate collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def build_index(self, raw_text: str, source: str) -> int:
        """
        Split text into chunks, embed, and store in Weaviate.
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

            # Initialize Weaviate vector store
            with tracer.start_as_current_span("index_documents") as span:
                span.set_attribute("docs.count", len(docs))

                vector_store = WeaviateVectorStore(
                    client=self.client,
                    index_name=self.collection_name,
                    text_key="page_content",
                    embedding=self.embeddings,
                    attributes=["source", "chunk_id"]
                )

                # Add documents to vector store
                vector_store.add_documents(docs)
            logger.info(f"Successfully indexed {len(docs)} chunks from {source}")

            return len(docs)

        except Exception as e:
            logger.error(f"Error building index for {source}: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5, alpha_override: Optional[float] = None) -> List[Document]:
        """
        Retrieve top-k most relevant chunks from Weaviate using hybrid search with beta weighting.
        """
        try:
            with tracer.start_as_current_span("hybrid_search") as span:
                span.set_attribute("query.length", len(query))
                span.set_attribute("top_k", top_k)
                span.set_attribute("embed_model", self.embed_model)
                alpha = alpha_override if alpha_override is not None else self.hybrid_beta
                span.set_attribute("hybrid_alpha", alpha)
                span.add_event("search.started", {"timestamp": datetime.now().isoformat()})

                # Use direct Weaviate collection for full control over hybrid search
                collection = self.client.collections.get(self.collection_name)

                # Embed query using the same model as documents to avoid server-side vectorizer requirement
                query_text = f"query: {query}"
                query_vector = self.embeddings.embed_query(query_text)
                span.set_attribute("query.vector_dim", len(query_vector))

                # Perform hybrid search with beta parameter (alpha in Weaviate terms)
                # beta = 0.3 means 30% weight on vector search, 70% on keyword search
                response = collection.query.hybrid(
                    query=query_text,
                    alpha=alpha,
                    limit=top_k,
                    vector=query_vector,
                    return_metadata=MetadataQuery(score=True, explain_score=True),
                    return_properties=["page_content", "source", "chunk_id"]
                )

                # Convert Weaviate objects to Document objects
                docs = []
                for obj in response.objects or []:
                    docs.append(Document(
                        page_content=obj.properties.get("page_content", ""),
                        metadata={
                            "source": obj.properties.get("source", ""),
                            "chunk_id": obj.properties.get("chunk_id", 0),
                            "score": obj.metadata.score if obj.metadata else None
                        }
                    ))

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
        """Get statistics about the Weaviate collection."""
        try:
            with tracer.start_as_current_span("get_collection_stats"):
                collection = self.client.collections.get(self.collection_name)

                # Get collection info using aggregate query
                response = collection.aggregate.over_all(total_count=True)

                return {
                    "total_documents": response.total_count,
                    "collection_name": self.collection_name,
                    "embed_model": self.embed_model,
                    "hybrid_beta": self.hybrid_beta
                }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "embed_model": self.embed_model,
                "hybrid_beta": self.hybrid_beta,
                "error": str(e)
            }
    
    def delete_documents_by_source(self, source: str) -> bool:
        """Delete all documents from a specific source."""
        try:
            collection = self.client.collections.get(self.collection_name)

            # Delete documents with the specified source using batch delete
            response = collection.data.delete_many(
                where=Filter.by_property("source").equal(source)
            )

            deleted_count = getattr(response, "matches", 0)
            logger.info(f"Deleted {deleted_count} documents from source: {source}")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from source {source}: {str(e)}")
            return False

    def close(self):
        """Close the Weaviate client connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("Weaviate client connection closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {str(e)}")