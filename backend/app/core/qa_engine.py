import os
from typing import List
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import Document

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class QAEngine:
    """
    Question-Answering engine using OpenAI GPT models.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"QA Engine initialized with model: {model}")
    
    def answer_question(self, question: str, contexts: List[Document]) -> dict:
        """
        Generate an answer to the question using the provided contexts.
        Returns a dictionary with answer, sources, and metadata.
        """
        if not self.client:
            return {
                "answer": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                "sources": [],
                "error": "API key not configured"
            }
        
        try:
            # Extract text content from documents
            context_texts = [doc.page_content for doc in contexts]
            sources = [doc.metadata.get("source", "Unknown") for doc in contexts]
            
            # Create context string
            context_str = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(context_texts)])
            
            # Create system prompt for medical context
            system_prompt = (
                "You are a knowledgeable medical assistant. "
                "Answer the user's question based strictly on the provided medical documents and context. "
                "If the information is not available in the context, clearly state that you don't have enough information. "
                "Always prioritize accuracy and patient safety in your responses. "
                "If the question requires immediate medical attention, recommend consulting a healthcare professional."
            )
            
            # Create user message
            user_message = f"""
Question: {question}

Context from medical documents:
{context_str}

Please provide a comprehensive answer based on the available information.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for more consistent medical responses
                max_tokens=500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Get unique sources
            unique_sources = list(set(sources))
            
            return {
                "answer": answer,
                "sources": unique_sources,
                "context_count": len(contexts),
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if the QA engine is properly configured and available."""
        return self.client is not None
    
    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "api_configured": self.client is not None,
            "api_key_set": bool(self.api_key)
        }