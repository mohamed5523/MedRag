import logging
import os
import secrets
import time
from typing import Optional

from opentelemetry import trace

from .redis_client import redis_client

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.session")

class SessionManager:
    """Manages session lifecycle using Redis."""
    
    def __init__(self):
        self.session_ttl = int(os.getenv("SESSION_TTL_MINUTES", "60")) * 60
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create or refresh session."""
        if not session_id:
            session_id = f"session_{secrets.token_urlsafe(32)}"
        
        with tracer.start_as_current_span("session.create") as span:
            span.set_attribute("session.id", session_id[:16] + "...")
            key = f"session:meta:{session_id}"
            redis_client.set(key, {"created_at": time.time()}, ex=self.session_ttl)
            logger.info(f"✅ Session created: {session_id[:16]}...")
            return session_id
    
    def is_valid(self, session_id: str) -> bool:
        return redis_client.exists(f"session:meta:{session_id}")
    
    def get_or_create(self, provided_id: Optional[str]) -> str:
        """
        Return a valid session ID. If provided_id is valid, refresh its TTL and return it,
        otherwise create a new session and return the new ID.
        """
        if provided_id and self.is_valid(provided_id):
            key = f"session:meta:{provided_id}"
            # Refresh TTL and mark updated_at; keep original created_at if present
            redis_client.set(key, {"updated_at": time.time()}, ex=self.session_ttl)
            logger.debug(f"🔄 Session refreshed: {provided_id[:16]}...")
            return provided_id
        return self.create_session()
    
    def delete(self, session_id: str) -> bool:
        redis_client.delete(f"session:meta:{session_id}")
        return True
    
# Global instance
session_manager = SessionManager()
