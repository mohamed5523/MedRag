import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar, List, Literal, Optional, Tuple

from opentelemetry import trace

from .redis_client import redis_client
from .state_manager import ConversationState

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("medrag.conversation_memory")

TTLPolicy = Literal["sliding", "fixed"]


@dataclass
class MemoryMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float

    def to_json(self) -> str:
        return json.dumps({
            "role": self.role,
            "content": self.content,
            "ts": self.timestamp,
        }, ensure_ascii=False)

    @staticmethod
    def from_json(raw: str) -> "MemoryMessage":
        obj = json.loads(raw)
        return MemoryMessage(
            role=obj.get("role", "user"),
            content=obj.get("content", ""),
            timestamp=float(obj.get("ts", time.time())),
        )


class ShortTermMemoryStore:
    """
    Session-scoped short-term memory using Redis Lists + TTL.
    - Each session has a list of messages: LPUSH for newest-first, LTRIM to cap length
    - TTL ensures automatic expiry; use 'sliding' to refresh on each write, or 'fixed'
    - A small metadata hash tracks created_at/updated_at for observability
    """

    def __init__(
        self,
        namespace: str = "medrag",
        session_ttl_seconds: int = 3600,
        max_messages: int = 20,
        ttl_policy: TTLPolicy = "sliding",
    ) -> None:
        self.namespace = namespace.strip(":")
        self.session_ttl_seconds = int(session_ttl_seconds)
        self.max_messages = int(max_messages)
        self.ttl_policy: TTLPolicy = ttl_policy
        self.redis = redis_client.client

        # Process-local fallback for pending actions if Redis is temporarily unavailable.
        # This is best-effort only (won't survive restarts / multi-instance), but prevents
        # silent loss of disambiguation context during transient Redis issues.
        # {session_id: (expires_at_epoch_seconds, payload_dict)}
        if not hasattr(self.__class__, "_pending_action_local"):
            self.__class__._pending_action_local = {}

    _pending_action_local: ClassVar[dict[str, Tuple[float, dict[str, Any]]]]

    # Key helpers
    def _k_messages(self, session_id: str) -> str:
        return f"{self.namespace}:s:{session_id}:messages"

    def _k_meta(self, session_id: str) -> str:
        return f"{self.namespace}:s:{session_id}:meta"
        
    def _k_state(self, session_id: str) -> str:
        return f"{self.namespace}:s:{session_id}:state"

    def _k_pending_action(self, session_id: str) -> str:
        return f"{self.namespace}:s:{session_id}:pending_action"

    # Core operations
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
    ) -> int:
        """Add message to session and return current length."""
        now = time.time() if timestamp is None else float(timestamp)
        msg = MemoryMessage(role=role, content=content, timestamp=now)
        k_msgs = self._k_messages(session_id)
        k_meta = self._k_meta(session_id)

        with tracer.start_as_current_span("memory.add_message") as span:
            span.set_attribute("session.id", session_id[:16] + "...")
            span.set_attribute("message.role", role)
            span.set_attribute("message.length", len(content))
            
            try:
                # Use pipeline for atomicity
                pipe = self.redis.pipeline(transaction=True)
                pipe.lpush(k_msgs, msg.to_json())
                pipe.ltrim(k_msgs, 0, self.max_messages - 1)
                pipe.hsetnx(k_meta, "created_at", now)
                pipe.hset(k_meta, mapping={"updated_at": now})
                
                if self.ttl_policy == "sliding":
                    pipe.expire(k_msgs, self.session_ttl_seconds)
                    pipe.expire(k_meta, self.session_ttl_seconds)
                else:  # fixed TTL - only set expiration if key has no expiration
                    # Handle nx parameter compatibility
                    try:
                        # Try with nx parameter (Redis 7.0+)
                        pipe.expire(k_msgs, self.session_ttl_seconds, nx=True)
                        pipe.expire(k_meta, self.session_ttl_seconds, nx=True)
                    except TypeError:
                        # Fallback: check TTL first, then set if no expiration exists
                        ttl_msgs = self.redis.ttl(k_msgs)
                        ttl_meta = self.redis.ttl(k_meta)
                        if ttl_msgs == -1:  # Key exists but has no expiration
                            pipe.expire(k_msgs, self.session_ttl_seconds)
                        if ttl_meta == -1:
                            pipe.expire(k_meta, self.session_ttl_seconds)
                
                pipe.execute()
                
            except Exception as e:
                logger.error(f"Redis pipeline error in add_message for session {session_id[:16]}...: {e}")
                span.record_exception(e)
                # Try to recover by doing operations individually
                try:
                    redis_client.lpush(k_msgs, msg.to_json())
                    redis_client.ltrim(k_msgs, 0, self.max_messages - 1)
                    redis_client.hsetnx(k_meta, "created_at", now)
                    redis_client.hset(k_meta, mapping={"updated_at": now})
                    if self.ttl_policy == "sliding":
                        redis_client.expire(k_msgs, self.session_ttl_seconds)
                        redis_client.expire(k_meta, self.session_ttl_seconds)
                    else:
                        redis_client.expire(k_msgs, self.session_ttl_seconds, nx=True)
                        redis_client.expire(k_meta, self.session_ttl_seconds, nx=True)
                except Exception as recovery_error:
                    logger.error(f"Redis recovery failed for session {session_id[:16]}...: {recovery_error}")
                    span.record_exception(recovery_error)
                    raise
            
            current_length = self.length(session_id)
            span.set_attribute("conversation.length", current_length)
            
            logger.debug(f"💬 Added {role} message to session {session_id[:16]}... (len: {current_length})")
            return current_length

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[MemoryMessage]:
        """Get messages oldest-first (chronological order)."""
        with tracer.start_as_current_span("memory.get_messages") as span:
            k_msgs = self._k_messages(session_id)
            end = (limit - 1) if (limit is not None and limit > 0) else -1
            try:
                raw = redis_client.lrange(k_msgs, 0, end)
            except Exception as e:
                logger.error(f"Redis lrange error for session {session_id[:16]}...: {e}")
                span.record_exception(e)
                raw = []
            
            # Stored newest-first; present oldest-first for consumers
            messages = []
            for x in reversed(raw):
                try:
                    messages.append(MemoryMessage.from_json(x))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse message JSON for session {session_id[:16]}...: {e}")
                    continue
            
            span.set_attribute("session.id", session_id[:16] + "...")
            span.set_attribute("messages.count", len(messages))
            
            return messages

    def length(self, session_id: str) -> int:
        """Get number of messages in session."""
        try:
            return redis_client.llen(self._k_messages(session_id))
        except Exception as e:
            logger.error(f"Redis llen error for session {session_id[:16]}...: {e}")
            return 0

    def get_ttl(self, session_id: str) -> int:
        """Get remaining TTL in seconds (-2: no key, -1: no TTL)."""
        return redis_client.ttl(self._k_messages(session_id))

    def set_ttl(self, session_id: str, ttl_seconds: int) -> None:
        """Manually set TTL for session."""
        k_msgs = self._k_messages(session_id)
        k_meta = self._k_meta(session_id)
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.expire(k_msgs, ttl_seconds)
            pipe.expire(k_meta, ttl_seconds)
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis pipeline error in set_ttl for session {session_id[:16]}...: {e}")
            # Fallback to individual operations
            try:
                redis_client.expire(k_msgs, ttl_seconds)
                redis_client.expire(k_meta, ttl_seconds)
            except Exception as recovery_error:
                logger.error(f"Redis recovery failed in set_ttl for session {session_id[:16]}...: {recovery_error}")
                raise

    def clear_session(self, session_id: str) -> int:
        """Delete all data for a session."""
        with tracer.start_as_current_span("memory.clear_session") as span:
            span.set_attribute("session.id", session_id[:16] + "...")
            try:
                deleted = redis_client.delete(
                    self._k_messages(session_id),
                    self._k_meta(session_id),
                    self._k_state(session_id)
                )
                logger.info(f"🗑️ Cleared session {session_id[:16]}... ({deleted} keys)")
                return deleted
            except Exception as e:
                logger.error(f"Redis delete error for session {session_id[:16]}...: {e}")
                span.record_exception(e)
                return 0

    def save_state(self, session_id: str, state: ConversationState) -> None:
        """Save the current conversation state."""
        k_state = self._k_state(session_id)
        try:
            # Save state and refresh TTL
            pipe = self.redis.pipeline(transaction=True)
            pipe.set(k_state, state.model_dump_json())
            pipe.expire(k_state, self.session_ttl_seconds)
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis error saving state for session {session_id[:16]}...: {e}")
            
    def get_state(self, session_id: str) -> Optional[ConversationState]:
        """Retrieve the current conversation state."""
        k_state = self._k_state(session_id)
        try:
            raw = redis_client.get(k_state)
            if raw:
                return ConversationState.model_validate_json(raw)
            return None
        except Exception as e:
            logger.error(f"Redis error getting state for session {session_id[:16]}...: {e}")
            return None

    # --------------------------------------------------------------------------
    # Pending Action (Conversation Controller)
    # --------------------------------------------------------------------------
    def save_pending_action(self, session_id: str, payload: dict[str, Any]) -> None:
        """
        Save a pending action (e.g., disambiguation candidates) for the session.
        Stored as JSON with the same TTL as the session keys.
        """
        k = self._k_pending_action(session_id)
        ok = False
        try:
            ok = bool(redis_client.set(k, payload, ex=self.session_ttl_seconds))
        except Exception as e:
            logger.error(f"Redis error saving pending_action for session {session_id[:16]}...: {e}")
            ok = False

        if not ok:
            expires_at = time.time() + float(self.session_ttl_seconds)
            self.__class__._pending_action_local[session_id] = (expires_at, payload)
            logger.warning(
                "Pending action saved to local fallback (Redis unavailable) for session %s...",
                session_id[:16],
            )

    def get_pending_action(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the pending action for the session (if any)."""
        k = self._k_pending_action(session_id)
        try:
            raw = redis_client.get(k)
            if raw:
                if isinstance(raw, str):
                    return json.loads(raw)
                # decode_responses=True should always yield str, but keep this safe
                return json.loads(str(raw))
        except Exception as e:
            logger.error(f"Redis error getting pending_action for session {session_id[:16]}...: {e}")

        # Fallback to in-process store (if present and not expired)
        local = self.__class__._pending_action_local.get(session_id)
        if not local:
            return None
        expires_at, payload = local
        if time.time() >= expires_at:
            self.__class__._pending_action_local.pop(session_id, None)
            return None
        return payload

    def clear_pending_action(self, session_id: str) -> None:
        """Remove any pending action for the session."""
        k = self._k_pending_action(session_id)
        deleted = 0
        try:
            deleted = int(redis_client.delete(k))
        except Exception as e:
            logger.error(f"Redis error clearing pending_action for session {session_id[:16]}...: {e}")
            deleted = 0

        # Always clear local fallback
        self.__class__._pending_action_local.pop(session_id, None)

        # If Redis delete failed (0 could also mean key missing), log only if key seems to still exist.
        if deleted == 0:
            try:
                if redis_client.exists(k):
                    logger.warning(
                        "Failed to clear pending_action in Redis for session %s... (key still exists).",
                        session_id[:16],
                    )
            except Exception:
                # ignore secondary errors
                pass

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        try:
            return redis_client.exists(self._k_messages(session_id))
        except Exception as e:
            logger.error(f"Redis exists error for session {session_id[:16]}...: {e}")
            return False

    def get_formatted_context(self, session_id: str, last_n: int = 5) -> str:
        """Get formatted conversation context for LLM prompts."""
        messages = self.get_messages(session_id, limit=last_n)
        
        if not messages:
            return ""
        
        context_parts = ["Previous conversation:"]
        for msg in messages:
            role_label = "Patient" if msg.role == "user" else "Assistant"
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            context_parts.append(f"{role_label}: {content}")
        
        context_str = "\n".join(context_parts)
        
        logger.debug(f"📜 Formatted context for session {session_id[:16]}... ({len(messages)} messages)")
        return context_str


# Global instance with app configuration
short_term_memory = ShortTermMemoryStore(
    namespace=os.getenv("REDIS_NAMESPACE", "medrag"),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_MINUTES", "60")) * 60,
    max_messages=int(os.getenv("CONVERSATION_HISTORY_LENGTH", "20")),
    ttl_policy=os.getenv("TTL_POLICY", "sliding"),  # type: ignore
)