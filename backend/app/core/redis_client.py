import json
import logging
import os
from typing import Any, Optional

import redis
from fakeredis import FakeRedis

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis client wrapper with fallback to fakeredis for development."""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._fake_client: Optional[FakeRedis] = None
        self._is_initialized = False
    
    def _initialize_client(self):
        """Initialize Redis or fallback to FakeRedis."""
        if self._is_initialized:
            return
        
        redis_url = os.getenv("REDIS_URL")
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        try:
            if redis_url:
                # Parse URL to check if database is already specified
                # redis://host:port/db or redis://:password@host:port/db
                import urllib.parse
                parsed = urllib.parse.urlparse(redis_url)
                
                # Extract db from URL path if present (format: /db_number)
                url_db = None
                if parsed.path and len(parsed.path) > 1:
                    try:
                        url_db = int(parsed.path[1:])  # Remove leading '/'
                    except ValueError:
                        pass
                
                # Use db from URL if present, otherwise use REDIS_DB env var
                # redis.from_url() will use the db from URL automatically, so we only
                # need to pass db parameter if we want to override it
                if url_db is not None:
                    # URL already specifies db, use it as-is
                    self._client = redis.from_url(redis_url, decode_responses=True)
                    logger.info(f"✅ Redis connected via URL (db={url_db}): {parsed.hostname}:{parsed.port or 6379}")
                else:
                    # URL doesn't specify db, use REDIS_DB env var
                    # Note: redis.from_url doesn't accept db parameter directly when URL doesn't have it
                    # So we need to create connection manually or append db to URL
                    if redis_db != 0:
                        # Append db to URL if not present
                        separator = "/" if not parsed.path else ""
                        redis_url_with_db = f"{redis_url}{separator}{redis_db}"
                        self._client = redis.from_url(redis_url_with_db, decode_responses=True)
                        logger.info(f"✅ Redis connected via URL (db={redis_db}): {parsed.hostname}:{parsed.port or 6379}")
                    else:
                        # Default db 0, use URL as-is
                        self._client = redis.from_url(redis_url, decode_responses=True)
                        logger.info(f"✅ Redis connected via URL (db=0): {parsed.hostname}:{parsed.port or 6379}")
            else:
                self._client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True
                )
                logger.info(f"✅ Redis connected: {redis_host}:{redis_port} (db={redis_db})")
            
            self._client.ping()
            self._is_initialized = True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis failed: {e}. Using FakeRedis")
            self._client = None
            self._fake_client = FakeRedis(decode_responses=True)
            self._is_initialized = True

    @property
    def client(self):
        """Get the active Redis client (initialize on first use)."""
        if not self._is_initialized:
            self._initialize_client()
        return self._client if self._client is not None else self._fake_client
    
    def get(self, key: str) -> Optional[str]:
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            return bool(self.client.set(key, value, ex=ex))
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, *keys: str) -> int:
        try:
            return int(self.client.delete(*keys))
        except Exception as e:
            logger.error(f"Redis delete error for keys {keys}: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    def expire(self, key: str, seconds: int, nx: bool = False) -> bool:
        """
        Set expiration on a key.
        Args:
            key: Redis key
            seconds: TTL in seconds
            nx: If True, set expiration only if key has no expiration (Redis 7.0+)
        """
        try:
            if nx:
                # Check if nx parameter is supported (Redis 7.0+)
                # Fallback to regular expire if not supported
                try:
                    return bool(self.client.expire(key, seconds, nx=True))
                except TypeError:
                    # nx parameter not supported, check if key exists and has no TTL
                    ttl_val = self.client.ttl(key)
                    if ttl_val == -1:  # Key exists but has no expiration
                        return bool(self.client.expire(key, seconds))
                    return False  # Key already has expiration
            else:
                return bool(self.client.expire(key, seconds))
        except Exception as e:
            logger.error(f"Redis expire error for key {key}: {e}")
            return False
    
    def lpush(self, key: str, *values) -> int:
        try:
            return int(self.client.lpush(key, *values))
        except Exception as e:
            logger.error(f"Redis lpush error for key {key}: {e}")
            return 0
    
    def lrange(self, key: str, start: int, end: int) -> list:
        try:
            return list(self.client.lrange(key, start, end))
        except Exception as e:
            logger.error(f"Redis lrange error for key {key}: {e}")
            return []
    
    def ltrim(self, key: str, start: int, end: int) -> bool:
        try:
            return bool(self.client.ltrim(key, start, end))
        except Exception as e:
            logger.error(f"Redis ltrim error for key {key}: {e}")
            return False
    
    def pipeline(self, transaction: bool = True):
        return self.client.pipeline(transaction=transaction)
    
    def ttl(self, key: str) -> int:
        """Get TTL in seconds (-2: no key, -1: no TTL)."""
        try:
            ttl = self.client.ttl(key)
            return int(ttl) if ttl is not None else -2
        except Exception as e:
            logger.error(f"Redis ttl error for key {key}: {e}")
            return -2
    
    # Hash operations
    def hset(self, key: str, mapping: Optional[dict] = None, **kwargs) -> int:
        """
        Set hash field(s). Supports both mapping dict and keyword arguments.
        Returns number of fields that were added.
        """
        try:
            if mapping is not None:
                return int(self.client.hset(key, mapping=mapping))
            elif kwargs:
                return int(self.client.hset(key, mapping=kwargs))
            else:
                logger.warning(f"Redis hset called with no data for key {key}")
                return 0
        except Exception as e:
            logger.error(f"Redis hset error for key {key}: {e}")
            return 0
    
    def hsetnx(self, key: str, field: str, value: Any) -> bool:
        """
        Set hash field only if it doesn't exist.
        Returns True if field was set, False if it already existed.
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            return bool(self.client.hsetnx(key, field, value))
        except Exception as e:
            logger.error(f"Redis hsetnx error for key {key}, field {field}: {e}")
            return False
    
    def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value."""
        try:
            return self.client.hget(key, field)
        except Exception as e:
            logger.error(f"Redis hget error for key {key}, field {field}: {e}")
            return None
    
    def hgetall(self, key: str) -> dict:
        """Get all hash fields and values."""
        try:
            result = self.client.hgetall(key)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Redis hgetall error for key {key}: {e}")
            return {}
    
    def llen(self, key: str) -> int:
        """Get length of a list."""
        try:
            return int(self.client.llen(key))
        except Exception as e:
            logger.error(f"Redis llen error for key {key}: {e}")
            return 0

# Global instance
redis_client = RedisClient()