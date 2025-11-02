"""
Helios Redis Queue Management v3

Redis-based caching, pub/sub, and queue management utilities
for high-performance data operations and task coordination.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError

from config.settings import get_settings, get_redis_config
from utils.logger import get_logger

logger = get_logger("redis")


class RedisManager:
    """
    Redis connection and operation management.
    
    Provides connection pooling, caching, pub/sub, and queue operations.
    """
    
    def __init__(self):
        """Initialize Redis manager."""
        self.settings = get_settings()
        self.redis_config = get_redis_config()
        self._pool = None
        self._async_pool = None
        self._client = None
        self._async_client = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize Redis connections.
        """
        if self._initialized:
            logger.warning("Redis already initialized")
            return
        
        try:
            # Parse Redis URL
            redis_url = self.redis_config["url"]
            
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                redis_url,
                max_connections=self.redis_config["max_connections"],
                socket_keepalive=self.redis_config["socket_keepalive"],
                socket_keepalive_options=self.redis_config["socket_keepalive_options"],
                decode_responses=True,
            )
            
            # Create async connection pool
            self._async_pool = aioredis.ConnectionPool.from_url(
                redis_url,
                max_connections=self.redis_config["max_connections"],
                decode_responses=True,
            )
            
            # Create clients
            self._client = redis.Redis(connection_pool=self._pool)
            self._async_client = aioredis.Redis(connection_pool=self._async_pool)
            
            # Test connections
            self.test_connection()
            
            self._initialized = True
            logger.info("Redis initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test Redis connectivity.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self._client.ping()
            logger.debug("Redis connection test successful")
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            return False
    
    async def test_async_connection(self) -> bool:
        """
        Test async Redis connectivity.
        
        Returns:
            bool: True if connection successful
        """
        try:
            await self._async_client.ping()
            logger.debug("Async Redis connection test successful")
            return True
        except Exception as e:
            logger.error(f"Async Redis connection test failed: {e}")
            return False
    
    def get_client(self) -> redis.Redis:
        """
        Get synchronous Redis client.
        
        Returns:
            redis.Redis: Redis client
        """
        if not self._initialized:
            self.initialize()
        return self._client
    
    def get_async_client(self) -> aioredis.Redis:
        """
        Get asynchronous Redis client.
        
        Returns:
            aioredis.Redis: Async Redis client
        """
        if not self._initialized:
            self.initialize()
        return self._async_client
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            client = self.get_client()
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            return client.set(key, serialized_value, ex=ttl)
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}: {e}")
            return False
    
    def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize JSON values
            
        Returns:
            Optional[Any]: Value or None if not found
        """
        try:
            client = self.get_client()
            value = client.get(key)
            
            if value is None:
                return None
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key to delete
            
        Returns:
            bool: True if key was deleted
        """
        try:
            client = self.get_client()
            return bool(client.delete(key))
        except Exception as e:
            logger.error(f"Redis DELETE failed for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.
        
        Args:
            key: Redis key to check
            
        Returns:
            bool: True if key exists
        """
        try:
            client = self.get_client()
            return bool(client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key {key}: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter in Redis.
        
        Args:
            key: Redis key
            amount: Amount to increment by
            
        Returns:
            Optional[int]: New value or None if failed
        """
        try:
            client = self.get_client()
            return client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCRBY failed for key {key}: {e}")
            return None
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Redis key
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            client = self.get_client()
            return bool(client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Redis EXPIRE failed for key {key}: {e}")
            return False
    
    # List operations
    def lpush(self, key: str, *values: Any) -> Optional[int]:
        """
        Push values to the left of a list.
        
        Args:
            key: Redis list key
            values: Values to push
            
        Returns:
            Optional[int]: New list length or None if failed
        """
        try:
            client = self.get_client()
            serialized_values = [json.dumps(v) if not isinstance(v, str) else v for v in values]
            return client.lpush(key, *serialized_values)
        except Exception as e:
            logger.error(f"Redis LPUSH failed for key {key}: {e}")
            return None
    
    def rpop(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Pop value from the right of a list.
        
        Args:
            key: Redis list key
            deserialize: Whether to deserialize JSON values
            
        Returns:
            Optional[Any]: Popped value or None if list empty
        """
        try:
            client = self.get_client()
            value = client.rpop(key)
            
            if value is None:
                return None
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Redis RPOP failed for key {key}: {e}")
            return None
    
    def llen(self, key: str) -> int:
        """
        Get length of a list.
        
        Args:
            key: Redis list key
            
        Returns:
            int: List length (0 if key doesn't exist)
        """
        try:
            client = self.get_client()
            return client.llen(key)
        except Exception as e:
            logger.error(f"Redis LLEN failed for key {key}: {e}")
            return 0
    
    # Async operations
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Async set a key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            client = self.get_async_client()
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            return await client.set(key, serialized_value, ex=ttl)
        except Exception as e:
            logger.error(f"Async Redis SET failed for key {key}: {e}")
            return False
    
    async def aget(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Async get a value from Redis.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize JSON values
            
        Returns:
            Optional[Any]: Value or None if not found
        """
        try:
            client = self.get_async_client()
            value = await client.get(key)
            
            if value is None:
                return None
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Async Redis GET failed for key {key}: {e}")
            return None
    
    # Pub/Sub operations
    def publish(self, channel: str, message: Any) -> Optional[int]:
        """
        Publish message to a channel.
        
        Args:
            channel: Redis channel
            message: Message to publish
            
        Returns:
            Optional[int]: Number of subscribers that received the message
        """
        try:
            client = self.get_client()
            serialized_message = json.dumps(message) if not isinstance(message, str) else message
            return client.publish(channel, serialized_message)
        except Exception as e:
            logger.error(f"Redis PUBLISH failed for channel {channel}: {e}")
            return None
    
    async def subscribe(self, channels: List[str], callback: Callable[[str, Any], None]) -> None:
        """
        Subscribe to channels and handle messages.
        
        Args:
            channels: List of channels to subscribe to
            callback: Function to call for each message
        """
        try:
            client = self.get_async_client()
            pubsub = client.pubsub()
            
            await pubsub.subscribe(*channels)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    
                    # Try to deserialize JSON
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    await callback(channel, data)
                    
        except Exception as e:
            logger.error(f"Redis SUBSCRIBE failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Dict[str, Any]: Redis server info
        """
        try:
            client = self.get_client()
            info = client.info()
            return {
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace": {k: v for k, v in info.items() if k.startswith("db")},
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    def cleanup(self) -> None:
        """
        Clean up Redis connections.
        """
        try:
            if self._client:
                self._client.close()
            if self._pool:
                self._pool.disconnect()
            logger.info("Redis connections cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up Redis connections: {e}")


# Global Redis manager instance
redis_manager = RedisManager()


# Convenience functions
def get_redis_client() -> redis.Redis:
    """Get synchronous Redis client."""
    return redis_manager.get_client()


def get_async_redis_client() -> aioredis.Redis:
    """Get asynchronous Redis client."""
    return redis_manager.get_async_client()


def initialize_redis():
    """Initialize Redis connection."""
    redis_manager.initialize()


def test_redis_connection() -> bool:
    """Test Redis connectivity."""
    return redis_manager.test_connection()


class RedisCache:
    """
    High-level Redis caching utility.
    """
    
    def __init__(self, prefix: str = "helios", default_ttl: int = 3600):
        """Initialize cache with prefix and default TTL."""
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.manager = redis_manager
    
    def _key(self, key: str) -> str:
        """Generate prefixed cache key."""
        return f"{self.prefix}:{key}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value."""
        return self.manager.set(self._key(key), value, ttl or self.default_ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return self.manager.get(self._key(key))
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        return self.manager.delete(self._key(key))
    
    def exists(self, key: str) -> bool:
        """Check if cached value exists."""
        return self.manager.exists(self._key(key))


# Pre-configured cache instances
market_data_cache = RedisCache(prefix="market_data", default_ttl=300)  # 5 minutes
strategy_cache = RedisCache(prefix="strategy", default_ttl=3600)  # 1 hour
performance_cache = RedisCache(prefix="performance", default_ttl=86400)  # 24 hours
