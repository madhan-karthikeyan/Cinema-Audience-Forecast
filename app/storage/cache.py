from __future__ import annotations

from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class RedisCache:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self._client = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._client.ping()
            logger.info("redis_connected", url=self.redis_url)
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))
            self._client = None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("redis_disconnected")

    async def get(self, key: str) -> str | None:
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            return None

    async def set(
        self, key: str, value: str, ttl: int | None = None
    ) -> bool:
        if not self._client:
            return False
        try:
            await self._client.set(
                key, value, ex=ttl or settings.redis_ttl_seconds
            )
            return True
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        if not self._client:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        if not self._client:
            return False
        try:
            return bool(await self._client.exists(key))
        except Exception:
            return False

    async def ping(self) -> bool:
        if not self._client:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False

    @property
    def connected(self) -> bool:
        return self._client is not None

    def cache_key_lag(self, theater_id: str, target_date: str) -> str:
        return f"lag:{theater_id}:{target_date}"

    def cache_key_features(self, theater_id: str, target_date: str) -> str:
        return f"features:{theater_id}:{target_date}"

    def cache_key_prediction(self, theater_id: str, target_date: str) -> str:
        return f"pred:{theater_id}:{target_date}"
