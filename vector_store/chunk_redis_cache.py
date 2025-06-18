# File: vector_store/chunk_redis_cache.py
import redis
import logging
import json
from typing import List, Dict

logger = logging.getLogger("ChunkRedisVectorCache")

class ChunkRedisVectorCache:
    """
    Step 3 complete: Created RedisVectorCache with support for:

    Caching document chunks by doc_id

    Retrieving and clearing cached summaries

    Listing cached document IDs

    Uses Redis HashSets and stores each chunk as JSON
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        logger.info("Connected to Redis 8.0 VectorSet")

    def _key(self, doc_id: str) -> str:
        return f"doc:{doc_id}"

    def cache_chunks(self, doc_id: str, chunks: List[Dict]):
        key = self._key(doc_id)
        try:
            for i, chunk in enumerate(chunks):
                value = json.dumps(chunk)
                self.redis.hset(key, f"chunk:{i}", value)
            logger.info(f"Cached {len(chunks)} chunks under {key}")
        except Exception as e:
            logger.error(f"Failed to cache chunks: {e}")

    def get_chunks(self, doc_id: str) -> List[Dict]:
        key = self._key(doc_id)
        try:
            raw = self.redis.hgetall(key)
            return [json.loads(v) for v in raw.values()]
        except Exception as e:
            logger.error(f"Failed to retrieve cached chunks: {e}")
            return []

    def clear_cache(self, doc_id: str):
        key = self._key(doc_id)
        self.redis.delete(key)
        logger.info(f"Cleared cache for {key}")

    def list_cached_docs(self) -> List[str]:
        try:
            keys = self.redis.keys("doc:*")
            return [k.split(":")[1] for k in keys]
        except Exception as e:
            logger.error(f"Failed to list cached docs: {e}")
            return []
