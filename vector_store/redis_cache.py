# File: vector_store/redis_cache.py
import redis
import logging
import json
import time
from typing import List, Dict, Optional
import asyncio

logger = logging.getLogger("RedisVectorCache")

class RedisVectorCache:
    """
    Redis ver 8.0 + support VectorSet
    RedisVectorCache Supports:

        timestamp filtering (min/max)

        keyword search within summaries

        ACL-based filtering

    All chunks are stored as JSON with metadata
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        logger.info("Connected to Redis 8.0 VectorSet")

    def _key(self, doc_id: str) -> str:
        return f"doc:{doc_id}"

    def cache_chunks(self, doc_id: str, chunks: List[Dict]):
        key = self._key(doc_id)
        try:
            pipe = self.redis.pipeline()
            for i, chunk in enumerate(chunks):
                chunk["timestamp"] = chunk.get("timestamp", int(time.time()))
                pipe.hset(key, f"chunk:{i}", json.dumps(chunk))
            pipe.execute()
            logger.info(f"Cached {len(chunks)} chunks under {key}")
        except Exception as e:
            logger.error(f"Failed to cache chunks: {e}")

    def get_chunks(self, doc_id: str, acl: Optional[str] = None, keyword: Optional[str] = None,
                   min_time: Optional[int] = None, max_time: Optional[int] = None) -> List[Dict]:
        key = self._key(doc_id)
        try:
            raw = self.redis.hgetall(key)
            results = []
            for v in raw.values():
                chunk = json.loads(v)
                if acl and chunk.get("acl") != acl:
                    continue
                if keyword and keyword.lower() not in chunk.get("summary", "").lower():
                    continue
                if min_time and chunk.get("timestamp", 0) < min_time:
                    continue
                if max_time and chunk.get("timestamp", 0) > max_time:
                    continue
                results.append(chunk)
            return results
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

# Optional wrapper for LangChain memory integration
class RedisConversationMemory:
    """
    LangChain-compatible wrapper Supports:

        Append messages (with role: user/assistant)

        Load last N messages (for chain-of-thought memory)

        Clear session memory

    Easy to integrate with LangChain memory or custom chat interfaces
    """
    def __init__(self, redis_client: redis.Redis, session_id: str):
        self.redis = redis_client
        self.session_key = f"memory:{session_id}"

    def append_message(self, role: str, content: str):
        entry = json.dumps({"role": role, "content": content, "timestamp": int(time.time())})
        self.redis.rpush(self.session_key, entry)

    def load_conversation(self, limit: int = 20) -> List[Dict]:
        raw = self.redis.lrange(self.session_key, -limit, -1)
        return [json.loads(x) for x in raw]

    def clear(self):
        self.redis.delete(self.session_key)
        logger.info(f"Cleared memory for {self.session_key}")

    async def clear_cache_for_document(self, document_id: str):
        """
        Deletes all cache keys related to the given document.
        Assuming your cache keys for chunks follow the pattern like: "{document_id}_chunk_{chunk_id}"
        If your Milvus client is synchronous but you want to use it async in FastAPI, wrap sync calls with:
        from starlette.concurrency import run_in_threadpool

        await run_in_threadpool(self.redis_cache.clear_cache_for_document, cache_key, document_id)
        """
        try:
            pattern = f"{document_id}_chunk_*"
            cursor = b'0'
            keys_to_delete = []
            while cursor:
                cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=100)
                keys_to_delete.extend(keys)
                if cursor == b'0':
                    break

            if keys_to_delete:
                await self.redis.delete(*keys_to_delete)
                print(f"[Redis] Deleted {len(keys_to_delete)} keys for document {document_id}")
            else:
                print(f"[Redis] No cache found for document {document_id}")
        except Exception as e:
            print(f"[Redis] Error clearing cache for document {document_id}: {e}")
            raise
