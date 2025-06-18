# graph/nodes/index.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger("index")

async def index_chunks(chunks: List[Dict[str, Any]], vector_db, cache_db) -> List[Dict[str, Any]]:
    """
    Index chunks into vector DB and cache.

    Each chunk dict should have at least:
      - 'summary': str
      - 'embedding': List[float]
      - 'metadata': Dict[str, Any]

    Returns list of indexed chunk metadata or enriched chunk info.
    """
    indexed_chunks = []

    for idx, chunk in enumerate(chunks):
        try:
            # Prepare vector id - can be customized as needed
            vector_id = chunk.get("metadata", {}).get("vector_id", f"chunk_{idx}")

            # Insert embedding into vector DB
            await vector_db.upsert_vector(
                collection_name=chunk["metadata"].get("document_id", "default_collection"),
                vector_id=vector_id,
                vector=chunk["embedding"],
                metadata=chunk["metadata"],
            )

            # Cache chunk for fast retrieval if needed
            await cache_db.set_vector(vector_id, chunk["embedding"], chunk["metadata"])

            indexed_chunks.append(chunk)
            logger.info(f"Indexed chunk {vector_id}")
        except Exception as e:
            logger.error(f"Failed to index chunk {idx}: {e}")

    return indexed_chunks
