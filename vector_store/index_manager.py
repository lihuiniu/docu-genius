import asyncio
import logging
from typing import List, Dict, Set
from utils import chunk_text, ChunkStrategy

logger = logging.getLogger("indexing")

class IndexingManager:
    def __init__(self, vector_db, cache_db, storage_client, chunk_strategy=ChunkStrategy.PARAGRAPH, max_chunk_size=1000, tokenizer=None):
        self.vector_db = vector_db
        self.cache_db = cache_db
        self.storage_client = storage_client
        self.index_state = {}
        self.chunk_strategy = chunk_strategy
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer




    async def get_document_version(self, document_id: str) -> str:
        meta = await self.storage_client.get_metadata(document_id)
        return meta.get("version", "unknown")

    async def get_indexing_state(self, document_id: str) -> Dict:
        return self.index_state.get(document_id, {
            "version": None,
            "chunks_indexed": set()
        })

    async def save_indexing_state(self, document_id: str, state: Dict):
        self.index_state[document_id] = state

    async def chunk_document(self, document_text: str) -> List[str]:
        # Use selected chunking strategy
        return chunk_text(
            document_text,
            strategy=self.chunk_strategy,
            max_chunk_size=self.max_chunk_size,
            tokenizer=self.tokenizer,
        )

    async def chunk_document_deprecate(self, document_text: str) -> List[str]:
        chunks = [chunk.strip() for chunk in document_text.split("\n\n") if chunk.strip()]
        return chunks

    async def index_chunk(self, document_id: str, chunk_id: int, chunk_text: str, metadata: Dict):
        embedding = await self.vector_db.embed(chunk_text)
        vector_id = f"{document_id}_chunk_{chunk_id}"

        await self.vector_db.upsert_vector(
            collection_name=document_id,
            vector_id=vector_id,
            vector=embedding,
            metadata=metadata,
        )
        await self.cache_db.set_vector(vector_id, embedding, metadata)
        logger.debug(f"Indexed chunk {chunk_id} for document {document_id}")

    async def clear_index_for_document(self, document_id: str):
        """
        Deletes all vectors and cache entries for the document.
        Call this before full reindex.
        Must be implemented in your vector DB and cache wrappers to delete all document data.
        """
        try:
            await self.vector_db.delete_collection(document_id)  # or delete_vectors_by_prefix
            await self.cache_db.clear_cache_for_document(document_id)
            self.index_state.pop(document_id, None)
            logger.info(f"Cleared index state for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to clear index for document {document_id}: {e}")
            raise

    async def reindex_document(self, document_id: str, full_reindex: bool = False):
        """
        Reindex document chunks.

        If full_reindex=True: clear previous index and reindex all chunks fresh.
        If full_reindex=False: resume indexing missing chunks only.
        If version changed, still reindex missing chunks if full_reindex=False.
        """
        logger.info(f"Starting {'full' if full_reindex else 'incremental'} reindex for document {document_id}")

        version = await self.get_document_version(document_id)
        state = await self.get_indexing_state(document_id)
        current_version = state.get("version")
        # Keeps track of indexed chunks in a set() for resumability.
        indexed_chunks: Set[int] = state.get("chunks_indexed", set())

        # If full reindex, clear all previous index/cache & reset state
        if full_reindex or current_version != version:
            if full_reindex:
                await self.clear_index_for_document(document_id)
                indexed_chunks = set()  # reset chunk state
            # On version mismatch but no full_reindex, still reindex missing chunks

        else:
            logger.info(f"Document {document_id} already indexed with version {version}")
            return  # no reindex needed

        # Load full document text
        content = await self.storage_client.read_document(document_id)
        chunks = await self.chunk_document(content)
        metadata = {"document_id": document_id, "version": version}

        # Index missing chunks only
        for idx, chunk_text in enumerate(chunks):
            if idx in indexed_chunks:
                logger.debug(f"Skipping already indexed chunk {idx}")
                continue

            await self.index_chunk(document_id, idx, chunk_text, metadata)
            indexed_chunks.add(idx)

            # Save state incrementally
            await self.save_indexing_state(document_id, {
                "version": version,
                "chunks_indexed": indexed_chunks
            })

        logger.info(f"Completed reindex for document {document_id}")

    async def get_indexing_progress(self, document_id: str) -> Dict:
        state = await self.get_indexing_state(document_id)
        return {
            "document_id": document_id,
            "version": state.get("version"),
            "chunks_indexed_count": len(state.get("chunks_indexed", [])),
        }

# Example using sentence chunking:
index_manager = IndexingManager(
    vector_db=milvus_store,
    cache_db=redis_cache,
    storage_client=s3_client,
    chunk_strategy=ChunkStrategy.SENTENCE,
    max_chunk_size=1200,
)

# Or with a tokenizer (like tiktoken for OpenAI models)
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base").encode

index_manager = IndexingManager(
    vector_db=milvus_store,
    cache_db=redis_cache,
    storage_client=s3_client,
    chunk_strategy=ChunkStrategy.TOKEN,
    max_chunk_size=500,
    tokenizer=tokenizer,
)
