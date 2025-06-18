import pytest
import asyncio
from langgraph_logic.index import index_chunks

@pytest.mark.asyncio
async def test_index_chunks_mock(mocker):
    chunks = ["Chunk A", "Chunk B"]
    embeddings = [[0.1]*384, [0.2]*384]
    metadata = [{"doc_id": "doc1"}, {"doc_id": "doc1"}]

    vector_db = mocker.AsyncMock()
    redis_cache = mocker.AsyncMock()

    await index_chunks({
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": metadata,
        "doc_id": "doc1",
        "vector_db": vector_db,
        "redis_cache": redis_cache,
    })

    assert vector_db.insert.call_count == 1
    assert redis_cache.cache_chunks.call_count == 1
