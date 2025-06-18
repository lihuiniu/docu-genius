# File: vector_store/milvus_client.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from starlette.concurrency import run_in_threadpool
import logging
import time
from typing import List, Dict


logger = logging.getLogger("MilvusClient")

class MilvusVectorStore:
    """
    Step 2 complete: Created MilvusVectorStore with the following capabilities:

    Connects to Milvus 2.6 server

    Creates schema with fields:

    embedding, summary, source_doc, timestamp, acl, bm25

    Inserts OpenAI-embedded summaries with metadata

    Searches top-k vector matches with ACL filtering


    """
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "document_summaries"):
        self.collection_name = collection_name
        self.dim = 1536  # OpenAI embedding dim
        self.connect(host, port)
        self._init_schema()

    def connect(self, host: str, port: str):
        connections.connect(alias="default", host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port}")

    def _init_schema(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="acl", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="bm25", dtype=DataType.VARCHAR, max_length=1024)
        ]

        schema = CollectionSchema(fields, description="Summarized chunks with metadata")
        self.collection = Collection(name=self.collection_name, schema=schema)
        self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
        logger.info(f"Created Milvus collection: {self.collection_name}")

    def insert(self, embeddings: List[List[float]], summaries: List[str], metadata: List[Dict]):
        insert_data = [
            embeddings,
            summaries,
            [m.get("source_doc", "") for m in metadata],
            [int(time.time()) for _ in summaries],
            [m.get("acl", "public") for m in metadata],
            [m.get("bm25", "") for m in metadata]
        ]
        self.collection.insert(insert_data)
        logger.info(f"Inserted {len(summaries)} chunks into Milvus")

    def search(self, embedding: List[float], top_k: int = 5, acl: str = "public") -> List[Dict]:
        self.collection.load()
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["summary", "source_doc", "timestamp", "acl"]
        )
        matches = []
        for hit in results[0]:
            row = hit.entity
            if row["acl"] == acl:
                matches.append({
                    "summary": row["summary"],
                    "source_doc": row["source_doc"],
                    "timestamp": row["timestamp"],
                    "score": hit.distance
                })
        return matches

    async def delete_collection(self, collection_name: str):
        """
        Deletes the entire collection from Milvus.
        Use if you want to remove all vectors belonging to the document.
        If your Milvus client is synchronous but you want to use it async in FastAPI, wrap sync calls with:
        from starlette.concurrency import run_in_threadpool

        await run_in_threadpool(self.milvus_client.drop_collection, collection_name)
        """
        try:
            # Assuming you have a sync Milvus client wrapped with async calls
            # or you wrap sync calls in run_in_threadpool if using FastAPI
            # For example:
            # await run_in_threadpool(self.milvus_client.drop_collection, collection_name)

            # self.milvus_client.drop_collection(collection_name)  # sync call example
            # Or async call if your milvus client supports
            # await self.milvus_client.drop_collection(collection_name)
            await run_in_threadpool(self.client.drop_collection, collection_name)
            # Log success
            print(f"Milvus collection '{collection_name}' deleted successfully.")
        except Exception as e:
            print(f"Failed to delete Milvus collection '{collection_name}': {e}")
            raise

    async def delete_vectors_by_prefix(self, collection_name: str, prefix: str):
        """
        Deletes vectors with IDs that start with prefix.
        If your Milvus client is synchronous but you want to use it async in FastAPI, wrap sync calls with:
        from starlette.concurrency import run_in_threadpool

        await run_in_threadpool(self.milvus_client.drop_collection, collection_name)
        """
        try:
            # Get all vector IDs (mocked here, replace with actual query if needed)
            all_ids = await run_in_threadpool(self.client.query, collection_name, f"id like \"{prefix}%\"")

            vector_ids = [v["id"] for v in all_ids]  # Adapt if your schema differs
            if vector_ids:
                await run_in_threadpool(self.client.delete, collection_name, f"id in {vector_ids}")
        except Exception as e:
            print(f"[Milvus] Error deleting vectors by prefix: {e}")
            raise
