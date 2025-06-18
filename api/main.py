# File: api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import logging
import os
import tempfile
import asyncio
from starlette.concurrency import run_in_threadpool

from core.summarizer_factory import SummarizerFactory  # ✅ Switched to factory
from vector_store.milvus_client import MilvusVectorStore
from vector_store.redis_cache import RedisVectorCache
from utils.file_io import save_to_s3, read_from_s3, save_to_azure, read_from_azure, load_from_deltalake
from utils.upload_tracker import UploadTracker
from fastapi import BackgroundTasks, Query

from llm_scrutiny import LLMScrutiny
from index_namager import IndexManager
from graph.reindex_graph import build_reindex_graph

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Global config via ENV or fallback
DEFAULT_ENGINE = os.getenv("SUMMARIZER_ENGINE", "openai")
DEFAULT_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize clients
summarizer = SummarizerFactory(DEFAULT_ENGINE, DEFAULT_MODEL, DEFAULT_API_KEY)
milvus_store = MilvusVectorStore()
redis_cache = RedisVectorCache()
upload_tracker = UploadTracker()

graph_app = build_reindex_graph()

class QueryRequest(BaseModel):
    embedding: Optional[List[float]] = None
    keyword: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    min_time: Optional[int] = None
    max_time: Optional[int] = None
    top_k: Optional[int] = 5
    top_p: Optional[float] = None
    word_order: Optional[bool] = False

class SummaryRequest(BaseModel):
    doc_id: str
    storage: str = "local"
    path: str
    acl: Optional[str] = "public"
    bm25: Optional[str] = ""
    use_cache: Optional[bool] = True

@app.post("/upload/{storage}")
async def upload_file(storage: str, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    doc_id = str(uuid.uuid4())
    try:
        upload_tracker.start_upload(doc_id)

        async def save_and_process():
            try:
                if storage == "s3":
                    await run_in_threadpool(save_to_s3, file.file, f"documents/{doc_id}.txt")
                elif storage == "azure":
                    await run_in_threadpool(save_to_azure, file.file, f"documents/{doc_id}.txt")
                else:
                    os.makedirs("./uploads", exist_ok=True)
                    with open(f"./uploads/{doc_id}.txt", "wb") as f:
                        f.write(file.file.read())

                upload_tracker.complete_upload(doc_id)
                await run_in_threadpool(process_uploaded_doc, doc_id, storage)
            except Exception as e:
                logger.error(f"Upload failed for {doc_id}: {e}")
                upload_tracker.fail_upload(doc_id)

        background_tasks.add_task(save_and_process)
        return JSONResponse({"doc_id": doc_id, "status": "uploading"})
    except Exception as e:
        logger.error(f"Upload trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Upload initialization failed")

@app.get("/upload/status/{doc_id}")
def check_upload_status(doc_id: str):
    status = upload_tracker.get_status(doc_id)
    return {"doc_id": doc_id, "status": status}

async def process_uploaded_doc(doc_id: str, storage: str):
    path = f"documents/{doc_id}.txt" if storage in ["s3", "azure"] else f"./uploads/{doc_id}.txt"
    try:
        if storage == "s3":
            content = read_from_s3(path)
        elif storage == "azure":
            content = read_from_azure(path)
        elif storage == "delta":
            content = load_from_deltalake(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        if not await document_passes_scrutiny(content):
            logger.warning(f"Document {doc_id} failed scrutiny")
            return

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        chunks = summarizer.summarize_file(tmp_path)
        summaries = [c["summary"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadata = [{"source_doc": doc_id, "acl": "public"} for _ in summaries]

        milvus_store.insert(embeddings, summaries, metadata)
        redis_cache.cache_chunks(doc_id, chunks)

        logger.info(f"Processing complete for document {doc_id}")
    except Exception as e:
        logger.error(f"Post-upload processing failed for {doc_id}: {e}")

async def document_passes_scrutiny(content: str) -> bool:
    if len(content) < 100:
        return False

    llm_scrutiny = LLMScrutiny()
    try:
        flagged = await llm_scrutiny.run_checks(content[:2000])
        if flagged:
            logger.warning("Document flagged as inappropriate")
        return not flagged
    except Exception as e:
        logger.warning(f"Scrutiny check error: {e}, defaulting to pass")
        return True

@app.post("/summarize")
def summarize_document(req: SummaryRequest):
    try:
        if req.use_cache:
            cached = redis_cache.get_chunks(req.doc_id)
            if cached:
                logger.info(f"Returning cached summary for {req.doc_id}")
                return {"summary_count": len(cached), "source": "cache", "chunks": cached}

        if req.storage == "s3":
            content = read_from_s3(req.path)
        elif req.storage == "azure":
            content = read_from_azure(req.path)
        elif req.storage == "delta":
            content = load_from_deltalake(req.path)
        else:
            with open(req.path, "r", encoding="utf-8") as f:
                content = f.read()

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        chunks = summarizer.summarize_file(tmp_path)
        summaries = [c["summary"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadata = [{"source_doc": req.doc_id, "acl": req.acl, "bm25": req.bm25} for _ in summaries]

        milvus_store.insert(embeddings, summaries, metadata)
        redis_cache.cache_chunks(req.doc_id, chunks)

        return {"summary_count": len(chunks), "source": "generated", "chunks": chunks}
    except Exception as e:
        logger.error(f"Summarize failed: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed")

@app.post("/query")
def query_similar_chunks(query: QueryRequest):
    try:
        results = []
        if query.embedding:
            results = milvus_store.search(query.embedding, top_k=query.top_k, acl=query.metadata.get("acl") if query.metadata else None)
        elif query.keyword:
            cached_docs = redis_cache.list_cached_docs()
            for doc_id in cached_docs:
                results.extend(redis_cache.get_chunks(doc_id, acl=query.metadata.get("acl") if query.metadata else None,
                                                     keyword=query.keyword, min_time=query.min_time, max_time=query.max_time))

        if query.top_p:
            limit = int(len(results) * query.top_p)
            results = results[:max(limit, 1)]

        return {"results": results[:query.top_k] if query.top_k else results}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")

@app.get("/cache/{doc_id}")
def get_cached_chunks(doc_id: str, acl: Optional[str] = None, keyword: Optional[str] = None,
                      min_time: Optional[int] = None, max_time: Optional[int] = None):
    try:
        results = redis_cache.get_chunks(doc_id, acl=acl, keyword=keyword, min_time=min_time, max_time=max_time)
        return {"chunks": results}
    except Exception as e:
        logger.error(f"Cache retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Cache retrieval failed")

@app.post("/reindex/{storage}/{doc_id}")
async def reindex_document_endpoint(
    doc_id: str,
    background_tasks: BackgroundTasks,
    full_reindex: bool = Query(False, description="If true, force full reindex")
):
    index_manager = IndexManager(milvus_store, redis_cache, storage)
    if full_reindex:
        background_tasks.add_task(index_manager.reindex_document, doc_id, full_reindex)
    return {"doc_id": doc_id, "status": "reindexing_started", "full_reindex": full_reindex}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/readyz")
def readiness_probe():
    return {"status": "ok"}