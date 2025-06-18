# File: langgraph_reindex.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import asyncio
import logging

from index_manager import IndexingManager
from llm_scrutiny import LLMScrutiny
from evaluator import SummaryEvaluator

# --- Define State ---
class ReindexState(TypedDict):
    doc_id: str
    content: str
    chunks: List[str]
    summaries: List[str]
    embeddings: List[List[float]]
    version: str
    scrutiny_passes: bool
    summary_scores: List[float]
    reindexed: bool

logger = logging.getLogger("LangGraph")

# --- Initialize ---
indexer = IndexingManager(...)  # Milvus, Redis, S3
scrutiny = LLMScrutiny()
evaluator = SummaryEvaluator()

# --- Define Steps ---
async def fetch_document(state: ReindexState) -> ReindexState:
    doc_id = state["doc_id"]
    version = await indexer.get_document_version(doc_id)
    content = await indexer.storage_client.read_document(doc_id)
    return {**state, "version": version, "content": content}

async def run_scrutiny(state: ReindexState) -> ReindexState:
    ok = await scrutiny.run_checks(state["content"][:2000])
    return {**state, "scrutiny_passes": not ok}

async def chunk_and_embed(state: ReindexState) -> ReindexState:
    chunks = await indexer.chunk_document(state["content"])
    summaries, embeddings = [], []
    for chunk in chunks:
        result = await indexer.vector_db.embed_with_summary(chunk)
        summaries.append(result["summary"])
        embeddings.append(result["embedding"])
    return {**state, "chunks": chunks, "summaries": summaries, "embeddings": embeddings}

async def evaluate_chunks(state: ReindexState) -> ReindexState:
    scores = await evaluator.score_summaries(state["summaries"])
    return {**state, "summary_scores": scores}

async def upload_embeddings(state: ReindexState) -> ReindexState:
    for i, (chunk, embedding) in enumerate(zip(state["chunks"], state["embeddings"])):
        metadata = {"doc_id": state["doc_id"], "version": state["version"]}
        await indexer.vector_db.upsert_vector(
            collection_name=state["doc_id"],
            vector_id=f"{state['doc_id']}_chunk_{i}",
            vector=embedding,
            metadata=metadata
        )
    return {**state, "reindexed": True}

async def log_event(event_type: str, state: ReindexState):
    logger.info(f"[{event_type}] doc_id={state['doc_id']} version={state.get('version')}")
    await redis_client.xadd("audit_log", {"event": event_type, "doc_id": state["doc_id"]})

def should_resummarize(state: ReindexState) -> str:
    scores = state.get("summary_scores", [])
    if not scores:
        return "resummarize"
    if any(score < 0.5 for score in scores):
        return "resummarize"
    return "index"

# --- Graph ---
graph = StateGraph(ReindexState)
graph.add_node("fetch", fetch_document)
graph.add_node("scrutiny", run_scrutiny)
graph.add_node("chunk", chunk_and_embed)
graph.add_node("evaluate", evaluate_chunks)
graph.add_node("upload", upload_embeddings)

graph.set_entry_point("fetch")
graph.add_edge("fetch", "scrutiny")
graph.add_conditional_edges(
    "scrutiny",
    lambda s: "chunk" if s["scrutiny_passes"] else END
)
graph.add_edge("chunk", "evaluate")
graph.add_edge("evaluate", "upload")
graph.set_finish_point("upload")

# --- Executor ---
app = graph.compile()

# --- CLI Entrypoint ---
if __name__ == "__main__":
    import sys
    doc_id = sys.argv[1]
    state = asyncio.run(app.invoke({"doc_id": doc_id}))
    print("Reindex complete:", state)


"""
Here's the LangGraph-based flow for triggering document reindexing with a structured state machine. You can run it by passing a doc_id from the CLI:

bash
python langgraph_reindex.py your-doc-id
"""