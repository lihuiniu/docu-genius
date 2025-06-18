# File: graph/reindex_graph.py
from langgraph.graph import StateGraph
from typing import List, Dict
from pydantic import BaseModel

from nodes.summarize import summarize_chunks
from nodes.evaluate import evaluate_chunks
from nodes.index import index_chunks
from nodes.utils import should_resummarize


class ReindexState(BaseModel):
    doc_id: str
    content: str
    chunks: List[str]
    summaries: List[str] = []
    summary_scores: List[float] = []
    embeddings: List[List[float]] = []
    metadata: List[Dict] = []


def build_reindex_graph():
    workflow = StateGraph(ReindexState)

    workflow.add_node("summarize", summarize_chunks)
    workflow.add_node("evaluate", evaluate_chunks)
    workflow.add_node("resummarize", summarize_chunks)  # Optional different prompt
    workflow.add_node("index", index_chunks)

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "evaluate")
    workflow.add_conditional_edges("evaluate", should_resummarize, {
        "resummarize": "resummarize",
        "index": "index",
    })

    return workflow.compile()
