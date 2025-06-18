# File: graph/reindex_workflow.py
from langgraph.graph import StateGraph
from graph.nodes.summarize import summarize_chunks
from graph.nodes.evaluate import evaluate_chunks, should_resummarize
from graph.nodes.index import index_chunks

# Define StateGraph with custom state class (optional)
workflow = StateGraph()

# Register workflow nodes
workflow.add_node("summarize", summarize_chunks)
workflow.add_node("evaluate", evaluate_chunks)
workflow.add_node("resummarize", summarize_chunks)  # fallback reuse
workflow.add_node("index", index_chunks)

# Define transitions between nodes
workflow.set_entry_point("summarize")
workflow.add_edge("summarize", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    should_resummarize,  # branching function: str -> node label
    {
        "resummarize": "resummarize",
        "index": "index"
    }
)

# Export buildable app (LangGraph API >= 0.0.21)
reindex_app = workflow.compile()
