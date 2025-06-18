from langgraph import StateGraph
from retriever.hybrid_search import hybrid_retrieve
from core.summarizer_factory import SummarizerFactory
from memory.redis_cache import RedisVectorCache
from telemetry.tracing import trace

@trace
def build_rag_controller():
    sg = StateGraph()
    sg.add_node("Retrieve", hybrid_retrieve)
    sg.add_node("Summarize", SummarizerFactory(...).summarize_query)
    sg.add_edge("start", "Retrieve")
    sg.add_edge("Retrieve", "Summarize")
    return sg.compile()
