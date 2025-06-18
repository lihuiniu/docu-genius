# File: nodes/summarize.py
from typing import List, Dict
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document


def summarize_chunks(state):
    doc_id = state.doc_id
    content = state.content

    # Split into chunks
    chunks = content.split("\n\n")
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Summarize each chunk
    llm = ChatOpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff")
    summaries = [chain.run([doc]) for doc in documents]

    return {
        **state.dict(),
        "chunks": chunks,
        "summaries": summaries
    }
