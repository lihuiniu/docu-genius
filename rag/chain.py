from langchain.chains import RetrievalQA
from retriever import get_milvus_retriever

def create_rag_chain(llm):
    retriever = get_milvus_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
