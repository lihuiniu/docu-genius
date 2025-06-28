from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings

"""
Milvus Hybrid Search Setup
"""
def get_milvus_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name="docs",
        connection_args={"host": "localhost", "port": "19530"}
    )
    return vectorstore.as_retriever(search_type="mmr", k=5)
