def hybrid_retrieve(query, top_k=5):
    """
    Milvus 2.6 + BM25
    """
    vector_hits = milvus_client.search(query.embedding, top_k=top_k)
    keyword_hits = pgsql_catalog.bm25_keyword_search(query.keyword, top_k=top_k)
    return merge_ranked_results(vector_hits, keyword_hits)