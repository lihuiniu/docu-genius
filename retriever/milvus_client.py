def insert(embeddings, summaries, metadata):
    metadata = [{**m, "keywords": extract_keywords(m["text"]), "last_modified": ...} for m in metadata]
    # Store embeddings + metadata
