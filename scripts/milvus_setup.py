from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections

# Connect to Milvus 2.6 instance
connections.connect(host="localhost", port="19530")

# Define fields including keyword and word_order
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536, is_primary=False),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
    FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="last_modified", dtype=DataType.INT64),
    FieldSchema(name="keyword", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="word_order", dtype=DataType.BOOL)
]

"""
keyword: supports BM25-style filtering or tag-based grouping.

word_order: useful if certain keyword-based searches require exact phrase match or token sequence logic.

"""
# Define schema
schema = CollectionSchema(fields, description="Enhanced document embedding store with metadata")


# Create the collection
collection = Collection(name="docu_embeddings", schema=schema)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)

# Output confirmation
print("Collection created:", collection.name)


# Run it after Milvus starts: docker exec -it <docu-genius-container> python scripts/milvus_setup.py