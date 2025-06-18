-- File: infra/postgres/init.sql

CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY,
    doc_name TEXT,
    summary TEXT,
    keyword TEXT[],
    last_modified TIMESTAMP,
    acl TEXT DEFAULT 'public',
    storage TEXT,
    bm25 TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_keyword_gin ON document_metadata USING GIN (keyword);
CREATE INDEX idx_last_modified ON document_metadata (last_modified);
