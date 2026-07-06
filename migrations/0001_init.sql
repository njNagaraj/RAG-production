-- Initial schema. {{DIMENSION}} is substituted by the migration runner with
-- the active embedding model's dimension at first startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source VARCHAR(255) UNIQUE,
    title VARCHAR(512),
    total_pages INT,
    file_path TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT,
    page_num INT,
    content TEXT,
    context TEXT,
    embedding VECTOR({{DIMENSION}}),
    content_tsvector TSVECTOR,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx ON document_chunks(document_id);

CREATE OR REPLACE FUNCTION document_chunks_tsvector_trigger() RETURNS trigger AS $$
begin
  new.content_tsvector := to_tsvector('english', coalesce(new.content, '') || ' ' || coalesce(new.context, ''));
  return new;
end
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvectorupdate ON document_chunks;
CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON document_chunks FOR EACH ROW EXECUTE FUNCTION document_chunks_tsvector_trigger();

CREATE INDEX IF NOT EXISTS document_chunks_fts_idx ON document_chunks USING GIN(content_tsvector);

CREATE INDEX IF NOT EXISTS document_chunks_hnsw_idx ON document_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS semantic_cache (
    id SERIAL PRIMARY KEY,
    query_text TEXT UNIQUE,
    query_embedding VECTOR({{DIMENSION}}),
    answer TEXT,
    citations JSONB,
    corpus_epoch INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS semantic_cache_hnsw_idx ON semantic_cache
USING hnsw (query_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS semantic_cache_epoch_idx ON semantic_cache(corpus_epoch);

CREATE TABLE IF NOT EXISTS rag_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT INTO rag_meta (key, value) VALUES ('corpus_epoch', '0')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    enrich BOOLEAN DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'queued',  -- queued | processing | done | failed
    error TEXT,
    chunks_count INT,
    document_id INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx ON ingest_jobs(status);

CREATE TABLE IF NOT EXISTS query_audits (
    id SERIAL PRIMARY KEY,
    question TEXT,
    answer TEXT,
    faithfulness_score REAL,
    faithful BOOLEAN,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
