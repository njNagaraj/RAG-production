-- RBAC: per-user API keys, roles, document group visibility, scoped cache.

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    api_key_hash CHAR(64) UNIQUE NOT NULL,   -- sha256 hex of the API key
    role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
    groups TEXT[] NOT NULL DEFAULT '{}',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS users_api_key_hash_idx ON users(api_key_hash);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS allowed_groups TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS uploaded_by INT;

-- Semantic cache is scoped per permission set: same question from users with
-- different group access must not share cache entries.
ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS access_key TEXT NOT NULL DEFAULT 'public';
ALTER TABLE semantic_cache DROP CONSTRAINT IF EXISTS semantic_cache_query_text_key;
CREATE UNIQUE INDEX IF NOT EXISTS semantic_cache_query_scope_idx ON semantic_cache(query_text, access_key);
CREATE INDEX IF NOT EXISTS semantic_cache_access_key_idx ON semantic_cache(access_key);

ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS uploaded_by INT;
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS allowed_groups TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE query_audits ADD COLUMN IF NOT EXISTS user_id INT;
