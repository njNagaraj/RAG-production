-- Migration: RBAC improvements
-- Fix 6: Add updated_at audit column to users with auto-update trigger
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION update_users_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS users_updated_at ON users;
CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_users_updated_at();

-- Fix 7: Composite index so RBAC-scoped cache lookups hit the index directly
--        instead of filtering after a full corpus_epoch scan.
CREATE INDEX IF NOT EXISTS semantic_cache_access_epoch_idx
    ON semantic_cache(access_key, corpus_epoch);
