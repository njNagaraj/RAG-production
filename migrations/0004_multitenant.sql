-- Multi-tenant RBAC: organizations, Org -> Department -> Team hierarchy,
-- per-node scoped role assignments, document inheritance, tenant isolation.
--
-- Safe on both a fresh DB (backfills touch 0 rows) and a populated dev DB
-- (existing users/documents are migrated into a synthetic "default" org).

-- 1. Organizations -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS organizations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    plan TEXT NOT NULL DEFAULT 'standard',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Hierarchy nodes ---------------------------------------------------------
-- ancestor_ids = materialized path from root to self INCLUSIVE, so:
--   ancestors-with-self(N) = N.ancestor_ids
--   subtree(N)             = rows where N.id = ANY(ancestor_ids)
CREATE TABLE IF NOT EXISTS org_units (
    id SERIAL PRIMARY KEY,
    org_id INT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    parent_id INT REFERENCES org_units(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('org', 'department', 'team')),
    name VARCHAR(200) NOT NULL,
    ancestor_ids INT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS org_units_org_idx ON org_units(org_id);
CREATE INDEX IF NOT EXISTS org_units_parent_idx ON org_units(parent_id);
CREATE INDEX IF NOT EXISTS org_units_ancestors_idx ON org_units USING GIN(ancestor_ids);

-- 3. Scoped role assignments -------------------------------------------------
CREATE TABLE IF NOT EXISTS role_assignments (
    id SERIAL PRIMARY KEY,
    org_id INT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    unit_id INT NOT NULL REFERENCES org_units(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, unit_id, role)
);
CREATE INDEX IF NOT EXISTS role_assignments_user_idx ON role_assignments(user_id);
CREATE INDEX IF NOT EXISTS role_assignments_unit_idx ON role_assignments(unit_id);

-- 4. Users: org membership, home unit, platform-admin flag -------------------
ALTER TABLE users ADD COLUMN IF NOT EXISTS org_id INT REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS home_unit_id INT REFERENCES org_units(id) ON DELETE SET NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_platform_admin BOOLEAN NOT NULL DEFAULT FALSE;

-- users.role now includes 'owner' (was admin/editor/viewer under flat RBAC)
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('owner', 'admin', 'editor', 'viewer'));

-- 5. Documents / chunks: tenancy + node attachment ---------------------------
ALTER TABLE documents ADD COLUMN IF NOT EXISTS org_id INT REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS unit_id INT REFERENCES org_units(id) ON DELETE SET NULL;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS org_id INT;

ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS org_id INT;
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS org_id INT;
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS unit_id INT;
ALTER TABLE query_audits ADD COLUMN IF NOT EXISTS org_id INT;

-- 6. Backfill existing rows into a synthetic default org ---------------------
DO $$
DECLARE
    default_org_id INT;
    default_unit_id INT;
    u RECORD;
BEGIN
    IF EXISTS (SELECT 1 FROM users WHERE org_id IS NULL)
       OR EXISTS (SELECT 1 FROM documents WHERE org_id IS NULL) THEN

        INSERT INTO organizations (name, slug)
        VALUES ('Default Organization', 'default')
        ON CONFLICT (slug) DO NOTHING;
        SELECT id INTO default_org_id FROM organizations WHERE slug = 'default';

        SELECT id INTO default_unit_id FROM org_units WHERE org_id = default_org_id AND type = 'org' LIMIT 1;
        IF default_unit_id IS NULL THEN
            INSERT INTO org_units (org_id, parent_id, type, name, ancestor_ids)
            VALUES (default_org_id, NULL, 'org', 'Default Organization', '{}')
            RETURNING id INTO default_unit_id;
            UPDATE org_units SET ancestor_ids = ARRAY[default_unit_id] WHERE id = default_unit_id;
        END IF;

        UPDATE users SET org_id = default_org_id, home_unit_id = default_unit_id WHERE org_id IS NULL;
        UPDATE documents SET org_id = default_org_id, unit_id = default_unit_id WHERE org_id IS NULL;
        UPDATE document_chunks dc SET org_id = default_org_id
            FROM documents d WHERE dc.document_id = d.id AND dc.org_id IS NULL;
        UPDATE ingest_jobs SET org_id = default_org_id, unit_id = default_unit_id WHERE org_id IS NULL;

        -- Convert each existing flat user.role into a role assignment at the root unit.
        FOR u IN SELECT id, role FROM users WHERE org_id = default_org_id LOOP
            INSERT INTO role_assignments (org_id, user_id, unit_id, role)
            VALUES (default_org_id, u.id, default_unit_id,
                    CASE WHEN u.role = 'admin' THEN 'admin'
                         WHEN u.role = 'editor' THEN 'editor'
                         ELSE 'viewer' END)
            ON CONFLICT (user_id, unit_id, role) DO NOTHING;
        END LOOP;
    END IF;
END $$;

-- 7. Drop the obsolete flat-model columns ------------------------------------
ALTER TABLE documents DROP COLUMN IF EXISTS allowed_groups;
ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS allowed_groups;
ALTER TABLE users DROP COLUMN IF EXISTS groups;

-- Per-org username uniqueness (replaces the old global unique constraint)
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_username_key;
CREATE UNIQUE INDEX IF NOT EXISTS users_org_username_idx ON users(org_id, username);

-- Document source uniqueness is now per-org (two orgs may upload the same filename)
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_key;
CREATE UNIQUE INDEX IF NOT EXISTS documents_org_source_idx ON documents(org_id, source);

-- 8. Tenancy / retrieval indexes ---------------------------------------------
CREATE INDEX IF NOT EXISTS documents_org_unit_idx ON documents(org_id, unit_id);
CREATE INDEX IF NOT EXISTS document_chunks_org_idx ON document_chunks(org_id);
CREATE INDEX IF NOT EXISTS semantic_cache_org_idx ON semantic_cache(org_id);
CREATE INDEX IF NOT EXISTS query_audits_org_idx ON query_audits(org_id);
