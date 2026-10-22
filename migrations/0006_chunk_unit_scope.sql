-- Migration 0006: Denormalize the hierarchy unit onto chunks
-- Search legs filter by dc.org_id + dc.unit_id directly, so the permission
-- filter needs no join and the planner can count / scan the visible set via
-- one composite index. Triggers keep dc.unit_id equal to documents.unit_id.

ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS unit_id INT;

UPDATE document_chunks dc SET unit_id = d.unit_id
FROM documents d
WHERE dc.document_id = d.id AND dc.unit_id IS DISTINCT FROM d.unit_id;

CREATE INDEX IF NOT EXISTS document_chunks_org_unit_idx ON document_chunks(org_id, unit_id);

-- New chunks inherit the parent document's unit when the writer omits it
CREATE OR REPLACE FUNCTION document_chunks_unit_fill() RETURNS trigger AS $$
begin
  if new.unit_id is null then
    select unit_id into new.unit_id from documents where id = new.document_id;
  end if;
  return new;
end
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS document_chunks_unit_fill_trg ON document_chunks;
CREATE TRIGGER document_chunks_unit_fill_trg BEFORE INSERT
ON document_chunks FOR EACH ROW EXECUTE FUNCTION document_chunks_unit_fill();

-- Moving a document (re-upload to another unit, or unit deleted -> SET NULL)
-- moves its chunks with it
CREATE OR REPLACE FUNCTION documents_unit_sync() RETURNS trigger AS $$
begin
  update document_chunks set unit_id = new.unit_id where document_id = new.id;
  return new;
end
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_unit_sync_trg ON documents;
CREATE TRIGGER documents_unit_sync_trg AFTER UPDATE OF unit_id
ON documents FOR EACH ROW WHEN (old.unit_id IS DISTINCT FROM new.unit_id)
EXECUTE FUNCTION documents_unit_sync();
