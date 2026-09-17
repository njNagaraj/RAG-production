-- Migration 0005: Parent-Child Hierarchical Chunking
-- Adds optional parent_content and parent_chunk_index columns to document_chunks.
-- Backward-compatible: existing corpora with NULL parent_content continue working normally.

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS parent_content TEXT,
    ADD COLUMN IF NOT EXISTS parent_chunk_index INT;

CREATE INDEX IF NOT EXISTS document_chunks_parent_idx
    ON document_chunks(document_id, parent_chunk_index);
