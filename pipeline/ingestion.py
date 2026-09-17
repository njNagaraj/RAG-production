import logging
import os
from typing import Dict, Any, Optional

from pipeline.config import settings
from pipeline.parser import DoclingParser, get_vision_client
from pipeline.chunker import Chunker
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store

logger = logging.getLogger("rag.ingestion")

_parser: Optional[DoclingParser] = None

def _get_parser() -> DoclingParser:
    global _parser
    if _parser is None:
        _parser = DoclingParser()
    return _parser

def ingest_file(file_path: str, org_id: int, unit_id: int, enrich: bool = False,
                source_name: Optional[str] = None, uploaded_by: Optional[int] = None) -> Dict[str, Any]:
    """
    Full local ingestion pipeline for one file:
    parse (Docling: structure/tables/OCR, optional Gemini vision for charts)
    -> adaptive chunking -> local embeddings (batched) -> bulk COPY insert
    -> bump corpus epoch (invalidates semantic cache).

    org_id/unit_id place the document in the tenant hierarchy (visible to that
    node and everything under it). Shared by the worker container and the CLI.
    """
    parser = _get_parser()
    chunker = Chunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embedder = get_embedder()
    vector_store = get_vector_store()

    vision_client = get_vision_client() if (enrich or settings.describe_images) else None

    logger.info("Parsing %s", file_path)
    doc_data = parser.parse(file_path, client=vision_client)

    if source_name:
        # Uploaded files are stored under job-scoped names; keep the original
        # filename as the citation source.
        doc_data["metadata"]["source"] = source_name
        doc_data["metadata"]["title"] = doc_data["metadata"].get("title") or source_name
    doc_data["metadata"]["file_path"] = source_name or os.path.basename(file_path)
    doc_data["metadata"]["org_id"] = org_id
    doc_data["metadata"]["unit_id"] = unit_id
    doc_data["metadata"]["uploaded_by"] = uploaded_by

    chunks = chunker.split_document(doc_data)
    if not chunks:
        raise ValueError(f"No text content extracted from {doc_data['metadata']['source']}")
    chunks = chunker.enrich_context(chunks, doc_data.get("content", ""))

    logger.info("Embedding %d chunks (strategy=%s)", len(chunks),
                chunks[0].doc_metadata.get("chunking_strategy", "?"))
    embeddings = embedder.embed_documents([c.full_text for c in chunks])

    doc_id = vector_store.insert_document(doc_data["metadata"])
    vector_store.insert_chunks(doc_id, [c.to_dict() for c in chunks], embeddings, org_id=org_id)
    vector_store.bump_corpus_epoch()

    logger.info("Ingested %s: doc_id=%s chunks=%d", doc_data["metadata"]["source"], doc_id, len(chunks))
    return {
        "document_id": doc_id,
        "chunks_count": len(chunks),
        "source": doc_data["metadata"]["source"],
        "chunking_strategy": chunks[0].doc_metadata.get("chunking_strategy"),
    }
