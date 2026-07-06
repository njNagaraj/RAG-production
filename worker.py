"""
Ingestion worker: claims queued jobs from Postgres (FOR UPDATE SKIP LOCKED, so
multiple workers are safe), runs the local ingestion pipeline, records status.
Also does periodic semantic-cache cleanup. Run as its own container.
"""
import logging
import os
import time

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import settings
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store
from pipeline.ingestion import ingest_file
from pipeline.logging_setup import setup_logging

logger = logging.getLogger("rag.worker")

CACHE_CLEANUP_INTERVAL_S = 3600

def main():
    setup_logging()
    logger.info("Worker starting: loading local models...")
    embedder = get_embedder()  # warm the ONNX model once
    store = get_vector_store()

    # Wait for DB, apply migrations (advisory-locked, safe alongside the API)
    while True:
        try:
            store.initialize_db(dimension=embedder.dimension)
            break
        except ConnectionError as e:
            logger.warning("Database not ready: %s. Retrying in 3s.", e)
            time.sleep(3)

    logger.info("Worker ready (embedding dim=%d). Polling every %.1fs.",
                embedder.dimension, settings.worker_poll_seconds)

    last_cleanup = 0.0
    while True:
        try:
            job = store.claim_next_ingest_job()
        except Exception as e:
            logger.error("Failed to poll jobs: %s", e)
            time.sleep(5)
            continue

        if job is None:
            now = time.time()
            if now - last_cleanup > CACHE_CLEANUP_INTERVAL_S:
                try:
                    store.cleanup_semantic_cache()
                    last_cleanup = now
                except Exception as e:
                    logger.warning("Cache cleanup failed: %s", e)
            time.sleep(settings.worker_poll_seconds)
            continue

        job_id = job["id"]
        logger.info("Processing job %d: %s", job_id, job["filename"])
        try:
            result = ingest_file(job["stored_path"], enrich=job["enrich"], source_name=job["filename"],
                                 allowed_groups=list(job.get("allowed_groups") or []),
                                 uploaded_by=job.get("uploaded_by"))
            store.finish_ingest_job(
                job_id, "done",
                chunks_count=result["chunks_count"],
                document_id=result["document_id"],
            )
            logger.info("Job %d done: %d chunks", job_id, result["chunks_count"])
        except Exception as e:
            logger.exception("Job %d failed", job_id)
            store.finish_ingest_job(job_id, "failed", error=str(e)[:2000])
        finally:
            try:
                if os.path.exists(job["stored_path"]):
                    os.remove(job["stored_path"])
            except OSError:
                pass

if __name__ == "__main__":
    main()
