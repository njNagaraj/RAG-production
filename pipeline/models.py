"""
Prefetch every local model the pipeline uses, so later runs need no network.
Used by `rag models pull` and the Dockerfile build step.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from pipeline.config import settings

logger = logging.getLogger(__name__)

def pull_models(include_docling: bool = True) -> Dict[str, str]:
    """Downloads models into the local caches. Returns {model: "ok" | error}."""
    status: Dict[str, str] = {}

    def run(name: str, fn):
        try:
            fn()
            status[name] = "ok"
        except Exception as e:
            status[name] = f"failed: {e}"
            logger.warning("Model pull failed for %s: %s", name, e)

    def embedder():
        from fastembed import TextEmbedding
        TextEmbedding(model_name=settings.local_embedding_model, cache_dir=settings.model_cache_dir or None)

    def reranker():
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        TextCrossEncoder(model_name=settings.local_reranker_model, cache_dir=settings.model_cache_dir or None)

    def tokenizer():
        # Chunker token counts; cached under HF_HOME
        from tokenizers import Tokenizer
        Tokenizer.from_pretrained(settings.local_embedding_model)

    def docling():
        # Layout, table and OCR models for PDF/DOCX parsing. Docling reads
        # DOCLING_ARTIFACTS_PATH at parse time, so download into that folder.
        try:
            from docling.utils.model_downloader import download_models
        except ImportError:
            # docling < 2.18: models live in the Hugging Face cache (HF_HOME)
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
            StandardPdfPipeline.download_models_hf()
            return
        target = os.getenv("DOCLING_ARTIFACTS_PATH")
        download_models(output_dir=Path(target) if target else None)

    run(settings.local_embedding_model, embedder)
    run(settings.local_reranker_model, reranker)
    run(f"{settings.local_embedding_model} (tokenizer)", tokenizer)
    if include_docling:
        run("docling layout/table models", docling)
    return status

def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url or "").hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama") or host.endswith(".local")

def locality_report(generator_name: str) -> List[Tuple[str, str, bool]]:
    """
    Where each stage sends data: [(stage, backend, stays_on_host)].
    generator_name is the resolved generator class (auto mode probes Ollama).
    """
    rows: List[Tuple[str, str, bool]] = []

    gen_local = generator_name in ("OllamaGenerator", "MockGenerator") or (
        generator_name == "OpenAIGenerator" and bool(settings.openai_base_url)
        and _is_local_url(settings.openai_base_url))
    rows.append(("Generation", generator_name, gen_local))

    emb = settings.embedding_provider.lower()
    rows.append(("Embedding", emb, emb in ("local", "ollama")))

    rr = settings.reranker_provider.lower()
    rows.append(("Reranking", rr, rr in ("local", "none")))

    if settings.describe_images:
        vp = settings.vision_provider.lower()
        rows.append(("Image description", vp, vp in ("ollama", "none")))

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        rows.append(("Tracing (Langfuse)", host, _is_local_url(host)))

    if os.getenv("ORT_DISABLE_TELEMETRY") != "1":
        rows.append(("Telemetry (onnxruntime)", "mobile.events.data.microsoft.com", False))

    if os.getenv("HF_HUB_OFFLINE") not in ("1", "true", "True"):
        rows.append(("Model downloads", "Hugging Face Hub on cache miss", False))
    return rows
