from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    # API Keys & Cloud Providers
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    cohere_api_key: Optional[str] = None

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None  # Supports vLLM, LocalAI, Together, etc.

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-haiku-20241022"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    groq_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_db"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_pool_min: int = 2
    db_pool_max: int = 10

    # Providers
    # generation: auto | gemini | openai | anthropic | ollama | groq | deepseek | mock
    generation_provider: str = "auto"
    # embedding: local (fastembed ONNX) | gemini | openai | ollama
    embedding_provider: str = "local"
    # reranker: local (cross-encoder ONNX) | cohere | none
    reranker_provider: str = "local"
    # evaluator: local (cross-encoder support scoring) | none
    evaluator_provider: str = "local"

    # Local model names (ONNX via fastembed, CPU-friendly)
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Multimodal ingestion: describe charts/photos at ingest time.
    # vision: ollama (local, default) | gemini (opt-in cloud) | none
    describe_images: bool = False
    vision_provider: str = "ollama"
    ollama_vision_model: str = "llama3.2-vision"
    vision_timeout_seconds: float = 120.0

    # Local model files. None = library defaults (~/.cache). Point both at a
    # baked or mounted directory and set HF_HUB_OFFLINE=1 for air-gapped hosts.
    model_cache_dir: Optional[str] = None

    # Chunking: auto (adaptive dispatcher) | parent_child | fixed | page | code | semantic
    chunking_strategy: str = "auto"
    chunk_size: int = 500
    chunk_overlap: int = 75
    parent_chunk_size: int = 800
    child_chunk_size: int = 150

    # Retrieval (IG pattern: retrieve top-20 per method, fuse, rerank all, keep 5)
    top_k_retrieval: int = 20
    top_k_reranked: int = 5
    rrf_constant: int = 60
    hnsw_ef_search: int = 80
    # Permission-scoped dense search. HNSW filters AFTER the graph walk, so a
    # narrow scope (small team) can starve the candidate list:
    # - visible set <= exact_scan_max_rows: exact scan over that set (100% recall)
    # - pgvector >= 0.8: iterative scan keeps walking until top_k rows pass the filter
    # - older pgvector: retry once with ef_search raised to hnsw_ef_search_max
    exact_scan_max_rows: int = 10000
    hnsw_iterative_scan: bool = True
    hnsw_max_scan_tuples: int = 20000
    hnsw_ef_search_max: int = 400

    # Semantic cache
    similarity_threshold: float = 0.95
    cache_ttl_hours: int = 168  # 7 days

    # Security
    # Multi-tenant RBAC: auth is DB-backed (users table). This key bootstraps the
    # platform administrator, which provisions customer organizations.
    # No users + no bootstrap key = auth disabled (dev only, synthetic org owner).
    platform_bootstrap_key: Optional[str] = None
    cors_origins: str = "*"  # comma-separated origins; "*" only for dev
    max_upload_mb: int = 50
    rate_limit_query: str = "60/minute"
    rate_limit_ingest: str = "10/minute"

    # Ingestion worker
    upload_dir: str = "uploads"
    worker_poll_seconds: float = 2.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

settings = Settings()
