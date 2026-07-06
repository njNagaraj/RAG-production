from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    # API Keys (Gemini is used ONLY for final answer generation and
    # optional ingestion-time image descriptions)
    gemini_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_db"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_pool_min: int = 2
    db_pool_max: int = 10

    # Providers
    # generation: gemini (the single sanctioned cloud call)
    generation_provider: str = "gemini"
    # embedding: local (fastembed ONNX) | gemini
    embedding_provider: str = "local"
    # reranker: local (cross-encoder ONNX) | cohere | none
    reranker_provider: str = "local"
    # evaluator: local (cross-encoder support scoring) | none
    evaluator_provider: str = "local"

    # Local model names (ONNX via fastembed, CPU-friendly)
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Multimodal ingestion: describe charts/photos with Gemini vision at
    # ingest time (one-off per document; query time stays fully local)
    describe_images: bool = False

    # Chunking: auto (adaptive dispatcher) | fixed | page | code | semantic
    chunking_strategy: str = "auto"
    chunk_size: int = 500
    chunk_overlap: int = 75

    # Retrieval (IG pattern: retrieve top-20 per method, fuse, rerank all, keep 5)
    top_k_retrieval: int = 20
    top_k_reranked: int = 5
    rrf_constant: int = 60
    hnsw_ef_search: int = 80

    # Semantic cache
    similarity_threshold: float = 0.95
    cache_ttl_hours: int = 168  # 7 days

    # Security
    # RBAC: auth is DB-backed (users table). This key bootstraps the first
    # admin account on startup. No users + no bootstrap key = auth disabled (dev only).
    admin_bootstrap_key: Optional[str] = None
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
