import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pipeline.config import settings
from pipeline.logging_setup import setup_logging, request_id_var
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from pipeline.evaluator import Evaluator
from pipeline.reranker import get_reranker
from pipeline.router import QueryRouter
from pipeline.tracer import Tracer
from pipeline.auth import (
    CurrentUser, get_current_user, require_role, generate_api_key, hash_key,
    compute_access_key, bootstrap_admin, invalidate_user_cache,
)

logger = logging.getLogger("rag.api")

# --------------------------------------------------------------------- metrics

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    METRICS = {
        "queries": Counter("rag_queries_total", "Total /api/query requests", ["route", "cached"]),
        "latency": Histogram("rag_query_latency_seconds", "Query latency"),
        "ingest_jobs": Counter("rag_ingest_jobs_total", "Ingest jobs enqueued"),
        "job_backlog": Gauge("rag_job_backlog", "Queued + processing ingest jobs"),
    }
except ImportError:
    METRICS = None

# -------------------------------------------------------------------- lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Loading local models (embedder, reranker, router)...")
    embedder = get_embedder()
    get_reranker()
    try:
        vector_store = get_vector_store()
        vector_store.initialize_db(dimension=embedder.dimension)
        bootstrap_admin()
        app.state.router = QueryRouter()  # embeds seed examples once
        app.state.db_ready = True
    except Exception as e:
        # API can start before Postgres; readiness reports the truth
        logger.error("Startup DB init failed (will retry on /ready): %s", e)
        app.state.router = None
        app.state.db_ready = False
    logger.info("Startup complete. embedding_dim=%d", embedder.dimension)
    yield
    try:
        get_vector_store().pool.close()
    except Exception:
        pass

app = FastAPI(title="Production RAG Assistant API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ middleware

@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id_var.get()},
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

# --------------------------------------------------------------------- schemas

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: Optional[List[Dict[str, str]]] = []

class CreateUserRequest(BaseModel):
    username: str = Field(min_length=2, max_length=100, pattern=r"^[\w.\-]+$")
    role: str = Field(pattern=r"^(admin|editor|viewer)$")
    groups: List[str] = []

class UpdateUserRequest(BaseModel):
    active: Optional[bool] = None
    role: Optional[str] = Field(None, pattern=r"^(admin|editor|viewer)$")
    groups: Optional[List[str]] = None

# ------------------------------------------------------------------- endpoints

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    try:
        store = get_vector_store()
        store.connect()
        store.get_corpus_epoch()
        if not app.state.db_ready:
            embedder = get_embedder()
            store.initialize_db(dimension=embedder.dimension)
            app.state.router = QueryRouter()
            app.state.db_ready = True
        return {"status": "ready"}
    except Exception as e:
        logger.warning("Readiness failed: %s", e)
        return JSONResponse(status_code=503, content={"status": "not_ready"})

@app.get("/metrics")
def metrics(user: CurrentUser = Depends(require_role("admin"))):
    if not METRICS:
        raise HTTPException(status_code=404, detail="prometheus-client not installed")
    try:
        METRICS["job_backlog"].set(get_vector_store().count_queued_jobs())
    except Exception:
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/me")
def whoami(user: CurrentUser = Depends(get_current_user)):
    return {"id": user.id, "username": user.username, "role": user.role, "groups": user.groups}

@app.get("/api/documents")
def list_documents(limit: int = 100, offset: int = 0, user: CurrentUser = Depends(get_current_user)):
    return get_vector_store().get_all_documents(limit=limit, offset=offset,
                                                allowed_groups=user.retrieval_groups())

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int, user: CurrentUser = Depends(require_role("admin", "editor"))):
    store = get_vector_store()
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not user.is_admin and doc["uploaded_by"] != user.id:
        raise HTTPException(status_code=403, detail="Editors can only delete their own documents")
    store.delete_document(document_id)
    store.bump_corpus_epoch()
    logger.info("Document %d (%s) deleted by %s", document_id, doc["source"], user.username)
    return {"status": "deleted", "id": document_id}

# ------------------------------------------------------------- user management

@app.post("/api/users", status_code=201)
def create_user(req: CreateUserRequest, user: CurrentUser = Depends(require_role("admin"))):
    api_key = generate_api_key()
    try:
        user_id = get_vector_store().create_user(req.username, hash_key(api_key), req.role,
                                                 [g.strip() for g in req.groups if g.strip()])
    except Exception:
        raise HTTPException(status_code=409, detail="Username already exists")
    invalidate_user_cache()
    logger.info("User %s (%s) created by %s", req.username, req.role, user.username)
    # The plaintext key is returned exactly once — it is stored only as a hash.
    return {"id": user_id, "username": req.username, "role": req.role,
            "groups": req.groups, "api_key": api_key}

@app.get("/api/users")
def list_users(user: CurrentUser = Depends(require_role("admin"))):
    return get_vector_store().list_users()

@app.patch("/api/users/{user_id}")
def update_user(user_id: int, req: UpdateUserRequest, user: CurrentUser = Depends(require_role("admin"))):
    store = get_vector_store()
    updated = store.update_user(
        user_id,
        role=req.role,
        groups=[g.strip() for g in req.groups if g.strip()] if req.groups is not None else None,
        active=req.active,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    invalidate_user_cache()
    logger.info("User %d updated by %s (role=%s groups=%s active=%s)",
                user_id, user.username, req.role, req.groups, req.active)
    return {"id": user_id, "role": req.role, "groups": req.groups, "active": req.active}

@app.delete("/api/users/{user_id}", status_code=204)
def delete_user_endpoint(user_id: int, user: CurrentUser = Depends(require_role("admin"))):
    if user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if not get_vector_store().delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    invalidate_user_cache()
    logger.info("User %d deleted by %s", user_id, user.username)

@app.post("/api/users/{user_id}/rotate-key")
def rotate_key(user_id: int, user: CurrentUser = Depends(get_current_user)):
    """Users can rotate their own key; admins can rotate anyone's key."""
    if not user.is_admin and user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot rotate another user's key")
    new_key = generate_api_key()
    if not get_vector_store().rotate_api_key(user_id, hash_key(new_key)):
        raise HTTPException(status_code=404, detail="User not found")
    invalidate_user_cache()
    logger.info("API key rotated for user %d by %s", user_id, user.username)
    return {"api_key": new_key}  # shown exactly once — not stored

_SAFE_FILENAME_RE = re.compile(r"[^\w.() \-]")

def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "")
    base = _SAFE_FILENAME_RE.sub("_", base).strip(" .")
    if not base:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return base[:200]

@app.post("/api/ingest", status_code=202)
@limiter.limit(settings.rate_limit_ingest)
def ingest_endpoint(request: Request, file: UploadFile = File(...), enrich: bool = Form(False),
                    groups: str = Form(""),
                    user: CurrentUser = Depends(require_role("admin", "editor"))):
    """
    Saves the upload and enqueues an ingestion job; the worker container does
    the heavy lifting. Returns immediately with a job id to poll.
    `groups`: comma-separated access groups; empty = visible to everyone.
    """
    filename = _sanitize_filename(file.filename)
    allowed_groups = [g.strip() for g in groups.split(",") if g.strip()]
    os.makedirs(settings.upload_dir, exist_ok=True)
    stored_path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}_{filename}")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with open(stored_path, "wb") as out:
            while True:
                data = file.file.read(1024 * 1024)
                if not data:
                    break
                written += len(data)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
                out.write(data)
    except HTTPException:
        if os.path.exists(stored_path):
            os.remove(stored_path)
        raise

    job_id = get_vector_store().enqueue_ingest_job(filename, stored_path, enrich,
                                                   allowed_groups=allowed_groups,
                                                   uploaded_by=user.id)
    if METRICS:
        METRICS["ingest_jobs"].inc()
    logger.info("Enqueued ingest job %d for %s (%d bytes, groups=%s, by=%s)",
                job_id, filename, written, allowed_groups, user.username)
    return {
        "status": "queued",
        "message": f"Ingestion of {filename} queued.",
        "job_id": job_id,
    }

@app.get("/api/jobs/{job_id}")
def get_job(job_id: int, user: CurrentUser = Depends(get_current_user)):
    job = get_vector_store().get_ingest_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not user.is_admin and job.get("uploaded_by") not in (None, user.id):
        raise HTTPException(status_code=404, detail="Job not found")
    return job

def _background_audit(question: str, answer: str, chunks: List[Dict], user_id: Optional[int] = None):
    """Local faithfulness audit — runs after the response is returned."""
    try:
        evaluator = Evaluator()
        res = evaluator.evaluate_faithfulness(question, answer, chunks)
        get_vector_store().save_query_audit(question, answer, res.score, res.faithful, res.reason, user_id=user_id)
        logger.info("Audit: score=%.2f faithful=%s", res.score, res.faithful)
    except Exception as e:
        logger.warning("Background audit failed: %s", e)

def _debug_block(standalone: str, queries: List[str], **kw) -> Dict:
    base = {
        "standalone_query": standalone,
        "expanded_queries": queries,
        "query_type": kw.get("query_type", "n/a"),
        "dense_count": 0,
        "lexical_count": 0,
        "retrieved_count": 0,
        "reranked_count": 0,
        "metadata_filter": {},
    }
    base.update(kw)
    return base

@app.post("/api/query")
@limiter.limit(settings.rate_limit_query)
def query_pipeline(request: Request, req: QueryRequest, background_tasks: BackgroundTasks,
                   user: CurrentUser = Depends(get_current_user)):
    tracer = Tracer()
    timer_ctx = METRICS["latency"].time() if METRICS else None
    if timer_ctx:
        timer_ctx.__enter__()
    try:
        return _run_query(req, tracer, background_tasks, user)
    finally:
        if timer_ctx:
            timer_ctx.__exit__(None, None, None)

def _run_query(req: QueryRequest, tracer: Tracer, background_tasks: BackgroundTasks, user: CurrentUser):
    router = app.state.router or QueryRouter()
    # RBAC context: None = unrestricted (admin); list = group-filtered retrieval
    allowed_groups = user.retrieval_groups()
    access_key = compute_access_key(user)

    # 0. Query routing — local embedding-similarity classifier
    route_step = tracer.start_step("Query Routing", "Local embedding-similarity classifier (no LLM)")
    decision = router.route(req.question)
    route_step.finish({"route": decision.route, "confidence": round(decision.confidence, 3)})

    if decision.route == "chitchat":
        if METRICS:
            METRICS["queries"].labels(route="chitchat", cached="false").inc()
        tracer.log_to_langfuse("RAG Query (Chitchat)", req.question, decision.response, {"route": "chitchat"})
        return {
            "answer": decision.response,
            "citations": [],
            "faithfulness": {"score": 1.0, "faithful": True, "reason": "Routed as conversational chitchat (local router)."},
            "cached": False,
            "trace": tracer.to_dict(),
            "debug": _debug_block(req.question, [req.question]),
        }

    embedder = get_embedder()
    vector_store = get_vector_store()
    try:
        vector_store.connect()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # 1. Semantic cache (local embedding; scoped to the caller's permission set)
    cache_step = tracer.start_step("Semantic Cache Check", "Local query embedding vs cache store (RBAC-scoped)")
    query_embedding = embedder.embed_query(req.question)
    cache_result = vector_store.check_semantic_cache(query_embedding,
                                                     similarity_threshold=settings.similarity_threshold,
                                                     access_key=access_key)

    if cache_result:
        cache_step.finish({"hit": True, "similarity": cache_result["similarity"], "cached_query": cache_result["query_text"]})
        if METRICS:
            METRICS["queries"].labels(route="rag", cached="true").inc()
        tracer.log_to_langfuse("RAG Query (Cached)", req.question, cache_result["answer"],
                               {"cached": True, "similarity": cache_result["similarity"]})
        return {
            "answer": cache_result["answer"],
            "citations": cache_result["citations"],
            "faithfulness": {"score": 1.0, "faithful": True,
                             "reason": f"Served from Semantic Cache (Similarity: {cache_result['similarity']:.2f})."},
            "cached": True,
            "trace": tracer.to_dict(),
            "debug": _debug_block(req.question, [req.question]),
        }
    cache_step.finish({"hit": False})

    retriever = Retriever(vector_store, embedder)
    generator = Generator()

    # 2. Local adaptive transform + hybrid retrieval + weighted RRF (group-filtered in SQL)
    retrieve_step = tracer.start_step("Retrieval & Fusion", "Adaptive transform (local rules) + dense + FTS + weighted RRF")
    chunks, debug_info = retriever.retrieve(req.question, req.history, allowed_groups=allowed_groups)
    retrieve_step.finish({
        "standalone_query": debug_info["standalone_query"],
        "query_type": debug_info["query_type"],
        "expanded_count": len(debug_info["expanded_queries"]),
        "fused_candidates": debug_info["fused_count"],
        "metadata_filter": debug_info["metadata_filter"],
    })

    # 3. Local cross-encoder rerank of ALL fused candidates
    rerank_step = tracer.start_step("Re-ranking", "Local ONNX cross-encoder over all fused candidates")
    reranker = get_reranker()
    reranked_chunks = reranker.rerank(req.question, chunks, top_k=settings.top_k_reranked)
    rerank_step.finish({"inputs": len(chunks), "outputs": len(reranked_chunks)})

    # 4. Grounded generation — the single cloud LLM call
    generate_step = tracer.start_step("Answer Generation", "Gemini grounded answer with inline citations (only cloud call)")
    answer, citations = generator.generate(req.question, reranked_chunks)
    generate_step.finish({"citations_count": len(citations)})

    # 5. Faithfulness audit — local cross-encoder, after response (background)
    background_tasks.add_task(_background_audit, req.question, answer, reranked_chunks, user.id)

    # 6. Cache write (scoped to this permission set)
    write_step = tracer.start_step("Semantic Cache Write", "Save query embedding and answer (current corpus epoch)")
    try:
        vector_store.add_to_semantic_cache(req.question, query_embedding, answer, citations,
                                           access_key=access_key)
        write_step.finish({"success": True})
    except Exception as cache_err:
        write_step.finish({"success": False, "error": str(cache_err)})

    if METRICS:
        METRICS["queries"].labels(route="rag", cached="false").inc()
    tracer.log_to_langfuse("RAG Query (Miss)", req.question, answer, {"cached": False})

    return {
        "answer": answer,
        "citations": citations,
        "faithfulness": {
            "score": None,
            "faithful": None,
            "reason": "Faithfulness audited in background by local cross-encoder; results stored in query_audits.",
        },
        "cached": False,
        "trace": tracer.to_dict(),
        "debug": _debug_block(
            debug_info["standalone_query"],
            debug_info["expanded_queries"],
            query_type=debug_info["query_type"],
            dense_count=debug_info["dense_count"],
            lexical_count=debug_info["lexical_count"],
            retrieved_count=debug_info["fused_count"],
            reranked_count=len(reranked_chunks),
            metadata_filter=debug_info["metadata_filter"],
        ),
    }

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
