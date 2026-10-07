import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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
    CurrentUser, get_current_user, require_org_admin, require_platform_admin,
    generate_api_key, hash_key, compute_access_key, bootstrap_platform_admin, invalidate_user_cache,
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
        bootstrap_platform_admin()
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
    role: str = Field(pattern=r"^(owner|admin|editor|viewer)$")
    # Node the user's initial role is granted at (default: caller's home unit / org root)
    unit_id: Optional[int] = None

class UpdateUserRequest(BaseModel):
    active: Optional[bool] = None

class CreateOrgRequest(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    slug: str = Field(min_length=2, max_length=100, pattern=r"^[a-z0-9\-]+$")
    owner_username: str = Field(min_length=2, max_length=100, pattern=r"^[\w.\-]+$")

class CreateUnitRequest(BaseModel):
    parent_id: int
    type: str = Field(pattern=r"^(department|team)$")
    name: str = Field(min_length=1, max_length=200)

class UpdateUnitRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)

class RoleAssignmentRequest(BaseModel):
    user_id: int
    unit_id: int
    role: str = Field(pattern=r"^(owner|admin|editor|viewer)$")

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
def metrics(user: CurrentUser = Depends(require_org_admin)):
    if not METRICS:
        raise HTTPException(status_code=404, detail="prometheus-client not installed")
    try:
        METRICS["job_backlog"].set(get_vector_store().count_queued_jobs())
    except Exception:
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/me")
def whoami(user: CurrentUser = Depends(get_current_user)):
    return {
        "id": user.id, "username": user.username, "org_id": user.org_id,
        "home_unit_id": user.home_unit_id, "role": user.role,
        "is_platform_admin": user.is_platform_admin, "org_wide": user.org_wide,
        "assignments": user.assignments,
        "can_upload": user.org_wide or bool(user.write_unit_ids),
        "write_unit_ids": user.write_unit_ids,
    }

@app.get("/api/documents")
def list_documents(limit: int = 100, offset: int = 0, user: CurrentUser = Depends(get_current_user)):
    return get_vector_store().get_all_documents(org_id=user.org_id,
                                                visible_node_ids=user.visible_node_ids,
                                                limit=limit, offset=offset)

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int, user: CurrentUser = Depends(get_current_user)):
    store = get_vector_store()
    doc = store.get_document(document_id)
    if not doc or doc["org_id"] != user.org_id:
        raise HTTPException(status_code=404, detail="Document not found")
    # Deletable by org admins, managers of the doc's unit, or the uploader.
    if not (user.can_manage(doc["unit_id"]) or doc["uploaded_by"] == user.id):
        raise HTTPException(status_code=403, detail="Not authorized to delete this document")
    store.delete_document(document_id)
    store.bump_corpus_epoch()
    logger.info("Document %d (%s) deleted by %s", document_id, doc["source"], user.username)
    return {"status": "deleted", "id": document_id}

# ---------------------------------------------------------- org provisioning

@app.post("/api/orgs", status_code=201)
def create_org(req: CreateOrgRequest, user: CurrentUser = Depends(require_platform_admin)):
    api_key = generate_api_key()
    try:
        ids = get_vector_store().create_organization(req.name, req.slug, req.owner_username, hash_key(api_key))
    except Exception:
        raise HTTPException(status_code=409, detail="Organization slug or owner username already exists")
    invalidate_user_cache()
    logger.info("Org %s (slug=%s) provisioned by %s", req.name, req.slug, user.username)
    # Owner key shown exactly once.
    return {"org_id": ids["org_id"], "root_unit_id": ids["root_unit_id"],
            "owner_username": req.owner_username, "owner_api_key": api_key}

# ----------------------------------------------------------------- hierarchy

@app.get("/api/units")
def list_units(user: CurrentUser = Depends(get_current_user)):
    return get_vector_store().get_unit_tree(user.org_id)

@app.post("/api/units", status_code=201)
def create_unit(req: CreateUnitRequest, user: CurrentUser = Depends(get_current_user)):
    store = get_vector_store()
    parent = store.get_unit(req.parent_id)
    if not parent or parent["org_id"] != user.org_id:
        raise HTTPException(status_code=404, detail="Parent unit not found")
    if not user.can_manage(req.parent_id):
        raise HTTPException(status_code=403, detail="Requires admin authority over the parent unit")
    unit_id = store.create_unit(user.org_id, req.parent_id, req.type, req.name)
    invalidate_user_cache()
    logger.info("Unit %d (%s '%s') created under %d by %s", unit_id, req.type, req.name, req.parent_id, user.username)
    return {"id": unit_id, "parent_id": req.parent_id, "type": req.type, "name": req.name}

@app.patch("/api/units/{unit_id}")
def rename_unit(unit_id: int, req: UpdateUnitRequest, user: CurrentUser = Depends(get_current_user)):
    if not user.can_manage(unit_id):
        raise HTTPException(status_code=403, detail="Requires admin authority over this unit")
    if not get_vector_store().update_unit(unit_id, user.org_id, req.name):
        raise HTTPException(status_code=404, detail="Unit not found")
    invalidate_user_cache()
    return {"id": unit_id, "name": req.name}

@app.delete("/api/units/{unit_id}", status_code=204)
def delete_unit(unit_id: int, user: CurrentUser = Depends(get_current_user)):
    if not user.can_manage(unit_id):
        raise HTTPException(status_code=403, detail="Requires admin authority over this unit")
    try:
        ok = get_vector_store().delete_unit(unit_id, user.org_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="Unit not found")
    invalidate_user_cache()

# ----------------------------------------------------- role assignments

@app.post("/api/role-assignments", status_code=201)
def add_role_assignment(req: RoleAssignmentRequest, user: CurrentUser = Depends(get_current_user)):
    store = get_vector_store()
    if not user.can_manage(req.unit_id):
        raise HTTPException(status_code=403, detail="Requires admin authority over the target unit")
    target = store.get_user(req.user_id)
    if not target or target["org_id"] != user.org_id:
        raise HTTPException(status_code=404, detail="User not found in this organization")
    unit = store.get_unit(req.unit_id)
    if not unit or unit["org_id"] != user.org_id:
        raise HTTPException(status_code=404, detail="Unit not found")
    rid = store.add_role_assignment(user.org_id, req.user_id, req.unit_id, req.role)
    invalidate_user_cache()
    logger.info("Role %s granted to user %d at unit %d by %s", req.role, req.user_id, req.unit_id, user.username)
    return {"id": rid, "user_id": req.user_id, "unit_id": req.unit_id, "role": req.role}

@app.delete("/api/role-assignments/{assignment_id}", status_code=204)
def remove_role_assignment(assignment_id: int, user: CurrentUser = Depends(require_org_admin)):
    if not get_vector_store().remove_role_assignment(assignment_id, user.org_id):
        raise HTTPException(status_code=404, detail="Assignment not found")
    invalidate_user_cache()

# ------------------------------------------------------------- user management

@app.post("/api/users", status_code=201)
def create_user(req: CreateUserRequest, user: CurrentUser = Depends(get_current_user)):
    store = get_vector_store()
    # Grant the initial role at the requested unit (default: caller's home / org root)
    unit_id = req.unit_id or user.home_unit_id or store.get_org_root_unit(user.org_id)
    if not user.can_manage(unit_id):
        raise HTTPException(status_code=403, detail="Requires admin authority over the target unit")
    api_key = generate_api_key()
    try:
        user_id = store.create_user(req.username, hash_key(api_key), req.role, user.org_id, unit_id)
        store.add_role_assignment(user.org_id, user_id, unit_id, req.role)
    except Exception:
        raise HTTPException(status_code=409, detail="Username already exists in this organization")
    invalidate_user_cache()
    logger.info("User %s (%s @unit %d) created by %s", req.username, req.role, unit_id, user.username)
    # Plaintext key returned exactly once — stored only as a hash.
    return {"id": user_id, "username": req.username, "role": req.role,
            "unit_id": unit_id, "api_key": api_key}

@app.get("/api/users")
def list_users(user: CurrentUser = Depends(require_org_admin)):
    return get_vector_store().list_users(user.org_id)

@app.patch("/api/users/{user_id}")
def update_user(user_id: int, req: UpdateUserRequest, user: CurrentUser = Depends(require_org_admin)):
    store = get_vector_store()
    target = store.get_user(user_id)
    if not target or target["org_id"] != user.org_id:
        raise HTTPException(status_code=404, detail="User not found")
    if req.active is not None:
        store.set_user_active(user_id, req.active, user.org_id)
    invalidate_user_cache()
    logger.info("User %d updated by %s (active=%s)", user_id, user.username, req.active)
    return {"id": user_id, "active": req.active}

@app.delete("/api/users/{user_id}", status_code=204)
def delete_user_endpoint(user_id: int, user: CurrentUser = Depends(require_org_admin)):
    if user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if not get_vector_store().delete_user(user_id, user.org_id):
        raise HTTPException(status_code=404, detail="User not found")
    invalidate_user_cache()
    logger.info("User %d deleted by %s", user_id, user.username)

@app.post("/api/users/{user_id}/rotate-key")
def rotate_key(user_id: int, user: CurrentUser = Depends(get_current_user)):
    """Users can rotate their own key; org admins can rotate anyone's in their org."""
    store = get_vector_store()
    if user.id != user_id:
        if not user.is_org_admin:
            raise HTTPException(status_code=403, detail="Cannot rotate another user's key")
        target = store.get_user(user_id)
        if not target or target["org_id"] != user.org_id:
            raise HTTPException(status_code=404, detail="User not found")
    new_key = generate_api_key()
    if not store.rotate_api_key(user_id, hash_key(new_key), user.org_id):
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
                    unit_id: Optional[int] = Form(None),
                    user: CurrentUser = Depends(get_current_user)):
    """
    Saves the upload and enqueues an ingestion job; the worker does the heavy
    lifting. Returns a job id to poll. `unit_id` attaches the document to a
    hierarchy node (default: caller's home unit); the doc is then visible to
    that node and everything under it. Requires write authority over the unit.
    """
    store = get_vector_store()
    target_unit = unit_id or user.home_unit_id or store.get_org_root_unit(user.org_id)
    unit = store.get_unit(target_unit)
    if not unit or unit["org_id"] != user.org_id:
        raise HTTPException(status_code=404, detail="Target unit not found")
    if not user.can_write(target_unit):
        raise HTTPException(status_code=403, detail="Requires editor authority over the target unit")

    filename = _sanitize_filename(file.filename)
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

    job_id = store.enqueue_ingest_job(filename, stored_path, org_id=user.org_id, unit_id=target_unit,
                                      enrich=enrich, uploaded_by=user.id)
    if METRICS:
        METRICS["ingest_jobs"].inc()
    logger.info("Enqueued ingest job %d for %s (%d bytes, unit=%s, by=%s)",
                job_id, filename, written, target_unit, user.username)
    return {
        "status": "queued",
        "message": f"Ingestion of {filename} queued.",
        "job_id": job_id,
    }

@app.get("/api/jobs/{job_id}")
def get_job(job_id: int, user: CurrentUser = Depends(get_current_user)):
    job = get_vector_store().get_ingest_job(job_id)
    if not job or job.get("org_id") not in (None, user.org_id):
        raise HTTPException(status_code=404, detail="Job not found")
    if not user.is_org_admin and job.get("uploaded_by") not in (None, user.id):
        raise HTTPException(status_code=404, detail="Job not found")
    return job

def _background_audit(question: str, answer: str, chunks: List[Dict],
                      org_id: Optional[int] = None, user_id: Optional[int] = None):
    """Local faithfulness audit — runs after the response is returned."""
    try:
        evaluator = Evaluator()
        res = evaluator.evaluate_faithfulness(question, answer, chunks)
        get_vector_store().save_query_audit(question, answer, res.score, res.faithful, res.reason,
                                            org_id=org_id, user_id=user_id)
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
    # RBAC context: org_id isolates the tenant; visible_node_ids (None = org-wide)
    # restricts document visibility through the hierarchy; access_key scopes cache.
    org_id = user.org_id
    visible_node_ids = user.visible_node_ids
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
    chunks, debug_info = retriever.retrieve(req.question, req.history, org_id=org_id,
                                            visible_node_ids=visible_node_ids)
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
    background_tasks.add_task(_background_audit, req.question, answer, reranked_chunks, org_id, user.id)

    # 6. Cache write (scoped to this tenant + permission set)
    write_step = tracer.start_step("Semantic Cache Write", "Save query embedding and answer (current corpus epoch)")
    try:
        vector_store.add_to_semantic_cache(req.question, query_embedding, answer, citations,
                                           org_id=org_id, access_key=access_key)
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

def _run_query_stream(req: QueryRequest, user: CurrentUser, background_tasks: BackgroundTasks):
    tracer = Tracer()
    router = app.state.router or QueryRouter()
    org_id = user.org_id
    visible_node_ids = user.visible_node_ids
    access_key = compute_access_key(user)

    # 0. Query routing
    route_step = tracer.start_step("Query Routing", "Local embedding-similarity classifier (no LLM)")
    decision = router.route(req.question)
    route_step.finish({"route": decision.route, "confidence": round(decision.confidence, 3)})

    if decision.route == "chitchat":
        yield f"data: {json.dumps({'type': 'route', 'route': 'chitchat'})}\n\n"
        words = decision.response.split(" ")
        for i, w in enumerate(words):
            token = w + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'citations': [], 'trace': tracer.to_dict(), 'debug': _debug_block(req.question, [req.question])})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'route', 'route': 'rag'})}\n\n"

    embedder = get_embedder()
    vector_store = get_vector_store()
    try:
        vector_store.connect()
    except Exception:
        yield f"data: {json.dumps({'type': 'error', 'detail': 'Database unavailable'})}\n\n"
        return

    # 1. Semantic cache check
    cache_step = tracer.start_step("Semantic Cache Check", "Local query embedding vs cache store (RBAC-scoped)")
    query_embedding = embedder.embed_query(req.question)
    cache_result = vector_store.check_semantic_cache(
        query_embedding,
        similarity_threshold=settings.similarity_threshold,
        access_key=access_key
    )

    if cache_result:
        cache_step.finish({"hit": True, "similarity": cache_result["similarity"], "cached_query": cache_result["query_text"]})
        yield f"data: {json.dumps({'type': 'cached', 'cached': True, 'citations': cache_result.get('citations') or []})}\n\n"
        words = cache_result["answer"].split(" ")
        for i, w in enumerate(words):
            token = w + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'citations': cache_result.get('citations') or [], 'trace': tracer.to_dict(), 'debug': _debug_block(req.question, [req.question])})}\n\n"
        return

    cache_step.finish({"hit": False})

    retriever = Retriever(vector_store, embedder)
    generator = Generator()

    # 2. Hybrid retrieval + weighted RRF
    retrieve_step = tracer.start_step("Retrieval & Fusion", "Adaptive transform (local rules) + dense + FTS + weighted RRF")
    chunks, debug_info = retriever.retrieve(req.question, req.history, org_id=org_id, visible_node_ids=visible_node_ids)
    retrieve_step.finish({
        "standalone_query": debug_info["standalone_query"],
        "query_type": debug_info["query_type"],
        "expanded_count": len(debug_info["expanded_queries"]),
        "fused_candidates": debug_info["fused_count"],
        "metadata_filter": debug_info["metadata_filter"],
    })

    # 3. Local cross-encoder rerank
    rerank_step = tracer.start_step("Re-ranking", "Local ONNX cross-encoder over all fused candidates")
    reranker = get_reranker()
    reranked_chunks = reranker.rerank(req.question, chunks, top_k=settings.top_k_reranked)
    rerank_step.finish({"inputs": len(chunks), "outputs": len(reranked_chunks)})

    from pipeline.generator import format_context_and_citations, filter_active_citations
    _, initial_citations = format_context_and_citations(reranked_chunks)

    # 4. Stream generation
    generate_step = tracer.start_step("Answer Generation", "Streaming grounded answer with inline citations")
    collected_tokens = []
    for token in generator.generate_stream(req.question, reranked_chunks):
        collected_tokens.append(token)
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    full_answer = "".join(collected_tokens)
    active_citations = filter_active_citations(full_answer, initial_citations)
    generate_step.finish({"citations_count": len(active_citations)})

    # 5. Faithfulness audit in background
    background_tasks.add_task(_background_audit, req.question, full_answer, reranked_chunks, org_id, user.id)

    # 6. Cache write
    write_step = tracer.start_step("Semantic Cache Write", "Save query embedding and answer (current corpus epoch)")
    try:
        vector_store.add_to_semantic_cache(
            req.question, query_embedding, full_answer, active_citations,
            org_id=org_id, access_key=access_key
        )
        write_step.finish({"success": True})
    except Exception as cache_err:
        write_step.finish({"success": False, "error": str(cache_err)})

    yield f"data: {json.dumps({
        'type': 'done',
        'citations': active_citations,
        'trace': tracer.to_dict(),
        'debug': _debug_block(
            debug_info['standalone_query'],
            debug_info['expanded_queries'],
            query_type=debug_info['query_type'],
            dense_count=debug_info['dense_count'],
            lexical_count=debug_info['lexical_count'],
            retrieved_count=debug_info['fused_count'],
            reranked_count=len(reranked_chunks),
            metadata_filter=debug_info['metadata_filter'],
        )
    })}\n\n"

@app.post("/api/query/stream")
@limiter.limit(settings.rate_limit_query)
def query_pipeline_stream(request: Request, req: QueryRequest, background_tasks: BackgroundTasks,
                          user: CurrentUser = Depends(get_current_user)):
    return StreamingResponse(
        _run_query_stream(req, user, background_tasks),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
