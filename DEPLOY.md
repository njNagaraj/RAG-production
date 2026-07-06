# Deployment Guide

Local-first RAG: every pipeline step (routing, retrieval, reranking, evaluation,
embeddings, parsing/OCR) runs on local ONNX models inside the containers.
The only cloud calls are Gemini for **final answer generation** and (optionally,
`DESCRIBE_IMAGES=true`) chart/image descriptions at ingestion time.

## Architecture

```
            ┌─────────────┐   enqueue    ┌──────────────┐
  client ──►│  api (:8000)│─────────────►│  postgres    │◄── worker
            │  gunicorn   │   jobs table │  + pgvector  │    (ingestion:
            │  2 workers  │◄─────────────│              │     parse/chunk/
            └─────────────┘   search     └──────────────┘     embed/COPY)
                   │ shared uploads volume ▲
                   └───────────────────────┘
```

- `api` — FastAPI. Query path: local router → semantic cache → hybrid retrieval
  (dense HNSW + Postgres FTS, weighted RRF) → local cross-encoder rerank →
  Gemini generation → background local faithfulness audit.
- `worker` — polls `ingest_jobs`, runs Docling parsing (tables/OCR), adaptive
  chunking, batched local embeddings, bulk `COPY` insert. Scale with
  `docker compose up --scale worker=3` (job claiming uses `SKIP LOCKED`).
- `postgres` — pgvector HNSW (m=16, ef_construction=64) + GIN FTS index.

## Quick start (any cloud VM)

```bash
cp .env.example .env        # set GEMINI_API_KEY, DB_PASSWORD, API_KEYS, CORS_ORIGINS
docker compose up -d --build
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Ingest and query:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "X-API-Key: $KEY" -F "file=@policy.pdf"
# → {"status":"queued","job_id":1}
curl -H "X-API-Key: $KEY" http://localhost:8000/api/jobs/1

curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question":"What is the casual leave policy?"}'
```

## VM sizing

| Corpus | VM | Notes |
|---|---|---|
| ≤ 1k docs | 2 vCPU / 4 GB | baseline |
| 1k–10k docs | 4 vCPU / 8 GB | recommended default |
| 10k+ docs | 8 vCPU / 16 GB | scale workers; consider partial HNSW rebuilds off-hours |

Bulk backfill tip: for very large first-time loads, drop
`document_chunks_hnsw_idx`, ingest, then recreate the index (much faster build).

## Production checklist

- [ ] `API_KEYS` set (auth is disabled when empty)
- [ ] `CORS_ORIGINS` set to your frontend origin(s), not `*`
- [ ] `DB_PASSWORD` changed; Postgres port not exposed publicly (remove the
      `ports:` mapping from the postgres service if the API is the only client)
- [ ] TLS via reverse proxy in front of :8000, e.g. Caddy:

```
your.domain.com {
    reverse_proxy localhost:8000
}
```

- [ ] Backups: `docker exec rag_postgres pg_dump -U postgres rag_db | gzip > backup.sql.gz`
      on a daily cron; keep 7 days.
- [ ] Monitoring: scrape `GET /metrics` (Prometheus) — query rate, cache hit
      rate, latency histogram, ingest job backlog. Logs are JSON on stdout
      (ship with any log driver).

## Changing the embedding model

The embedding dimension is baked into the schema. Changing
`LOCAL_EMBEDDING_MODEL` to a model with a different dimension makes startup
fail with instructions (by design — no silent data loss). Procedure: back up
DB → add a migration recreating `document_chunks`/`semantic_cache` with the new
dimension → re-ingest the corpus.
