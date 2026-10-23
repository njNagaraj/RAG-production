import argparse
import sys
import os
import time
import glob
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import settings
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store
from pipeline.retriever import Retriever
from pipeline.generator import get_generator
from pipeline.evaluator import Evaluator
from pipeline.reranker import get_reranker
from pipeline.ingestion import ingest_file

SUPPORTED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx", "*.py", "*.js", "*.ts", "*.java", "*.go", "*.csv", "*.json")

BANNER = """
===================================================================
       PRODUCTION RAG - Enterprise Local-First RAG Engine          
  Developer-Agnostic | Hybrid Search | Multi-Tenant | Guardrails   
===================================================================
"""

def print_banner():
    print(BANNER)

def _resolve_org(vector_store, slug: str):
    """Resolve org slug -> (org_id, root_unit_id), creating 'default' if needed."""
    if slug == "default":
        ids = vector_store.ensure_default_org()
        return ids["org_id"], ids["root_unit_id"]
    for org in vector_store.list_organizations():
        if org["slug"] == slug:
            return org["id"], vector_store.get_org_root_unit(org["id"])
    print(f"[!] Organization '{slug}' not found.")
    sys.exit(1)

def handle_doctor():
    """System diagnostic: verify database, pgvector, embedding models, and LLM configuration."""
    print("\n[*] Running Production RAG Diagnostics (doctor)...")
    all_ok = True

    # 1. Config & Providers
    print(f"\n[1] Configuration:")
    print(f"    - Generation Provider: {settings.generation_provider}")
    print(f"    - Embedding Provider:  {settings.embedding_provider} ({settings.local_embedding_model})")
    print(f"    - Reranker Provider:   {settings.reranker_provider} ({settings.local_reranker_model})")
    print(f"    - Chunking Strategy:   {settings.chunking_strategy}")

    # 2. Database Connection
    print(f"\n[2] Database Health:")
    try:
        store = get_vector_store()
        store.connect()
        print(f"    [+] Connected to Postgres at {settings.db_host}:{settings.db_port}/{settings.db_name}")
        with store.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                row = cur.fetchone()
                if row:
                    print(f"    [+] pgvector extension active (version {row['extversion']})")
                else:
                    print(f"    [!] pgvector extension NOT detected.")
                    all_ok = False
                
                # Check tables
                cur.execute("SELECT COUNT(*) as cnt FROM documents;")
                docs_cnt = cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
                chunks_cnt = cur.fetchone()["cnt"]
                print(f"    [+] Corpus Stats: {docs_cnt} documents, {chunks_cnt} chunks indexed.")
    except Exception as e:
        print(f"    [!] Database check failed: {e}")
        all_ok = False

    # 3. Model Engine
    print(f"\n[3] Model Engine:")
    try:
        embedder = get_embedder()
        print(f"    [+] Embedder ready. Dimension: {embedder.dimension}")
    except Exception as e:
        print(f"    [!] Embedder failed: {e}")
        all_ok = False

    generator_name = "unavailable"
    try:
        generator = get_generator()
        generator_name = generator.__class__.__name__
        print(f"    [+] Generator provider: {generator_name}")
    except Exception as e:
        print(f"    [!] Generator init failed: {e}")
        all_ok = False

    # 4. Data locality: which stages send data off this host
    from pipeline.models import locality_report
    print("\n[4] Data Locality:")
    remote = 0
    for stage, backend, local in locality_report(generator_name):
        print(f"    [{'local' if local else 'REMOTE'}] {stage}: {backend}")
        remote += not local
    print("    [+] 100% local: no stage leaves this host." if remote == 0
          else f"    [i] {remote} stage(s) can reach the network. See .env.example for local options.")

    print("\n" + "=" * 60)
    if all_ok:
        print("[+] System Diagnostics PASSED! Ready for production traffic.")
    else:
        print("[!] Diagnostics detected issues. Review logs above.")
    print("=" * 60 + "\n")

def handle_ingest(path: str, enrich: bool, org_slug: str = "default", unit_id: int = None):
    print(f"[*] Starting ingestion for path: {path}")

    files = []
    if os.path.isdir(path):
        for ext in SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
    elif os.path.isfile(path):
        files.append(path)
    else:
        print(f"[!] Path not found: {path}")
        sys.exit(1)

    if not files:
        print(f"[!] No supported documents found ({', '.join(SUPPORTED_EXTS)}).")
        return

    print(f"[*] Found {len(files)} file(s) to process.")

    embedder = get_embedder()
    vector_store = get_vector_store()
    print("[*] Initializing database schema / applying migrations...")
    vector_store.initialize_db(dimension=embedder.dimension)

    org_id, root_unit = _resolve_org(vector_store, org_slug)
    target_unit = unit_id or root_unit
    print(f"[*] Scope: org='{org_slug}' (id={org_id}), unit={target_unit}")

    failed = 0
    t0 = time.time()
    for f in files:
        try:
            result = ingest_file(f, org_id=org_id, unit_id=target_unit, enrich=enrich)
            print(f"  [+] Ingested {result['source']}: {result['chunks_count']} chunks (strategy: {result['chunking_strategy']})")
        except Exception as e:
            failed += 1
            print(f"  [!] Failed to ingest {os.path.basename(f)}: {e}")

    elapsed = time.time() - t0
    print(f"\n[+] Ingestion complete in {elapsed:.2f}s: {len(files) - failed} succeeded, {failed} failed.")

def handle_query(query_text: str, org_slug: str = "default"):
    print(f"\n[*] Question: {query_text}\n")
    t0 = time.time()

    embedder = get_embedder()
    vector_store = get_vector_store()

    try:
        vector_store.connect()
    except ConnectionError as ce:
        print(f"[!] Database Connection Error: {ce}")
        sys.exit(1)

    org_id, _ = _resolve_org(vector_store, org_slug)
    retriever = Retriever(vector_store, embedder)
    generator = get_generator()
    evaluator = Evaluator()

    # 1. Retrieval
    t_ret0 = time.time()
    chunks, debug_info = retriever.retrieve(query_text, org_id=org_id, visible_node_ids=None)
    t_ret = time.time() - t_ret0

    # 2. Rerank
    t_rerank0 = time.time()
    reranker = get_reranker()
    reranked_chunks = reranker.rerank(query_text, chunks, top_k=settings.top_k_reranked)
    t_rerank = time.time() - t_rerank0

    # 3. Generation
    t_gen0 = time.time()
    answer, citations = generator.generate(query_text, reranked_chunks)
    t_gen = time.time() - t_gen0

    # 4. Evaluation
    eval_res = evaluator.evaluate_faithfulness(query_text, answer, reranked_chunks)
    total_time = time.time() - t0

    print("=" * 65)
    print("ANSWER:")
    print("=" * 65)
    print(answer)
    print("=" * 65)

    if citations:
        print("\nCITATIONS:")
        for cit in citations:
            print(f"  [{cit['id']}] {cit['source']} (Page {cit['page']})")

    print("\nPERFORMANCE & QUALITY METRICS:")
    print(f"  - Pipeline Latency:  {total_time * 1000:.1f}ms (Retrieval: {t_ret*1000:.1f}ms, Rerank: {t_rerank*1000:.1f}ms, Gen: {t_gen*1000:.1f}ms)")
    print(f"  - Query Route/Type:  {debug_info.get('query_type', 'conceptual')}")
    print(f"  - Retrieved Chunks:  {len(chunks)} fused candidates -> {len(reranked_chunks)} reranked")
    print(f"  - Faithfulness:      {eval_res.score:.2f} ({'Grounded' if eval_res.faithful else 'Hallucination Warning'})")
    print(f"  - Faithfulness Note: {eval_res.reason}\n")

def handle_serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Launch the FastAPI server."""
    import uvicorn
    print(f"[*] Launching Production RAG Server on http://{host}:{port}...")
    uvicorn.run("app:app", host=host, port=port, reload=reload)

def handle_benchmark():
    """Benchmark local retrieval and reranking latency."""
    print("\n[*] Running Performance Benchmark...")
    embedder = get_embedder()
    vector_store = get_vector_store()
    vector_store.connect()
    retriever = Retriever(vector_store, embedder)
    reranker = get_reranker()

    test_queries = [
        "What is the vacation and annual leave policy?",
        "How are expense reimbursements processed for travel?",
        "Security requirements for password rotation and MFA",
        "Termination and severance guidelines",
    ]

    latencies = []
    print(f"[*] Benchmarking {len(test_queries)} queries across Hybrid Search + Cross-Encoder...")
    for q in test_queries:
        t0 = time.time()
        chunks, _ = retriever.retrieve(q, org_id=1, visible_node_ids=None)
        reranked = reranker.rerank(q, chunks, top_k=5)
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        print(f"  - Query: \"{q[:40]}...\" -> {len(reranked)} chunks in {dt:.1f}ms")

    avg_lat = sum(latencies) / len(latencies)
    print(f"\n[+] Benchmark Complete: Average Hybrid+Rerank Latency = {avg_lat:.1f}ms\n")

def handle_eval(skip_generation: bool = False, org_slug: str = "default"):
    """Run automated golden dataset continuous evaluation."""
    from eval_suite import run_evaluation_suite
    if not run_evaluation_suite(skip_generation=skip_generation, org_slug=org_slug):
        sys.exit(1)

def handle_models(skip_docling: bool = False):
    from pipeline.models import pull_models
    status = pull_models(include_docling=not skip_docling)
    failed = 0
    for name, result in status.items():
        print(f"  [{'OK' if result == 'ok' else 'FAIL'}] {name}" + ("" if result == "ok" else f": {result}"))
        failed += result != "ok"
    if failed:
        sys.exit(1)
    print("All models cached. Set HF_HUB_OFFLINE=1 to block any further downloads.")

def main():
    parser = argparse.ArgumentParser(
        description="Production RAG - Enterprise Developer-Agnostic RAG Engine",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a document or directory into the vector store")
    p_ingest.add_argument("path", help="Path to file or directory to ingest")
    p_ingest.add_argument("--enrich", action="store_true", help="Describe charts/images at ingest time")
    p_ingest.add_argument("--org", default="default", help="Organization slug (default: 'default')")
    p_ingest.add_argument("--unit", type=int, default=None, help="Org-unit ID")

    # query
    p_query = subparsers.add_parser("query", help="Query the RAG pipeline")
    p_query.add_argument("text", help="The question to ask")
    p_query.add_argument("--org", default="default", help="Organization slug (default: 'default')")

    # doctor
    subparsers.add_parser("doctor", help="Run system diagnostics and verify database/models")

    # benchmark
    subparsers.add_parser("benchmark", help="Run retrieval and reranker latency benchmarks")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run automated golden dataset evaluation (Recall@K, MRR, Faithfulness)")
    p_eval.add_argument("--skip-generation", action="store_true", help="Run retrieval-only evaluation without generating answers")
    p_eval.add_argument("--org", default="default", help="Organization slug to evaluate (default: 'default')")

    # bench
    p_bench = subparsers.add_parser("bench", help="Retrieval quality benchmark on a BEIR dataset (nDCG@10, Recall@10, MRR@10)")
    p_bench.add_argument("--dataset", default="scifact", help="BEIR dataset name on Hugging Face (default: scifact)")
    p_bench.add_argument("--limit", type=int, default=0, help="Only the first N test queries (0 = all)")
    p_bench.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder row")

    # models
    p_models = subparsers.add_parser("models", help="Download local models for offline use")
    p_models.add_argument("action", choices=["pull"], help="pull: fetch embedder, reranker, tokenizer, Docling models")
    p_models.add_argument("--skip-docling", action="store_true", help="Skip the Docling layout/table models")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the API web server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable code auto-reload")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args.path, args.enrich, args.org, args.unit)
    elif args.command == "query":
        handle_query(args.text, args.org)
    elif args.command == "doctor":
        handle_doctor()
    elif args.command == "benchmark":
        handle_benchmark()
    elif args.command == "eval":
        handle_eval(args.skip_generation, args.org)
    elif args.command == "bench":
        from bench_beir import run_benchmark
        run_benchmark(args.dataset, args.limit, rerank=not args.no_rerank)
    elif args.command == "models":
        handle_models(args.skip_docling)
    elif args.command == "serve":
        handle_serve(args.host, args.port, args.reload)
    else:
        print_banner()
        parser.print_help()

if __name__ == "__main__":
    main()
