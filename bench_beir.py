"""
Reproducible retrieval benchmark on a BEIR dataset (default: SciFact, 5,183
abstracts, 300 test queries). Runs the real pipeline on this machine:
Chunker -> local embedder -> Postgres (dense + FTS) -> fusion -> cross-encoder.

Usage: rag bench --dataset scifact [--limit 100] [--no-rerank]
Needs: pip install -e ".[bench]" (pyarrow), Postgres, and the local models.
"""
import json
import math
import os
import secrets
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

from pipeline.auth import hash_key
from pipeline.chunker import Chunker
from pipeline.config import settings
from pipeline.embedder import get_embedder
from pipeline.reranker import get_reranker
from pipeline.retriever import Retriever
from pipeline.vector_store import get_vector_store

K = 10
INGEST_BATCH = 256
RESULTS_DIR = "benchmarks"

def load_beir(name: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Downloads a BEIR dataset from Hugging Face. Returns (corpus, queries, test qrels)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo = f"BeIR/{name}"
    corpus_path = hf_hub_download(repo, "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
    queries_path = hf_hub_download(repo, "queries/queries-00000-of-00001.parquet", repo_type="dataset")
    qrels_path = hf_hub_download(f"{repo}-qrels", "test.tsv", repo_type="dataset")

    corpus = {str(r["_id"]): {"title": r.get("title") or "", "text": r.get("text") or ""}
              for r in pq.read_table(corpus_path).to_pylist()}
    queries = {str(r["_id"]): r["text"] for r in pq.read_table(queries_path).to_pylist()}
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        next(f)  # header: query-id corpus-id score
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return corpus, queries, qrels

def _ensure_org(store, name: str) -> Tuple[int, int]:
    slug = f"beir-{name}"
    for org in store.list_organizations():
        if org["slug"] == slug:
            return org["id"], store.get_org_root_unit(org["id"])
    ids = store.create_organization(f"BEIR {name}", slug, f"{slug}-owner", hash_key(secrets.token_hex(16)))
    return ids["org_id"], ids["root_unit_id"]

def ingest_corpus(store, corpus: Dict[str, Dict[str, str]], org_id: int, unit_id: int) -> float:
    """Chunks, embeds and stores every document once. Re-runs resume where the last stopped."""
    with store.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT source FROM documents WHERE org_id = %s;", (org_id,))
        done = {r["source"] for r in cur.fetchall()}
    items = [(doc_id, doc) for doc_id, doc in corpus.items() if doc_id not in done]
    if not items:
        return 0.0

    chunker = Chunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embedder = get_embedder()
    t0 = time.time()
    for start in range(0, len(items), INGEST_BATCH):
        batch = []
        for doc_id, doc in items[start:start + INGEST_BATCH]:
            meta = {"source": doc_id, "title": doc["title"] or doc_id, "file_path": f"{doc_id}.txt",
                    "total_pages": 1, "org_id": org_id, "unit_id": unit_id}
            text = f"{doc['title']}\n\n{doc['text']}".strip()
            chunks = chunker.split_document({"metadata": meta, "content": text,
                                             "pages": [{"page_num": 1, "text": text}]})
            batch.append((meta, chunker.enrich_context(chunks, text)))
        vectors = embedder.embed_documents([c.full_text for _, chunks in batch for c in chunks])
        pos = 0
        for meta, chunks in batch:
            doc_pk = store.insert_document(meta)
            store.insert_chunks(doc_pk, [c.to_dict() for c in chunks], vectors[pos:pos + len(chunks)],
                                org_id=org_id)
            pos += len(chunks)
        print(f"    ingested {len(done) + min(start + INGEST_BATCH, len(items))}/{len(corpus)} documents", flush=True)
    store.bump_corpus_epoch()
    return time.time() - t0

def _doc_ranking(rows: List[Dict[str, Any]]) -> List[str]:
    """Chunk hits -> document ids, best rank first, each document once."""
    seen, ranked = set(), []
    for r in rows:
        if r["source_doc"] not in seen:
            seen.add(r["source_doc"])
            ranked.append(r["source_doc"])
    return ranked

def ndcg_at_k(ranked: List[str], rel: Dict[str, int], k: int = K) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def recall_at_k(ranked: List[str], rel: Dict[str, int], k: int = K) -> float:
    return len(set(ranked[:k]) & set(rel)) / len(rel)

def mrr_at_k(ranked: List[str], rel: Dict[str, int], k: int = K) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0

def _pct(values: List[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

def run_benchmark(dataset: str = "scifact", limit: int = 0, rerank: bool = True) -> Dict[str, Any]:
    print(f"[*] Loading BEIR/{dataset} from Hugging Face...")
    corpus, queries, qrels = load_beir(dataset)
    test_ids = [q for q in qrels if q in queries]
    if limit:
        test_ids = test_ids[:limit]
    print(f"[+] {len(corpus)} documents, {len(test_ids)} test queries.")

    store = get_vector_store()
    embedder = get_embedder()
    store.initialize_db(dimension=embedder.dimension)
    org_id, unit_id = _ensure_org(store, dataset)
    ingest_s = ingest_corpus(store, corpus, org_id, unit_id)
    if ingest_s:
        print(f"[+] Ingested in {ingest_s:.0f}s.")

    retriever = Retriever(store, embedder)
    reranker = get_reranker() if rerank else None
    depth = settings.top_k_retrieval

    def dense(q: str) -> List[Dict[str, Any]]:
        return store.dense_search(embedder.embed_query(q), depth, org_id)

    def lexical(q: str) -> List[Dict[str, Any]]:
        return store.lexical_search(q, depth, org_id)

    def hybrid(q: str) -> List[Dict[str, Any]]:
        return retriever.retrieve(q, org_id=org_id)[0]

    def hybrid_rerank(q: str) -> List[Dict[str, Any]]:
        return reranker.rerank(q, hybrid(q), top_k=K)

    methods: List[Tuple[str, Callable[[str], List[Dict[str, Any]]]]] = [
        ("Postgres full-text search", lexical),
        (f"Dense ({settings.local_embedding_model})", dense),
        (f"Hybrid (dense + FTS, {settings.fusion_method})", hybrid),
    ]
    if reranker:
        methods.append((f"Hybrid + rerank ({settings.local_reranker_model})", hybrid_rerank))

    results = []
    for name, fn in methods:
        scores = {"ndcg": [], "recall": [], "mrr": [], "ms": []}
        for qid in test_ids:
            t0 = time.perf_counter()
            ranked = _doc_ranking(fn(queries[qid]))
            scores["ms"].append((time.perf_counter() - t0) * 1000)
            scores["ndcg"].append(ndcg_at_k(ranked, qrels[qid]))
            scores["recall"].append(recall_at_k(ranked, qrels[qid]))
            scores["mrr"].append(mrr_at_k(ranked, qrels[qid]))
        row = {"method": name,
               "ndcg@10": statistics.mean(scores["ndcg"]),
               "recall@10": statistics.mean(scores["recall"]),
               "mrr@10": statistics.mean(scores["mrr"]),
               "p50_ms": _pct(scores["ms"], 0.5),
               "p95_ms": _pct(scores["ms"], 0.95)}
        results.append(row)
        print(f"    {name}: nDCG@10 {row['ndcg@10']:.3f}, p95 {row['p95_ms']:.0f} ms", flush=True)

    report = {"dataset": f"BEIR/{dataset}", "documents": len(corpus), "queries": len(test_ids),
              "cpu_threads": os.cpu_count(), "top_k_retrieval": depth, "results": results}
    print_table(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"beir_{dataset}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved {out}")
    return report

def print_table(report: Dict[str, Any]):
    print(f"\n{report['dataset']}: {report['documents']} docs, {report['queries']} queries, "
          f"{report['cpu_threads']} CPU threads\n")
    print("| Method | nDCG@10 | Recall@10 | MRR@10 | p50 | p95 |")
    print("|---|---|---|---|---|---|")
    for r in report["results"]:
        print(f"| {r['method']} | {r['ndcg@10']:.3f} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} "
              f"| {r['p50_ms']:.0f} ms | {r['p95_ms']:.0f} ms |")
    print()
