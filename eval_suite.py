import os
import json
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import settings
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from pipeline.evaluator import Evaluator
from pipeline.reranker import get_reranker

def _target_rank(chunks, target: str):
    """1-based rank of the first chunk from the target source, or None."""
    for i, c in enumerate(chunks):
        if target.lower() in (c.get("source_doc") or "").lower():
            return i + 1
    return None

def run_evaluation_suite(skip_generation: bool = False, org_slug: str = "default"):
    print("=" * 70)
    print("      PRODUCTION RAG: AUTOMATED CONTINUOUS EVALUATION SUITE")
    print("=" * 70)

    dataset_path = "eval_golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"[!] Error: Golden dataset not found at {dataset_path}")
        return False

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    print(f"[+] Loaded {len(golden_cases)} golden test cases.")

    embedder = get_embedder()
    store = get_vector_store()
    try:
        store.connect()
    except Exception as e:
        print(f"[!] Database connection error: {e}")
        print("[!] Retrieval metrics require the database. Aborting.")
        return False

    # Retrieval is tenant-scoped: without an org_id the SQL filter matches nothing.
    # Org-wide scope (visible_node_ids=None) = what the org owner sees.
    if org_slug == "default":
        org_id = store.ensure_default_org()["org_id"]
    else:
        org_id = next((o["id"] for o in store.list_organizations() if o["slug"] == org_slug), None)
        if org_id is None:
            print(f"[!] Organization '{org_slug}' not found.")
            return False
    print(f"[+] Evaluating against org '{org_slug}' (id={org_id}).")

    retriever = Retriever(store, embedder)
    reranker = get_reranker()
    evaluator = Evaluator()
    generator = None
    if not skip_generation:
        try:
            generator = Generator()
        except Exception as e:
            print(f"[!] Generator unavailable ({e}). Running retrieval-only evaluation.")

    total = len(golden_cases)
    recall_at_k_hits = 0        # target source within top_k_reranked after rerank
    recall_fused_hits = 0       # target source anywhere in fused candidates
    reciprocal_ranks = []       # MRR over reranked list
    passed_faithfulness = 0
    total_faithfulness = 0.0
    generated = 0
    latencies = []

    print("\nStarting runs...\n")

    for case in golden_cases:
        cid = case["id"]
        q = case["question"]
        target = case["target_source"]
        keywords = case.get("expected_keywords", [])

        print(f"[{cid}/{total}] Question: '{q}'")
        print(f"      Expected Target Source: {target}")

        t0 = time.time()
        try:
            chunks, debug = retriever.retrieve(q, history=[], org_id=org_id)
        except Exception as e:
            print(f"      [!] Retrieval failed: {e}")
            reciprocal_ranks.append(0.0)
            print("-" * 50)
            continue

        reranked = reranker.rerank(q, chunks, top_k=settings.top_k_reranked)
        latencies.append(time.time() - t0)

        # Retrieval metrics
        fused_rank = _target_rank(chunks, target)
        reranked_rank = _target_rank(reranked, target)
        if fused_rank:
            recall_fused_hits += 1
        if reranked_rank:
            recall_at_k_hits += 1
            reciprocal_ranks.append(1.0 / reranked_rank)
            print(f"      [PASS] Recall@{settings.top_k_reranked} (rank {reranked_rank}, query_type={debug['query_type']})")
        else:
            reciprocal_ranks.append(0.0)
            print(f"      [FAIL] Recall@{settings.top_k_reranked} "
                  f"(fused rank: {fused_rank or 'miss'}, query_type={debug['query_type']})")

        # Generation + faithfulness (requires the single cloud call)
        if generator:
            try:
                answer, citations = generator.generate(q, reranked)
                generated += 1
                print(f"      Answer: \"{answer[:80]}...\"")

                keyword_matches = [kw for kw in keywords if kw.lower() in answer.lower()]
                print(f"      Keyword Matches: {len(keyword_matches)}/{len(keywords)}")

                eval_res = evaluator.evaluate_faithfulness(q, answer, reranked)
                total_faithfulness += eval_res.score
                if eval_res.faithful:
                    passed_faithfulness += 1
                    print(f"      [PASS] Faithfulness (Score: {eval_res.score * 100:.0f}%)")
                else:
                    print(f"      [FAIL] Faithfulness (Score: {eval_res.score * 100:.0f}%) - {eval_res.reason}")
            except Exception as e:
                print(f"      [!] Generation/Evaluation failed: {e}")

        print("-" * 50)

    # Summary
    recall_at_k = recall_at_k_hits / total * 100 if total else 0.0
    recall_fused = recall_fused_hits / total * 100 if total else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_faith = total_faithfulness / generated if generated else 0.0
    faith_rate = passed_faithfulness / generated * 100 if generated else 0.0

    print("\n" + "=" * 70)
    print("                       EVALUATION RESULTS CARD")
    print("=" * 70)
    print(f"Total Test Cases:            {total}")
    print(f"Recall@{settings.top_k_reranked} (post-rerank):     {recall_at_k:.1f}% ({recall_at_k_hits}/{total})")
    print(f"Recall (fused candidates):   {recall_fused:.1f}% ({recall_fused_hits}/{total})")
    print(f"MRR (post-rerank):           {mrr:.3f}")
    print(f"Avg Retrieval+Rerank Latency:{avg_latency * 1000:.0f} ms")
    if generated:
        print(f"Grounded Faithfulness Rate:  {faith_rate:.1f}% ({passed_faithfulness}/{generated})")
        print(f"Average Faithfulness Score:  {avg_faith * 100:.1f}%")
    print("=" * 70)

    success = recall_at_k >= 80.0 and mrr >= 0.5 and (not generated or avg_faith >= 0.8)
    if success:
        print("[+] SUCCESS: Pipeline passed all evaluation threshold metrics.")
    else:
        print("[!] FAILURE: Pipeline metrics are below target thresholds.")
    return success

if __name__ == "__main__":
    ok = run_evaluation_suite(skip_generation="--retrieval-only" in sys.argv)
    sys.exit(0 if ok else 1)
