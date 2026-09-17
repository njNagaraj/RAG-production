from dotenv import load_dotenv
load_dotenv()

from pipeline.config import settings
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store

def test_semantic_cache():
    print("=" * 60)
    print("TESTING SEMANTIC CACHING SYSTEM (local embeddings)")
    print("=" * 60)

    embedder = get_embedder()
    store = get_vector_store()

    try:
        store.connect()
        print("[+] Connected to database.")
    except Exception as e:
        print(f"[!] Database connection failed: {e}")
        return

    store.initialize_db(dimension=embedder.dimension)

    with store.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE semantic_cache;")
        conn.commit()
    print("[*] Cleared semantic_cache table.")

    query_1 = "What is the annual lodging budget for L3 employees?"
    answer_1 = "The annual lodging budget allowance for L3 employees is $2,000 as defined in Section 4.2."
    citations_1 = [{"id": 1, "source": "travel_policy.pdf", "page": 4,
                    "content": "Grade L3 is restricted to a maximum lodging budget of $2,000 annually."}]

    print(f"\n[*] Embedding primary query locally: '{query_1}'")
    emb_1 = embedder.embed_query(query_1)
    store.add_to_semantic_cache(query_1, emb_1, answer_1, citations_1, org_id=1, access_key="test_scope")
    print("[+] Cached query and response.")

    # Exact hit (same access scope)
    hit_1 = store.check_semantic_cache(emb_1, similarity_threshold=0.98, access_key="test_scope")
    print(f"\n[*] Exact hit: {'PASS' if hit_1 else 'FAIL'}"
          + (f" (similarity {hit_1['similarity']:.4f})" if hit_1 else ""))

    # Semantic variant
    query_2 = "How much can a grade L3 staff spend on hotels each year?"
    emb_2 = embedder.embed_query(query_2)
    hit_2 = store.check_semantic_cache(emb_2, similarity_threshold=0.85, access_key="test_scope")
    if hit_2:
        print(f"[*] Semantic variant hit at 0.85: similarity {hit_2['similarity']:.4f}")
    else:
        print("[*] Semantic variant below 0.85 threshold (acceptable for bge-small).")

    # RBAC isolation: a different access scope must never see this entry
    hit_other = store.check_semantic_cache(emb_1, similarity_threshold=0.98, access_key="other_scope")
    print(f"[*] Cross-scope cache isolation: {'PASS' if hit_other is None else 'FAIL'}")

    # Epoch invalidation: bumping the corpus epoch must invalidate old entries
    print("\n[*] Testing corpus-epoch invalidation...")
    store.bump_corpus_epoch()
    hit_3 = store.check_semantic_cache(emb_1, similarity_threshold=0.98, access_key="test_scope")
    print(f"[*] Post-ingest cache serve blocked: {'PASS' if hit_3 is None else 'FAIL'}")

    store.cleanup_semantic_cache()
    print("[+] Cache cleanup ran.")
    print("=" * 60)

if __name__ == "__main__":
    test_semantic_cache()
