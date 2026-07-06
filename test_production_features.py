from dotenv import load_dotenv
load_dotenv()

from pipeline.router import QueryRouter
from pipeline.retriever import classify_query, transform_query, extract_metadata_filter

def verify_production_features():
    print("=" * 60)
    print("  VERIFYING LOCAL ROUTER, QUERY TRANSFORM & METADATA FILTER")
    print("=" * 60)

    # 1. Router (embedding-similarity, no LLM)
    print("[*] Initializing local QueryRouter...")
    router = QueryRouter()

    cases = [
        ("Hey there! How is it going?", "chitchat"),
        ("Who built you?", "chitchat"),
        ("Thanks!", "chitchat"),
        ("What is the policy regarding casual leaves and paid time off?", "rag"),
        ("lodging limits for L3 employees", "rag"),
    ]
    for q, expected in cases:
        dec = router.route(q)
        status = "PASS" if dec.route == expected else "FAIL"
        print(f"    [{status}] '{q}' -> {dec.route} (expected {expected}, conf {dec.confidence:.2f})")

    # 2. Adaptive query transformation (rule-based, local)
    print("\n[*] Query-type classification and transformation:")
    tests = [
        ("Show details for invoice INV-2024-001", "exact_lookup"),
        ('Find the section that mentions "gross negligence"', "exact_lookup"),
        ("What is the leave policy? And how do I apply for it?", "multi_part"),
        ("Explain the travel reimbursement process", "conceptual"),
    ]
    for q, expected in tests:
        t = transform_query(q)
        status = "PASS" if t["query_type"] == expected else "FAIL"
        print(f"    [{status}] '{q[:55]}' -> {t['query_type']} "
              f"(queries={len(t['search_queries'])}, lex_w={t['lexical_weight']})")

    # Follow-up detection needs history
    history = [{"role": "user", "content": "What is the maternity leave policy?"},
               {"role": "assistant", "content": "It provides 12 weeks paid leave."}]
    t = transform_query("What about for adoptive parents?", history)
    status = "PASS" if t["query_type"] == "follow_up" else "FAIL"
    print(f"    [{status}] follow-up rewrite -> '{t['standalone_query'][:80]}'")

    # 3. Metadata filter extraction (regex, local)
    print("\n[*] Metadata filter extraction:")
    f1 = extract_metadata_filter("Read page 14 of hr_policy.pdf and explain travel costs.",
                                 known_sources=["hr_policy.pdf", "benefits_guide.docx"])
    print(f"    {'PASS' if f1.get('page_num') == 14 and f1.get('source_doc') == 'hr_policy.pdf' else 'FAIL'} "
          f"-> {f1}")

    f2 = extract_metadata_filter("Summarize benefits_guide.docx lodging guidelines.",
                                 known_sources=["hr_policy.pdf", "benefits_guide.docx"])
    print(f"    {'PASS' if f2.get('source_doc') == 'benefits_guide.docx' else 'FAIL'} -> {f2}")

    print("=" * 60)

if __name__ == "__main__":
    verify_production_features()
