from dotenv import load_dotenv
load_dotenv()

def test_imports():
    print("[*] Testing imports...")
    try:
        from pipeline.config import settings
        from pipeline.parser import DoclingParser
        from pipeline.chunker import Chunker
        from pipeline.embedder import get_embedder
        from pipeline.vector_store import get_vector_store
        from pipeline.retriever import Retriever
        from pipeline.generator import Generator
        from pipeline.evaluator import Evaluator
        from pipeline.ingestion import ingest_file
        print("[+] All imports successful!")
        return True
    except Exception as e:
        print(f"[!] Import failed: {e}")
        return False

def test_config():
    print("\n[*] Testing Configuration...")
    from pipeline.config import settings
    print(f"- Generation Provider: {settings.generation_provider} (only cloud step)")
    print(f"- Embedding Provider: {settings.embedding_provider}")
    print(f"- Reranker Provider: {settings.reranker_provider}")
    print(f"- Evaluator Provider: {settings.evaluator_provider}")
    print(f"- Chunking Strategy: {settings.chunking_strategy}")
    print(f"- Database Host: {settings.db_host}:{settings.db_port}")
    if not settings.gemini_api_key:
        print("[!] Warning: GEMINI_API_KEY not set — answer generation will fail (retrieval still works).")
    else:
        print("[+] GEMINI_API_KEY is set.")
    return True

def test_db_connection():
    print("\n[*] Testing Postgres Database Connection...")
    from pipeline.vector_store import get_vector_store
    from pipeline.embedder import get_embedder

    store = get_vector_store()
    try:
        store.connect()
        print("[+] Successfully connected to Postgres!")

        embedder = get_embedder()
        dimension = embedder.dimension
        print(f"[*] Applying migrations with embedder dimension={dimension}...")
        store.initialize_db(dimension=dimension)
        print("[+] Migrations applied.")

        print("[*] Verifying pgvector insert (COPY) and search...")
        org = store.ensure_default_org()
        doc_id = store.insert_document({
            "source": "test_doc.txt",
            "title": "Test Document",
            "total_pages": 1,
            "file_path": "test_doc.txt",
            "org_id": org["org_id"],
            "unit_id": org["root_unit_id"],
        })

        dummy_vector = [0.1] * dimension
        store.insert_chunks(doc_id, [
            {"chunk_index": 0, "page_num": 1, "text": "This is test database content.", "context": "testing"}
        ], [dummy_vector], org_id=org["org_id"])

        results = store.dense_search(dummy_vector, top_k=1, org_id=org["org_id"])
        if results:
            print("[+] pgvector search test successful!")
        else:
            print("[!] pgvector search test returned no results.")

        store.delete_document(doc_id)
        print("[+] Cleanup done.")
        return True
    except Exception as e:
        print(f"[!] Database check failed: {e}")
        print("    Please ensure 'docker compose up -d postgres' is running.")
        return False

def main():
    print("=" * 60)
    print("RAG PIPELINE INTEGRATION TEST")
    print("=" * 60)

    if not test_imports():
        return
    test_config()
    test_db_connection()
    print("=" * 60)

if __name__ == "__main__":
    main()
