import argparse
import sys
import os
import glob

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import settings
from pipeline.embedder import get_embedder
from pipeline.vector_store import get_vector_store
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from pipeline.evaluator import Evaluator
from pipeline.reranker import get_reranker
from pipeline.ingestion import ingest_file

SUPPORTED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx", "*.py", "*.js", "*.ts", "*.java", "*.go")

def main():
    parser = argparse.ArgumentParser(description="Modular RAG Pipeline CLI (local-first)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file or directory of documents")
    ingest_parser.add_argument("--path", required=True, help="Path to file or directory to ingest")
    ingest_parser.add_argument("--enrich", action="store_true", help="Describe charts/images with Gemini vision at ingest time")
    ingest_parser.add_argument("--groups", default="", help="Comma-separated access groups (RBAC); empty = visible to everyone")

    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("text", help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        handle_ingest(args.path, args.enrich, groups)
    elif args.command == "query":
        handle_query(args.text)
    else:
        parser.print_help()

def handle_ingest(path: str, enrich: bool, groups: list = None):
    print(f"[*] Starting ingestion for path: {path}")

    files = []
    if os.path.isdir(path):
        for ext in SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(path, ext)))
    elif os.path.isfile(path):
        files.append(path)
    else:
        print(f"[!] Path not found: {path}")
        sys.exit(1)

    if not files:
        print(f"[!] No supported documents found ({', '.join(SUPPORTED_EXTS)}).")
        return

    print(f"[*] Found {len(files)} files to process.")

    embedder = get_embedder()
    vector_store = get_vector_store()
    print("[*] Applying database migrations...")
    vector_store.initialize_db(dimension=embedder.dimension)

    failed = 0
    for f in files:
        try:
            result = ingest_file(f, enrich=enrich, allowed_groups=groups or [])
            print(f"[+] Ingested {result['source']}: {result['chunks_count']} chunks "
                  f"(strategy: {result['chunking_strategy']})")
        except Exception as e:
            failed += 1
            print(f"[!] Failed to ingest {os.path.basename(f)}: {e}")

    print(f"\n[+] Ingestion complete: {len(files) - failed} succeeded, {failed} failed.")

def handle_query(query_text: str):
    print(f"[*] Query: {query_text}\n")

    embedder = get_embedder()
    vector_store = get_vector_store()

    try:
        vector_store.connect()
    except ConnectionError as ce:
        print(f"[!] Database Connection Error: {ce}")
        sys.exit(1)

    retriever = Retriever(vector_store, embedder)
    generator = Generator()
    evaluator = Evaluator()

    print("[*] Retrieving (local adaptive transform + dense + FTS hybrid)...")
    chunks, debug_info = retriever.retrieve(query_text)
    print(f"[*] Query type: {debug_info['query_type']}. Found {debug_info['fused_count']} candidates.")

    print("[*] Reranking with local cross-encoder...")
    reranker = get_reranker()
    reranked_chunks = reranker.rerank(query_text, chunks, top_k=settings.top_k_reranked)
    print(f"[*] Kept top {len(reranked_chunks)} reranked chunks.")

    print("[*] Generating grounded response (Gemini — the only cloud call)...")
    answer, citations = generator.generate(query_text, reranked_chunks)

    print("[*] Evaluating faithfulness (local cross-encoder)...")
    eval_res = evaluator.evaluate_faithfulness(query_text, answer, reranked_chunks)

    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(answer)
    print("=" * 50)

    if citations:
        print("\nCITATIONS:")
        for cit in citations:
            print(f"[{cit['id']}] {cit['source']} (Page {cit['page']})")

    print("\nMETRICS:")
    print(f"- Faithfulness: {eval_res.score:.2f} ({'Grounded' if eval_res.faithful else 'Hallucinated'})")
    print(f"- Reason: {eval_res.reason}")

if __name__ == "__main__":
    main()
