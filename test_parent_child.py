from pipeline.chunker import Chunker, ParentChildStrategy, Chunk, count_tokens
from pipeline.embedder import get_embedder

def test_parent_child_splitting():
    print("[*] Testing ParentChildStrategy splitting logic...")
    # Create sample document text with multiple paragraphs
    paragraphs = [
        f"Paragraph {i}: This is an enterprise document section discussing operational procedure number {i}. "
        f"Employees must review all compliance requirements annually. " * 3
        for i in range(1, 12)
    ]
    sample_text = "\n\n".join(paragraphs)
    doc_data = {
        "metadata": {
            "source": "operations_manual.pdf",
            "title": "Operations Manual",
            "file_path": "operations_manual.pdf"
        },
        "pages": [{"page_num": 1, "text": sample_text}]
    }

    strategy = ParentChildStrategy(parent_size=300, child_size=80, child_overlap=20)
    chunks = strategy.split(doc_data)

    assert len(chunks) > 0, "Expected chunks to be generated"
    print(f"[+] Generated {len(chunks)} child chunks from parent blocks.")

    for i, c in enumerate(chunks):
        assert c.parent_content is not None, f"Chunk {i} missing parent_content"
        assert c.parent_chunk_index is not None, f"Chunk {i} missing parent_chunk_index"
        child_tokens = count_tokens(c.text)
        parent_tokens = count_tokens(c.parent_content)
        assert parent_tokens >= child_tokens, "Parent chunk must be larger than or equal to child chunk"
        assert c.text in c.parent_content or any(w in c.parent_content for w in c.text.split()[:5]), "Child content must belong to parent"

    print(f"[+] Verified all {len(chunks)} child chunks contain valid parent_content and parent_chunk_index!")

def test_parent_child_chunker_dispatch():
    print("\n[*] Testing Chunker dispatch for parent_child...")
    chunker = Chunker()
    doc_data = {
        "metadata": {"source": "test.txt", "file_path": "test.txt"},
        "pages": [{"page_num": 1, "text": "This is test text for chunking."}]
    }
    strategy = chunker._make("parent_child")
    assert isinstance(strategy, ParentChildStrategy)
    print("[+] Chunker correctly instantiates ParentChildStrategy.")

def test_child_chunk_embedding():
    print("\n[*] Testing child chunk embedding...")
    embedder = get_embedder()
    test_child_text = "Child chunk discussing compliance requirements and employee policy."
    emb = embedder.embed_query(test_child_text)
    assert len(emb) == embedder.dimension
    print(f"[+] Child chunk embedded successfully: vector dim = {len(emb)}")

if __name__ == "__main__":
    print("=" * 60)
    print("PARENT-CHILD (HIERARCHICAL) CHUNKING TEST")
    print("=" * 60)
    test_parent_child_splitting()
    test_parent_child_chunker_dispatch()
    test_child_chunk_embedding()
    print("\n[+] ALL PARENT-CHILD CHUNKING TESTS PASSED!")
    print("=" * 60)
