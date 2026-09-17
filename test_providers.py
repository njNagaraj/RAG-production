from pipeline.config import settings
from pipeline.generator import (
    get_generator, MockGenerator, GeminiGenerator, OpenAIGenerator,
    format_context_and_citations, filter_active_citations
)
from pipeline.embedder import get_embedder, FastEmbedEmbedder

def test_mock_generator():
    gen = MockGenerator()
    chunks = [
        {"source_doc": "handbook.pdf", "page_num": 1, "content": "Employees receive 25 days PTO annually."}
    ]
    answer, citations = gen.generate("How many PTO days?", chunks)
    assert len(citations) > 0
    assert "handbook.pdf" in answer
    assert "25 days" in answer
    print("[+] test_mock_generator PASSED")

def test_context_formatting():
    chunks = [
        {"source_doc": "doc1.txt", "page_num": 2, "content": "Section A content", "context": "Document Header"}
    ]
    context_str, citations = format_context_and_citations(chunks)
    assert "CONTEXT SOURCE [1]" in context_str
    assert citations[0]["id"] == 1
    print("[+] test_context_formatting PASSED")

def test_citation_filtering():
    citations = [
        {"id": 1, "source": "a.txt"},
        {"id": 2, "source": "b.txt"}
    ]
    # Answer citing only [1]
    filtered = filter_active_citations("According to policy [1], this is true.", citations)
    assert len(filtered) == 1
    assert filtered[0]["id"] == 1
    print("[+] test_citation_filtering PASSED")

def test_embedder_default():
    embedder = get_embedder()
    assert embedder.dimension == 384
    emb = embedder.embed_query("testing query embedding")
    assert len(emb) == 384
    print("[+] test_embedder_default PASSED")

if __name__ == "__main__":
    print("Running Provider Tests...")
    test_mock_generator()
    test_context_formatting()
    test_citation_filtering()
    test_embedder_default()
    print("All Provider Tests PASSED!")
