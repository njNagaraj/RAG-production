"""
Local-first behaviour that needs no database or downloaded models:
parent dedupe, local vision via Ollama, provider selection, locality report.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pipeline.config import settings
from pipeline.models import locality_report
from pipeline.parser import OllamaVisionDescriber, get_vision_client
from pipeline.retriever import collapse_shared_parents

def test_collapse_shared_parents_keeps_best_sibling():
    fused = [
        {"id": 1, "source_doc": "a.md", "content": "PARENT-1", "rrf_score": 0.9},
        {"id": 2, "source_doc": "a.md", "content": "PARENT-1", "rrf_score": 0.8},
        {"id": 3, "source_doc": "b.md", "content": "PARENT-1", "rrf_score": 0.7},
        {"id": 4, "source_doc": "a.md", "content": "PARENT-2", "rrf_score": 0.6},
    ]
    assert [r["id"] for r in collapse_shared_parents(fused)] == [1, 3, 4]

@pytest.fixture
def fake_ollama():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # /api/tags probe
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"models": []}')

        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"response": "  Revenue grew from $2M to $5M.  "}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()

def test_ollama_vision_describer_sends_image_locally(fake_ollama):
    url, seen = fake_ollama
    desc = OllamaVisionDescriber(base_url=url, model_name="llava").describe(b"\x89PNG", "image/png")
    assert desc == "Revenue grew from $2M to $5M."
    assert seen["path"] == "/api/generate"
    assert seen["body"]["model"] == "llava"
    assert seen["body"]["images"] == ["iVBORw=="]

def test_ollama_vision_describer_returns_none_when_down():
    assert OllamaVisionDescriber(base_url="http://127.0.0.1:9", model_name="x").describe(b"x", "image/png") is None

def test_vision_client_selection(fake_ollama, monkeypatch):
    url, _ = fake_ollama
    monkeypatch.setattr(settings, "ollama_base_url", url)
    monkeypatch.setattr(settings, "describe_images", False)
    monkeypatch.setattr(settings, "vision_provider", "ollama")
    assert get_vision_client() is None
    assert isinstance(get_vision_client(force=True), OllamaVisionDescriber)
    monkeypatch.setattr(settings, "vision_provider", "none")
    assert get_vision_client(force=True) is None
    monkeypatch.setattr(settings, "vision_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    assert get_vision_client(force=True) is None

def test_locality_report(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "reranker_provider", "local")
    monkeypatch.setattr(settings, "describe_images", False)
    assert all(local for _, _, local in locality_report("OllamaGenerator"))

    rows = dict((stage, local) for stage, _, local in locality_report("GeminiGenerator"))
    assert rows["Generation"] is False

    monkeypatch.setattr(settings, "openai_base_url", "http://localhost:8001/v1")
    rows = dict((stage, local) for stage, _, local in locality_report("OpenAIGenerator"))
    assert rows["Generation"] is True
