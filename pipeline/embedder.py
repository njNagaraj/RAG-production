import abc
import threading
from typing import List, Optional
from pipeline.config import settings

class BaseEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        pass

class FastEmbedEmbedder(BaseEmbedder):
    """
    Local ONNX embedder (CPU-friendly, no torch). Default: BAAI/bge-small-en-v1.5 (384d).
    Model loads once per process; instances are shared via get_embedder().
    """
    # bge family expects a query instruction prefix for retrieval queries
    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: Optional[str] = None):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("fastembed package not installed. Run: pip install fastembed")
        self.model_name = model_name or settings.local_embedding_model
        self.model = TextEmbedding(model_name=self.model_name)
        dummy = next(iter(self.model.embed(["hello"])))
        self._dimension = len(dummy)

    def embed_query(self, text: str) -> List[float]:
        prefixed = self._QUERY_PREFIX + text if "bge" in self.model_name.lower() else text
        emb = next(iter(self.model.embed([prefixed])))
        return emb.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [emb.tolist() for emb in self.model.embed(texts, batch_size=64)]

    @property
    def dimension(self) -> int:
        return self._dimension

class GeminiEmbedder(BaseEmbedder):
    """Optional cloud embedder. Not used by default; local-first is the requirement."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiEmbedder")
        from google import genai
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self._model = "gemini-embedding-2"
        self._dimension = 768

    def embed_query(self, text: str) -> List[float]:
        from google.genai import types
        response = self.client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self._dimension)
        )
        embeddings = response.embeddings
        if embeddings and len(embeddings) > 0:
            return embeddings[0].values
        raise ValueError("No embeddings returned by Gemini API")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from google.genai import types
        response = self.client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self._dimension)
        )
        return [emb.values for emb in response.embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension

_embedder_instance: Optional[BaseEmbedder] = None
_embedder_lock = threading.Lock()

def get_embedder() -> BaseEmbedder:
    """Process-wide singleton: the ONNX model must load once, not per request."""
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                provider = settings.embedding_provider.lower()
                if provider in ("local", "fastembed", "sentence_transformers"):
                    _embedder_instance = FastEmbedEmbedder()
                elif provider == "gemini":
                    _embedder_instance = GeminiEmbedder()
                else:
                    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return _embedder_instance
