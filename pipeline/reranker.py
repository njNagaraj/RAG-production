import abc
import threading
from typing import List, Dict, Any, Optional
from pipeline.config import settings

class BaseReranker(abc.ABC):
    @abc.abstractmethod
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        pass

class PassThroughReranker(BaseReranker):
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return candidates[:top_k]

class LocalCrossEncoderReranker(BaseReranker):
    """
    Local ONNX cross-encoder (CPU). Scores every fused candidate against the
    query — no arbitrary top-20 truncation before reranking.
    """

    def __init__(self, model_name: Optional[str] = None):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            raise ImportError("fastembed package not installed. Run: pip install fastembed")
        self.model = TextCrossEncoder(model_name=model_name or settings.local_reranker_model)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        documents = [
            f"[Context: {c.get('context', '')}] {c['content']}" if c.get("context") else c["content"]
            for c in candidates
        ]
        scores = list(self.model.rerank(query, documents))
        ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        reranked = []
        for idx in ranked[:top_k]:
            item = candidates[idx].copy()
            item["rerank_score"] = float(scores[idx])
            reranked.append(item)
        return reranked

class CohereReranker(BaseReranker):
    def __init__(self):
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required for CohereReranker")
        import cohere
        self.client = cohere.Client(api_key=settings.cohere_api_key)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        documents = [f"[Context: {c.get('context', '')}] {c['content']}" for c in candidates]
        try:
            response = self.client.rerank(
                model='rerank-v4.0',
                query=query,
                documents=documents,
                top_n=top_k
            )
            reranked = []
            for result in response.results:
                item = candidates[result.index].copy()
                item["rerank_score"] = result.relevance_score
                reranked.append(item)
            return reranked
        except Exception as e:
            print(f"Cohere Reranker failed: {e}. Falling back to pass-through.")
            return candidates[:top_k]

_reranker_instance: Optional[BaseReranker] = None
_reranker_lock = threading.Lock()

def get_reranker() -> BaseReranker:
    """Process-wide singleton: cross-encoder model loads once."""
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                provider = settings.reranker_provider.lower()
                if provider in ("local", "cross_encoder"):
                    _reranker_instance = LocalCrossEncoderReranker()
                elif provider == "cohere":
                    _reranker_instance = CohereReranker()
                else:
                    _reranker_instance = PassThroughReranker()
    return _reranker_instance
