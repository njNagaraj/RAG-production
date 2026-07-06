import re
import random
import threading
from typing import List, Optional
from pydantic import BaseModel, Field
from pipeline.embedder import get_embedder

class RouterDecision(BaseModel):
    route: str = Field(description="'chitchat' (greetings/small talk) or 'rag' (factual document searches).")
    response: str = Field(default="", description="Canned reply when route is 'chitchat'; empty for 'rag'.")
    confidence: float = Field(default=1.0, description="Router confidence in [0,1].")

# Fast-path patterns: pure greetings/pleasantries need no model at all
_GREETING_RE = re.compile(
    r"^\s*(hi+|hii+|hello+|hey+|yo|good\s*(morning|afternoon|evening)|"
    r"thanks?|thank\s*you|thx|ty|ok(ay)?|bye|goodbye|see\s*ya|how\s*are\s*you\??)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_CHITCHAT_SEEDS = [
    "hello there",
    "hi, how are you doing today?",
    "good morning!",
    "thanks a lot, that was helpful",
    "who are you?",
    "what can you do?",
    "who built you?",
    "tell me a joke",
    "bye, see you later",
    "nice talking to you",
]

_RAG_SEEDS = [
    "what is the casual leave policy?",
    "lodging limits for L3 employees",
    "when does the holiday calendar start?",
    "explain the travel reimbursement process",
    "what does the employee handbook say about probation?",
    "summarize the security guidelines document",
    "what is the notice period for resignation?",
    "invoice number INV-2024-001 details",
    "revenue figures in the annual report",
    "which page covers the dress code?",
]

_CHITCHAT_REPLIES = [
    "Hello! I'm your document assistant. Ask me anything about the uploaded policies and documents.",
    "Hi there! I can help you find answers from your company documents — what would you like to know?",
    "You're welcome! Let me know if you have any other questions about the documents.",
]

_IDENTITY_REPLY = (
    "I'm an AI assistant that answers questions grounded in your uploaded documents, "
    "with citations to the exact source and page."
)

class QueryRouter:
    """
    Local, deterministic query router — no LLM.
    1. Regex fast-path catches pure greetings/pleasantries.
    2. Otherwise: cosine similarity of the query embedding against chitchat
       vs rag seed-example centroids. Ambiguous queries default to 'rag'
       (safe: worst case we search documents for a greeting).
    """

    _seed_lock = threading.Lock()
    _chitchat_centroid: Optional[List[float]] = None
    _rag_centroid: Optional[List[float]] = None

    def __init__(self):
        self.embedder = get_embedder()
        self._ensure_centroids()

    @classmethod
    def _mean(cls, vectors: List[List[float]]) -> List[float]:
        n = len(vectors)
        dim = len(vectors[0])
        return [sum(v[i] for v in vectors) / n for i in range(dim)]

    def _ensure_centroids(self):
        cls = type(self)
        if cls._chitchat_centroid is not None:
            return
        with cls._seed_lock:
            if cls._chitchat_centroid is not None:
                return
            cls._chitchat_centroid = self._mean(self.embedder.embed_documents(_CHITCHAT_SEEDS))
            cls._rag_centroid = self._mean(self.embedder.embed_documents(_RAG_SEEDS))

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _chitchat_response(self, query: str) -> str:
        if re.search(r"\b(who|what)\s+(are|r)\s+(you|u)\b|\bwho\s+built\b|\bwhat\s+can\s+you\s+do\b", query, re.IGNORECASE):
            return _IDENTITY_REPLY
        if re.search(r"\bthanks?\b|\bthank\s*you\b|\bthx\b", query, re.IGNORECASE):
            return _CHITCHAT_REPLIES[2]
        return random.choice(_CHITCHAT_REPLIES[:2])

    def route(self, query: str) -> RouterDecision:
        text = query.strip()
        if not text:
            return RouterDecision(route="chitchat", response=self._chitchat_response(text), confidence=1.0)

        # 1. Regex fast-path
        if _GREETING_RE.match(text):
            return RouterDecision(route="chitchat", response=self._chitchat_response(text), confidence=1.0)

        # 2. Embedding-similarity classification
        try:
            q_emb = self.embedder.embed_query(text)
            sim_chit = self._cosine(q_emb, type(self)._chitchat_centroid)
            sim_rag = self._cosine(q_emb, type(self)._rag_centroid)
            margin = sim_chit - sim_rag
            # Require a clear margin to call it chitchat; ties go to rag.
            if margin > 0.05:
                return RouterDecision(
                    route="chitchat",
                    response=self._chitchat_response(text),
                    confidence=min(1.0, 0.5 + margin),
                )
            return RouterDecision(route="rag", response="", confidence=min(1.0, 0.5 + max(0.0, -margin)))
        except Exception as e:
            print(f"[!] Router failed: {e}. Defaulting to 'rag'.")
            return RouterDecision(route="rag", response="", confidence=0.0)
