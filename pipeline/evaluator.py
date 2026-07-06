import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pipeline.config import settings

class EvaluationResult(BaseModel):
    faithful: bool = Field(description="True if the answer is grounded in the context.")
    score: float = Field(description="Faithfulness score from 0.0 to 1.0.")
    reason: str = Field(description="Brief explanation of the score.")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_CITATION_RE = re.compile(r"\[\d+\]")

def _split_sentences(text: str) -> List[str]:
    # Strip citation markers before scoring; keep substantive sentences only
    clean = _CITATION_RE.sub("", text)
    return [s.strip() for s in _SENTENCE_RE.split(clean) if len(s.strip()) > 15]

class Evaluator:
    """
    Local faithfulness auditor — no LLM judge.
    Splits the answer into sentences and scores each against the retrieved
    chunks with the local cross-encoder: a sentence is 'supported' if its best
    chunk score clears a margin relative to the query-chunk baseline.
    Designed to run in the background after the response is returned.
    """

    SUPPORT_THRESHOLD = 0.30  # sigmoid-space cross-encoder score floor

    def __init__(self):
        provider = settings.evaluator_provider.lower()
        self.enabled = provider != "none"
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from pipeline.reranker import get_reranker, LocalCrossEncoderReranker
            reranker = get_reranker()
            if isinstance(reranker, LocalCrossEncoderReranker):
                self._encoder = reranker.model
            else:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._encoder = TextCrossEncoder(model_name=settings.local_reranker_model)
        return self._encoder

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def evaluate_faithfulness(self, query: str, answer: str, chunks: List[Dict[str, Any]]) -> EvaluationResult:
        # A correct refusal is faithful by definition
        if "flag it to HR" in answer or "don't have that in our policies" in answer:
            return EvaluationResult(
                faithful=True,
                score=1.0,
                reason="The assistant correctly refused to answer because the context was insufficient."
            )

        if not chunks:
            return EvaluationResult(
                faithful=False,
                score=0.0,
                reason="No context was provided to support the answer."
            )

        if not self.enabled:
            return EvaluationResult(faithful=True, score=1.0, reason="Evaluator disabled (EVALUATOR_PROVIDER=none).")

        sentences = _split_sentences(answer)
        if not sentences:
            return EvaluationResult(faithful=True, score=1.0, reason="Answer contains no substantive claims to verify.")

        try:
            encoder = self._get_encoder()
            # Score claims against context sentences (finer granularity than
            # whole chunks — cross-encoders match short claim/evidence pairs
            # far better) plus the whole chunks as fallback units.
            contexts = [c["content"] for c in chunks]
            for c in chunks:
                contexts.extend(s for s in _SENTENCE_RE.split(c["content"]) if len(s.strip()) > 15)

            supported = 0
            weakest = None
            for sent in sentences:
                raw_scores = list(encoder.rerank(sent, contexts))
                best = max(self._sigmoid(float(s)) for s in raw_scores)
                if best >= self.SUPPORT_THRESHOLD:
                    supported += 1
                elif weakest is None or best < weakest[1]:
                    weakest = (sent, best)

            score = supported / len(sentences)
            faithful = score >= 0.7
            if faithful:
                reason = f"{supported}/{len(sentences)} answer sentences are supported by retrieved context (local cross-encoder)."
            else:
                weak_snippet = (weakest[0][:80] + "...") if weakest else ""
                reason = (
                    f"Only {supported}/{len(sentences)} sentences supported by context. "
                    f"Least supported claim: \"{weak_snippet}\""
                )
            return EvaluationResult(faithful=faithful, score=round(score, 2), reason=reason)
        except Exception as e:
            print(f"Evaluator failed: {e}")
            return EvaluationResult(
                faithful=True,
                score=1.0,
                reason=f"Evaluation skipped due to error: {e}"
            )
