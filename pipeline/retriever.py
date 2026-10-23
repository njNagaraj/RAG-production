import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from pipeline.config import settings
from pipeline.embedder import BaseEmbedder
from pipeline.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# --- Local adaptive query transformation (no LLM) ---------------------------
# Query types and their retrieval strategies:
#   exact_lookup — IDs, invoice numbers, dates, quoted strings → weight lexical
#                  (BM25/FTS) side higher in fusion so exact tokens win
#   multi_part   — compound questions → split into sub-queries, retrieve each, fuse
#   follow_up    — short/pronoun-heavy query with history → append last user turns
#   conceptual   — default pass-through

_ID_TOKEN_RE = re.compile(
    r"(\"[^\"]+\"|'[^']+')"                 # quoted phrases
    r"|\b[A-Z]{2,}[-_/]?\d+[\w-]*\b"        # INV-2024-001, PO123, L3-A
    r"|\b\d{4}-\d{2}-\d{2}\b"               # ISO dates
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"   # 12/05/2024
    r"|#\w+"                                # #4521
    r"|\b[\w.+-]+@[\w-]+\.\w+\b"            # emails
)

_FOLLOW_UP_PRONOUN_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|him|her|the same|also|too|what about|and for)\b",
    re.IGNORECASE,
)

_MULTI_PART_SPLIT_RE = re.compile(
    r"\?\s+(?=[a-zA-Z])"          # "...? And what..."
    r"|\s*;\s*"
    r"|\s+\band\s+(?=(what|when|where|who|how|which|why|is|are|does|do|can)\b)",
    re.IGNORECASE,
)

def classify_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    if _ID_TOKEN_RE.search(query):
        return "exact_lookup"
    parts = [p for p in _MULTI_PART_SPLIT_RE.split(query) if p and len(p.strip()) > 10 and not re.fullmatch(r"(what|when|where|who|how|which|why|is|are|does|do|can)", p.strip(), re.IGNORECASE)]
    if query.count("?") >= 2 or len(parts) >= 2:
        return "multi_part"
    words = query.split()
    if history and (len(words) <= 6 or _FOLLOW_UP_PRONOUN_RE.search(query)):
        return "follow_up"
    return "conceptual"

def transform_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Returns: {"query_type", "search_queries": [...], "standalone_query", "lexical_weight"}
    """
    qtype = classify_query(query, history)

    if qtype == "exact_lookup":
        return {
            "query_type": qtype,
            "search_queries": [query],
            "standalone_query": query,
            "lexical_weight": 2.0,  # exact tokens matter — boost FTS side of RRF
        }

    if qtype == "multi_part":
        sub_queries = [p.strip(" ?.,;") + "?" for p in _MULTI_PART_SPLIT_RE.split(query) if p and len(p.strip()) > 10]
        if not sub_queries:
            sub_queries = [query]
        return {
            "query_type": qtype,
            "search_queries": sub_queries[:4],
            "standalone_query": query,
            "lexical_weight": 1.0,
        }

    if qtype == "follow_up":
        user_turns = [t.get("content", "") for t in (history or []) if t.get("role") == "user"]
        context_turns = " ".join(user_turns[-2:])
        standalone = f"{context_turns} {query}".strip()
        return {
            "query_type": qtype,
            "search_queries": [standalone],
            "standalone_query": standalone,
            "lexical_weight": 1.0,
        }

    return {
        "query_type": "conceptual",
        "search_queries": [query],
        "standalone_query": query,
        "lexical_weight": 1.0,
    }

# --- Metadata filter extraction (regex, no LLM) ------------------------------

_PAGE_RE = re.compile(r"\b(?:on\s+|at\s+)?page\s+(\d+)\b", re.IGNORECASE)
_FILE_RE = re.compile(r"\b([\w][\w .()\-]*\.(?:pdf|txt|md|markdown|docx))\b", re.IGNORECASE)

def extract_metadata_filter(query: str, known_sources: Optional[List[str]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    page_match = _PAGE_RE.search(query)
    if page_match:
        meta["page_num"] = int(page_match.group(1))
    file_match = _FILE_RE.search(query)
    if file_match:
        candidate = file_match.group(1).strip()
        if known_sources:
            lowered = candidate.lower()
            for src in known_sources:
                if lowered in src.lower() or src.lower() in lowered:
                    meta["source_doc"] = src
                    break
            else:
                meta["source_doc"] = candidate
        else:
            meta["source_doc"] = candidate
    return meta

# --- Weighted Reciprocal Rank Fusion -----------------------------------------

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k: int = 60,
    lexical_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    RRF Score = w_m * 1 / (rank_m + k), summed over methods.
    lexical_weight > 1 boosts exact-token matches (exact_lookup queries).
    """
    rrf_scores: Dict[Any, float] = {}
    doc_map: Dict[Any, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results):
        chunk_id = item["id"]
        doc_map.setdefault(chunk_id, item)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rank + 1 + k))

    for rank, item in enumerate(lexical_results):
        chunk_id = item["id"]
        doc_map.setdefault(chunk_id, item)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (lexical_weight / (rank + 1 + k))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for chunk_id in sorted_ids:
        item = doc_map[chunk_id].copy()
        item["rrf_score"] = rrf_scores[chunk_id]
        fused_results.append(item)
    return fused_results

def convex_fusion(
    dense_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    dense_weight: float = 0.7,
    lexical_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Score = a * norm(dense) + (1 - a) * w * norm(lexical), with each list
    min-max normalized to [0, 1] and a chunk missing from a list scoring 0
    there. Unlike RRF it keeps how far apart the scores are, so a weak
    lexical leg cannot outvote a confident dense match.
    BEIR nDCG@10, dense alone -> RRF -> convex 0.7:
    SciFact 0.702 -> 0.696 -> 0.723; NFCorpus 0.348 -> 0.356 -> 0.358.
    """
    def normalize(results: List[Dict[str, Any]]) -> Dict[Any, float]:
        if not results:
            return {}
        scores = [float(r["score"]) for r in results]
        lo, hi = min(scores), max(scores)
        return {r["id"]: (float(r["score"]) - lo) / (hi - lo) if hi > lo else 1.0 for r in results}

    nd, nl = normalize(dense_results), normalize(lexical_results)
    wl = (1.0 - dense_weight) * lexical_weight
    doc_map: Dict[Any, Dict[str, Any]] = {}
    for item in dense_results + lexical_results:
        doc_map.setdefault(item["id"], item)
    fused = {cid: dense_weight * nd.get(cid, 0.0) + wl * nl.get(cid, 0.0) for cid in doc_map}

    fused_results = []
    for chunk_id in sorted(fused, key=fused.get, reverse=True):
        item = doc_map[chunk_id].copy()
        item["fusion_score"] = fused[chunk_id]
        fused_results.append(item)
    return fused_results

def collapse_shared_parents(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parent-child chunks return the parent text as `content`, so sibling
    children of one parent arrive as duplicates. Keep the best-ranked sibling
    per (source, parent text); the reranker and LLM then see distinct passages.
    """
    seen = set()
    collapsed = []
    for item in results:
        key = (item.get("source_doc"), item.get("content"))
        if key in seen:
            continue
        seen.add(key)
        collapsed.append(item)
    return collapsed

# --- Retriever ----------------------------------------------------------------

class Retriever:
    def __init__(self, vector_store: BaseVectorStore, embedder: BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, history: List[Dict[str, str]] = None,
                 org_id: int = None, visible_node_ids: Optional[List[int]] = None
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fully local hybrid retrieval:
        adaptive transform -> dense + FTS per sub-query -> weighted RRF fusion.
        org_id: tenant scope (required). visible_node_ids: hierarchy document
        visibility (None = org-wide owner/admin); both enforced inside the SQL
        WHERE clause of both search legs.
        """
        # 1. Adaptive query transformation (rule-based, local)
        transform = transform_query(query, history or [])
        search_queries = transform["search_queries"]
        standalone_query = transform["standalone_query"]

        # 2. Metadata filters (regex, validated against sources the user can see)
        try:
            known_sources = self.vector_store.list_sources(org_id=org_id, visible_node_ids=visible_node_ids)
        except Exception:
            known_sources = []
        metadata_filter = extract_metadata_filter(standalone_query, known_sources)

        # 3. Dense retrieval per sub-query
        all_dense_results: List[Dict[str, Any]] = []
        for q in search_queries:
            try:
                q_emb = self.embedder.embed_query(q)
                results = self.vector_store.dense_search(
                    q_emb, top_k=settings.top_k_retrieval, org_id=org_id,
                    visible_node_ids=visible_node_ids, metadata_filter=metadata_filter,
                )
                all_dense_results.extend(results)
            except Exception as e:
                logger.warning("Dense search failed for query %r: %s", q, e)

        # Deduplicate dense results keeping highest score
        dense_dict: Dict[Any, Dict[str, Any]] = {}
        for item in all_dense_results:
            cid = item["id"]
            if cid not in dense_dict or item["score"] > dense_dict[cid]["score"]:
                dense_dict[cid] = item
        dedup_dense = sorted(dense_dict.values(), key=lambda x: x["score"], reverse=True)
        dedup_dense = dedup_dense[:settings.top_k_retrieval]

        # 4. Lexical retrieval (Postgres FTS) per sub-query
        all_lexical_results: List[Dict[str, Any]] = []
        for q in search_queries:
            try:
                all_lexical_results.extend(
                    self.vector_store.lexical_search(q, top_k=settings.top_k_retrieval, org_id=org_id,
                                                     visible_node_ids=visible_node_ids,
                                                     metadata_filter=metadata_filter)
                )
            except Exception as e:
                logger.warning("Lexical search failed for query %r: %s", q, e)
        lexical_dict: Dict[Any, Dict[str, Any]] = {}
        for item in all_lexical_results:
            cid = item["id"]
            if cid not in lexical_dict or item["score"] > lexical_dict[cid]["score"]:
                lexical_dict[cid] = item
        dedup_lexical = sorted(lexical_dict.values(), key=lambda x: x["score"], reverse=True)
        dedup_lexical = dedup_lexical[:settings.top_k_retrieval]

        # 5. Fusion: convex score mix (default) or weighted Reciprocal Rank Fusion
        if settings.fusion_method.lower() == "rrf":
            fused_results = reciprocal_rank_fusion(
                dedup_dense,
                dedup_lexical,
                k=settings.rrf_constant,
                lexical_weight=transform["lexical_weight"],
            )
        else:
            fused_results = convex_fusion(
                dedup_dense,
                dedup_lexical,
                dense_weight=settings.fusion_dense_weight,
                lexical_weight=transform["lexical_weight"],
            )
        fused_results = collapse_shared_parents(fused_results)

        debug_info = {
            "standalone_query": standalone_query,
            "query_type": transform["query_type"],
            "expanded_queries": search_queries,
            "dense_count": len(dedup_dense),
            "lexical_count": len(dedup_lexical),
            "fused_count": len(fused_results),
            "metadata_filter": metadata_filter,
        }
        return fused_results, debug_info
