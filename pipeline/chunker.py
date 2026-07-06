import os
import re
import threading
from typing import List, Dict, Any, Optional
from pipeline.config import settings

# ---------------------------------------------------------------------------
# Token counting — tokenizer-accurate (embedding model's tokenizer), with a
# deterministic fallback so chunking never depends on the network at runtime.
# ---------------------------------------------------------------------------

_tokenizer = None
_tokenizer_lock = threading.Lock()
_tokenizer_failed = False

def count_tokens(text: str) -> int:
    global _tokenizer, _tokenizer_failed
    if _tokenizer is None and not _tokenizer_failed:
        with _tokenizer_lock:
            if _tokenizer is None and not _tokenizer_failed:
                try:
                    from tokenizers import Tokenizer
                    _tokenizer = Tokenizer.from_pretrained(settings.local_embedding_model)
                except Exception:
                    _tokenizer_failed = True
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception:
            pass
    # Fallback: ~1.3 tokens per whitespace word is a decent BERT-family estimate
    return max(1, int(len(text.split()) * 1.3))

# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class Chunk:
    def __init__(self, text: str, doc_metadata: Dict[str, Any], chunk_index: int,
                 page_num: int = 1, context: Optional[str] = None):
        self.text = text
        self.doc_metadata = doc_metadata
        self.chunk_index = chunk_index
        self.page_num = page_num
        self.context = context

    @property
    def full_text(self) -> str:
        if self.context:
            return f"[Context: {self.context}]\n{self.text}"
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "doc_metadata": self.doc_metadata,
            "chunk_index": self.chunk_index,
            "page_num": self.page_num,
            "context": self.context,
            "full_text": self.full_text,
        }

# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala", ".sql",
}

_CODE_BOUNDARY_RE = re.compile(
    r"^(?=(?:export\s+)?(?:public|private|protected|static|async)?\s*"
    r"(?:def|class|function|func|fn|interface|struct|impl|module|CREATE)\b)",
    re.MULTILINE,
)

def _split_sentences(text: str) -> List[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]

def _split_blocks(page_text: str) -> List[Dict[str, Any]]:
    """
    Splits page text into blocks that must never be broken internally:
    markdown tables stay whole; everything else splits by blank lines.
    Tracks the current heading path for structural context.
    """
    blocks: List[Dict[str, Any]] = []
    heading_path: List[str] = []
    lines = page_text.split("\n")
    buf: List[str] = []
    in_table = False

    def flush(is_table: bool):
        nonlocal buf
        joined = "\n".join(buf).strip()
        if joined:
            blocks.append({"text": joined, "heading": " > ".join(heading_path), "is_table": is_table})
        buf = []

    for line in lines:
        heading_m = _HEADING_RE.match(line)
        is_table_line = bool(_TABLE_LINE_RE.match(line))
        if heading_m:
            flush(in_table)
            in_table = False
            level = len(heading_m.group(1))
            heading_path[:] = heading_path[: level - 1]
            heading_path.append(heading_m.group(2).strip())
            blocks.append({"text": line.strip(), "heading": " > ".join(heading_path), "is_table": False, "is_heading": True})
        elif is_table_line:
            if not in_table:
                flush(False)
                in_table = True
            buf.append(line)
        elif not line.strip():
            flush(in_table)
            in_table = False
        else:
            if in_table:
                flush(True)
                in_table = False
            buf.append(line)
    flush(in_table)
    return blocks

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class _BaseStrategy:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        raise NotImplementedError

    def _pack_units(self, units: List[Dict[str, Any]], doc_metadata: Dict[str, Any],
                    start_index: int) -> List[Chunk]:
        """
        Greedy packer: accumulates units (sentences/paragraphs/tables) into
        chunks up to chunk_size tokens, carrying overlap. Oversized atomic
        units (e.g. a huge table) become their own chunk — never split.
        """
        chunks: List[Chunk] = []
        current: List[Dict[str, Any]] = []
        current_len = 0
        chunk_idx = start_index

        def emit():
            nonlocal current, current_len, chunk_idx
            if not current:
                return
            text = "\n\n".join(u["text"] for u in current)
            heading = current[0].get("heading") or ""
            chunks.append(Chunk(
                text=text,
                doc_metadata=doc_metadata,
                chunk_index=chunk_idx,
                page_num=current[0].get("page_num", 1),
                context=self._context_for(doc_metadata, heading),
            ))
            chunk_idx += 1
            # Overlap: carry trailing units within the overlap budget
            carried, carried_len = [], 0
            for u in reversed(current):
                u_len = count_tokens(u["text"])
                if carried_len + u_len > self.chunk_overlap:
                    break
                carried.insert(0, u)
                carried_len += u_len
            current = carried
            current_len = carried_len

        for unit in units:
            u_len = count_tokens(unit["text"])
            if u_len > self.chunk_size:
                # Atomic oversized unit (table/paragraph): emit pending, then emit alone
                if unit.get("is_table"):
                    emit()
                    current = [unit]
                    current_len = u_len
                    emit()
                    current, current_len = [], 0
                    continue
                # Non-atomic: recurse into sentences
                for sent in _split_sentences(unit["text"]):
                    sub = {**unit, "text": sent}
                    s_len = count_tokens(sent)
                    if current_len + s_len > self.chunk_size and current:
                        emit()
                    current.append(sub)
                    current_len += s_len
                continue
            if current_len + u_len > self.chunk_size and current:
                emit()
            current.append(unit)
            current_len += u_len
        # Final flush without overlap carry
        if current:
            text = "\n\n".join(u["text"] for u in current)
            heading = current[0].get("heading") or ""
            chunks.append(Chunk(
                text=text,
                doc_metadata=doc_metadata,
                chunk_index=chunk_idx,
                page_num=current[0].get("page_num", 1),
                context=self._context_for(doc_metadata, heading),
            ))
        return chunks

    @staticmethod
    def _context_for(doc_metadata: Dict[str, Any], heading: str) -> Optional[str]:
        """Structural contextual enrichment: document title + section path. No LLM."""
        title = doc_metadata.get("title") or doc_metadata.get("source") or ""
        parts = [p for p in (title, heading) if p]
        return " — ".join(parts) if parts else None

class StructureAwareStrategy(_BaseStrategy):
    """Recursive splitting on headings/sections/table boundaries, then paragraphs, then sentences."""

    def split(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        doc_metadata = doc_data["metadata"]
        chunks: List[Chunk] = []
        for page in doc_data.get("pages", []):
            page_text = page.get("text", "")
            if not page_text.strip():
                continue
            units = _split_blocks(page_text)
            for u in units:
                u["page_num"] = page.get("page_num", 1)
            chunks.extend(self._pack_units(units, doc_metadata, start_index=len(chunks)))
        return chunks

class PageLevelStrategy(_BaseStrategy):
    """Each page is one chunk (invoices, resumes, statements). Oversized pages split structurally."""

    def split(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        doc_metadata = doc_data["metadata"]
        chunks: List[Chunk] = []
        for page in doc_data.get("pages", []):
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
            page_num = page.get("page_num", 1)
            if count_tokens(page_text) <= self.chunk_size * 2:
                chunks.append(Chunk(
                    text=page_text,
                    doc_metadata=doc_metadata,
                    chunk_index=len(chunks),
                    page_num=page_num,
                    context=self._context_for(doc_metadata, f"Page {page_num}"),
                ))
            else:
                units = _split_blocks(page_text)
                for u in units:
                    u["page_num"] = page_num
                chunks.extend(self._pack_units(units, doc_metadata, start_index=len(chunks)))
        return chunks

class CodeAwareStrategy(_BaseStrategy):
    """Splits source files on function/class/module boundaries."""

    def split(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        doc_metadata = doc_data["metadata"]
        content = doc_data.get("content", "")
        if not content.strip():
            return []
        pieces = _CODE_BOUNDARY_RE.split(content)
        units = [{"text": p.rstrip(), "heading": "", "is_table": False} for p in pieces if p.strip()]
        return self._pack_units(units, doc_metadata, start_index=0)

class SemanticStrategy(_BaseStrategy):
    """
    Opt-in: local-embedding topic-shift breakpoints. Groups consecutive
    sentences; starts a new chunk when cosine similarity to the running
    chunk centroid drops below threshold.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, breakpoint_similarity: float = 0.55):
        super().__init__(chunk_size, chunk_overlap)
        self.breakpoint_similarity = breakpoint_similarity

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def split(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        from pipeline.embedder import get_embedder
        embedder = get_embedder()
        doc_metadata = doc_data["metadata"]
        chunks: List[Chunk] = []

        for page in doc_data.get("pages", []):
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
            page_num = page.get("page_num", 1)
            sentences = _split_sentences(page_text)
            if not sentences:
                continue
            embeddings = embedder.embed_documents(sentences)

            group: List[str] = []
            group_len = 0
            centroid: Optional[List[float]] = None

            def emit_group():
                nonlocal group, group_len, centroid
                if group:
                    chunks.append(Chunk(
                        text=" ".join(group),
                        doc_metadata=doc_metadata,
                        chunk_index=len(chunks),
                        page_num=page_num,
                        context=self._context_for(doc_metadata, ""),
                    ))
                group, group_len, centroid = [], 0, None

            for sent, emb in zip(sentences, embeddings):
                s_len = count_tokens(sent)
                topic_shift = centroid is not None and self._cosine(emb, centroid) < self.breakpoint_similarity
                if group and (topic_shift or group_len + s_len > self.chunk_size):
                    emit_group()
                group.append(sent)
                group_len += s_len
                if centroid is None:
                    centroid = list(emb)
                else:
                    n = len(group)
                    centroid = [(c * (n - 1) + e) / n for c, e in zip(centroid, emb)]
            emit_group()
        return chunks

# ---------------------------------------------------------------------------
# Adaptive dispatcher (IG "hybrid/agentic chunking")
# ---------------------------------------------------------------------------

class Chunker:
    """
    Chooses the chunking strategy by document type:
      code file          -> code-aware (function/class boundaries)
      page-unit document -> page-level (invoices/resumes/statements)
      everything else    -> recursive structure-aware (headings/sections/tables,
                            then paragraphs, then sentences)
    CHUNKING_STRATEGY env forces a single strategy (fixed|page|code|semantic).
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def _make(self, name: str) -> _BaseStrategy:
        if name == "code":
            return CodeAwareStrategy(self.chunk_size, self.chunk_overlap)
        if name == "page":
            return PageLevelStrategy(self.chunk_size, self.chunk_overlap)
        if name == "semantic":
            return SemanticStrategy(self.chunk_size, self.chunk_overlap)
        return StructureAwareStrategy(self.chunk_size, self.chunk_overlap)

    def choose_strategy(self, doc_data: Dict[str, Any]) -> str:
        forced = settings.chunking_strategy.lower()
        if forced not in ("auto", ""):
            return forced

        file_path = doc_data.get("metadata", {}).get("file_path", "") or ""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _CODE_EXTENSIONS:
            return "code"

        pages = doc_data.get("pages", [])
        # Page-unit heuristic: multi-page doc whose pages are short, self-contained
        # units (invoices, resumes, statements average well under chunk_size tokens)
        if len(pages) >= 2:
            page_lens = [count_tokens(p.get("text", "")) for p in pages if p.get("text", "").strip()]
            if page_lens and (sum(page_lens) / len(page_lens)) <= self.chunk_size * 0.6:
                return "page"
        return "structure"

    def split_document(self, doc_data: Dict[str, Any]) -> List[Chunk]:
        strategy_name = self.choose_strategy(doc_data)
        strategy = self._make(strategy_name)
        chunks = strategy.split(doc_data)
        for c in chunks:
            c.doc_metadata = {**c.doc_metadata, "chunking_strategy": strategy_name}
        return chunks

    def enrich_context(self, chunks: List[Chunk], full_document_content: str = "") -> List[Chunk]:
        """
        Structural contextual enrichment (local, no LLM): chunks already carry
        'document title — section path' context from the strategy packers.
        Kept for API compatibility; fills context only where missing.
        """
        for chunk in chunks:
            if not chunk.context:
                title = chunk.doc_metadata.get("title") or chunk.doc_metadata.get("source") or ""
                chunk.context = title or None
        return chunks
