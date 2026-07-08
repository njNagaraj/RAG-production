import abc
import json
import os
import threading
from typing import List, Dict, Any, Optional
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from pipeline.config import settings

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations")

class BaseVectorStore(abc.ABC):
    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def initialize_db(self, dimension: int):
        pass

    @abc.abstractmethod
    def insert_document(self, metadata: Dict[str, Any]) -> int:
        pass

    @abc.abstractmethod
    def insert_chunks(self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        pass

    @abc.abstractmethod
    def get_all_documents(self, limit: int = 100, offset: int = 0,
                          allowed_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def list_sources(self, allowed_groups: Optional[List[str]] = None) -> List[str]:
        pass

    @abc.abstractmethod
    def dense_search(self, query_vector: List[float], top_k: int, metadata_filter: Dict[str, Any] = None,
                     allowed_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def lexical_search(self, query_text: str, top_k: int, metadata_filter: Dict[str, Any] = None,
                       allowed_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def check_semantic_cache(self, query_embedding: List[float], similarity_threshold: float = 0.95,
                             access_key: str = "public") -> Any:
        pass

    @abc.abstractmethod
    def add_to_semantic_cache(self, query_text: str, query_embedding: List[float], answer: str,
                              citations: List[Dict[str, Any]], access_key: str = "public"):
        pass

def _vec(embedding: List[float]) -> str:
    """pgvector text representation."""
    return "[" + ",".join(f"{x:.8g}" for x in embedding) + "]"

class PostgresVectorStore(BaseVectorStore):
    """
    Pooled Postgres + pgvector store. One pool per process; every operation
    borrows a connection and returns it. Schema is managed by versioned SQL
    migrations — never destructive DDL at runtime.
    """

    def __init__(self):
        conninfo = (
            f"host={settings.db_host} port={settings.db_port} "
            f"dbname={settings.db_name} user={settings.db_user} "
            f"password={settings.db_password} connect_timeout=5"
        )
        self.pool = ConnectionPool(
            conninfo,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        self._opened = False
        self._open_lock = threading.Lock()

    def connect(self):
        if self._opened:
            return
        with self._open_lock:
            if self._opened:
                return
            try:
                self.pool.open(wait=True, timeout=10)
                self._opened = True
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to PostgreSQL database: {e}.\n"
                    "Please make sure the database is running (docker compose up -d postgres)."
                )

    # ------------------------------------------------------------------ schema

    def initialize_db(self, dimension: int):
        """
        Applies pending migrations (leader-elected via advisory lock) and
        verifies the embedding dimension. A dimension mismatch is a HARD ERROR:
        re-embedding a corpus is an operator decision, never automatic data loss.
        """
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Serialize schema work across api/worker containers
                cur.execute("SELECT pg_advisory_lock(727272);")
            try:
                with conn.cursor() as cur:
                    # Dimension guard FIRST — fail loudly, never drop.
                    # to_regclass returns NULL (no rows) when the table doesn't exist yet.
                    for table, column in (("document_chunks", "embedding"), ("semantic_cache", "query_embedding")):
                        cur.execute(
                            """
                            SELECT atttypmod FROM pg_attribute
                            WHERE attrelid = to_regclass(%s) AND attname = %s;
                            """,
                            (table, column),
                        )
                        row = cur.fetchone()
                        if row and row["atttypmod"] not in (-1, dimension):
                            raise RuntimeError(
                                f"Embedding dimension mismatch on {table}.{column}: database has "
                                f"{row['atttypmod']}, current embedding model produces {dimension}. "
                                f"You changed the embedding model. To proceed, re-embed the corpus: "
                                f"back up the database, drop/recreate the affected tables via a new "
                                f"migration, and re-ingest all documents. Refusing to touch data automatically."
                            )

                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version TEXT PRIMARY KEY,
                            applied_at TIMESTAMPTZ DEFAULT now()
                        );
                    """)
                    cur.execute("SELECT version FROM schema_migrations;")
                    applied = {row["version"] for row in cur.fetchall()}

                    if os.path.isdir(MIGRATIONS_DIR):
                        for fname in sorted(os.listdir(MIGRATIONS_DIR)):
                            if not fname.endswith(".sql") or fname in applied:
                                continue
                            with open(os.path.join(MIGRATIONS_DIR, fname), "r", encoding="utf-8") as f:
                                sql = f.read().replace("{{DIMENSION}}", str(dimension))
                            cur.execute(sql)
                            cur.execute("INSERT INTO schema_migrations (version) VALUES (%s);", (fname,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                # Advisory locks are session-level: they survive the rollback
                # and must be released on a clean transaction.
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(727272);")
                conn.commit()

    # ------------------------------------------------------------------- users

    def create_user(self, username: str, api_key_hash: str, role: str, groups: List[str]) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, api_key_hash, role, groups)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                    """,
                    (username, api_key_hash, role, groups),
                )
                return cur.fetchone()["id"]

    def get_user_by_key_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, groups, active FROM users WHERE api_key_hash = %s;",
                    (key_hash,),
                )
                return cur.fetchone()

    def list_users(self) -> List[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, role, groups, active, created_at FROM users ORDER BY id;")
                return cur.fetchall()

    def set_user_active(self, user_id: int, active: bool) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET active = %s WHERE id = %s RETURNING id;", (active, user_id))
                return cur.fetchone() is not None

    def update_user(self, user_id: int, role: Optional[str] = None,
                    groups: Optional[List[str]] = None, active: Optional[bool] = None) -> bool:
        """Update role, groups, and/or active flag. Only provided fields are changed."""
        self.connect()
        fields, params = [], []
        if role is not None:
            fields.append("role = %s")
            params.append(role)
        if groups is not None:
            fields.append("groups = %s")
            params.append(groups)
        if active is not None:
            fields.append("active = %s")
            params.append(active)
        if not fields:
            return True  # nothing to update
        params.append(user_id)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE users SET {', '.join(fields)} WHERE id = %s RETURNING id;",
                    tuple(params),
                )
                return cur.fetchone() is not None

    def delete_user(self, user_id: int) -> bool:
        """Hard-delete a user. query_audits rows are kept for compliance (user_id becomes orphaned)."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s RETURNING id;", (user_id,))
                return cur.fetchone() is not None

    def rotate_api_key(self, user_id: int, new_key_hash: str) -> bool:
        """Replace a user's API key hash. The old key is immediately invalidated."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET api_key_hash = %s WHERE id = %s RETURNING id;",
                    (new_key_hash, user_id),
                )
                return cur.fetchone() is not None

    def admin_exists(self) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE role = 'admin' AND active LIMIT 1;")
                return cur.fetchone() is not None

    def any_users_exist(self) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users LIMIT 1;")
                return cur.fetchone() is not None

    # --------------------------------------------------------------- documents

    def insert_document(self, metadata: Dict[str, Any]) -> int:
        self.connect()
        params = {
            "source": metadata.get("source"),
            "title": metadata.get("title"),
            "total_pages": metadata.get("total_pages"),
            "file_path": metadata.get("file_path"),
            "allowed_groups": metadata.get("allowed_groups") or [],
            "uploaded_by": metadata.get("uploaded_by"),
        }
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (source, title, total_pages, file_path, allowed_groups, uploaded_by)
                    VALUES (%(source)s, %(title)s, %(total_pages)s, %(file_path)s, %(allowed_groups)s, %(uploaded_by)s)
                    ON CONFLICT (source) DO UPDATE SET
                        title = EXCLUDED.title,
                        total_pages = EXCLUDED.total_pages,
                        file_path = EXCLUDED.file_path,
                        allowed_groups = EXCLUDED.allowed_groups,
                        uploaded_by = COALESCE(EXCLUDED.uploaded_by, documents.uploaded_by)
                    RETURNING id;
                    """,
                    params,
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("Failed to retrieve document ID after insert")
                return row["id"]

    def insert_chunks(self, document_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Replaces a document's chunks atomically using COPY (bulk load)."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                with cur.copy(
                    "COPY document_chunks (document_id, chunk_index, page_num, content, context, embedding) FROM STDIN"
                ) as copy:
                    for chunk, embedding in zip(chunks, embeddings):
                        copy.write_row((
                            document_id,
                            chunk["chunk_index"],
                            chunk["page_num"],
                            chunk["text"],
                            chunk.get("context"),
                            _vec(embedding),
                        ))
            conn.commit()

    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, source, allowed_groups, uploaded_by FROM documents WHERE id = %s;",
                    (document_id,),
                )
                return cur.fetchone()

    def delete_document(self, document_id: int) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s RETURNING id;", (document_id,))
                return cur.fetchone() is not None

    @staticmethod
    def _group_visibility_sql(allowed_groups: Optional[List[str]], params: list, alias: str = "d") -> List[str]:
        """None = unrestricted (admin). Otherwise: public docs OR group overlap."""
        if allowed_groups is None:
            return []
        params.append(allowed_groups)
        return [f"({alias}.allowed_groups = '{{}}' OR {alias}.allowed_groups && %s)"]

    def get_all_documents(self, limit: int = 100, offset: int = 0,
                          allowed_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self.connect()
        params: list = []
        clauses = self._group_visibility_sql(allowed_groups, params)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([min(limit, 500), max(offset, 0)])
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, source, title, total_pages, file_path, allowed_groups, created_at
                    FROM documents d {where_sql} ORDER BY id DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cur.fetchall()

    def list_sources(self, allowed_groups: Optional[List[str]] = None) -> List[str]:
        self.connect()
        params: list = []
        clauses = self._group_visibility_sql(allowed_groups, params)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT source FROM documents d {where_sql};", tuple(params))
                return [row["source"] for row in cur.fetchall() if row["source"]]

    # ------------------------------------------------------------------ search

    def _filter_sql(self, metadata_filter: Optional[Dict[str, Any]], params: list,
                    allowed_groups: Optional[List[str]] = None) -> List[str]:
        # RBAC visibility is part of the WHERE clause — retrieval-level
        # enforcement, never post-filtering.
        clauses = self._group_visibility_sql(allowed_groups, params)
        if metadata_filter:
            if "source_doc" in metadata_filter:
                clauses.append("d.source ILIKE %s")
                params.append(f"%{metadata_filter['source_doc']}%")
            if "page_num" in metadata_filter:
                clauses.append("dc.page_num = %s")
                params.append(metadata_filter["page_num"])
        return clauses

    def dense_search(self, query_vector: List[float], top_k: int, metadata_filter: Dict[str, Any] = None,
                     allowed_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self.connect()
        vec = _vec(query_vector)
        params: list = [vec]
        where_clauses = self._filter_sql(metadata_filter, params, allowed_groups)
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.extend([vec, top_k])

        query_sql = f"""
            SELECT
                dc.id,
                dc.chunk_index,
                dc.page_num,
                dc.content,
                dc.context,
                d.source as source_doc,
                1 - (dc.embedding <=> %s::vector) AS score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            {where_sql}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL hnsw.ef_search = %s;" % int(settings.hnsw_ef_search))
                cur.execute(query_sql, tuple(params))
                return cur.fetchall()

    def lexical_search(self, query_text: str, top_k: int, metadata_filter: Dict[str, Any] = None,
                       allowed_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self.connect()
        if not query_text.strip():
            return []

        params: list = [query_text, query_text]
        where_clauses = ["dc.content_tsvector @@ websearch_to_tsquery('english', %s)"]
        where_clauses += self._filter_sql(metadata_filter, params, allowed_groups)
        where_sql = "WHERE " + " AND ".join(where_clauses)
        params.append(top_k)

        query_sql = f"""
            SELECT
                dc.id,
                dc.chunk_index,
                dc.page_num,
                dc.content,
                dc.context,
                d.source as source_doc,
                ts_rank_cd(dc.content_tsvector, websearch_to_tsquery('english', %s)) AS score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            {where_sql}
            ORDER BY score DESC
            LIMIT %s;
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, tuple(params))
                return cur.fetchall()

    # ---------------------------------------------------------- semantic cache

    def get_corpus_epoch(self) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM rag_meta WHERE key = 'corpus_epoch';")
                row = cur.fetchone()
                return int(row["value"]) if row else 0

    def bump_corpus_epoch(self):
        """Called after every ingest/delete: invalidates all older cache entries."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_meta (key, value) VALUES ('corpus_epoch', '1')
                    ON CONFLICT (key) DO UPDATE SET value = (rag_meta.value::int + 1)::text;
                    """
                )
            conn.commit()

    def check_semantic_cache(self, query_embedding: List[float], similarity_threshold: float = 0.95,
                             access_key: str = "public") -> Any:
        self.connect()
        vec = _vec(query_embedding)
        epoch = self.get_corpus_epoch()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        query_text,
                        answer,
                        citations,
                        1 - (query_embedding <=> %s::vector) AS similarity
                    FROM semantic_cache
                    WHERE access_key = %s
                      AND corpus_epoch = %s
                      AND created_at > now() - make_interval(hours => %s)
                      AND 1 - (query_embedding <=> %s::vector) >= %s
                    ORDER BY query_embedding <=> %s::vector
                    LIMIT 1;
                    """,
                    (vec, access_key, epoch, settings.cache_ttl_hours, vec, similarity_threshold, vec),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "query_text": row["query_text"],
                        "answer": row["answer"],
                        "citations": row["citations"],
                        "similarity": float(row["similarity"]),
                    }
                return None

    def add_to_semantic_cache(self, query_text: str, query_embedding: List[float], answer: str,
                              citations: List[Dict[str, Any]], access_key: str = "public"):
        self.connect()
        epoch = self.get_corpus_epoch()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_cache (query_text, query_embedding, answer, citations, corpus_epoch, access_key)
                    VALUES (%s, %s::vector, %s, %s, %s, %s)
                    ON CONFLICT (query_text, access_key) DO UPDATE SET
                        query_embedding = EXCLUDED.query_embedding,
                        answer = EXCLUDED.answer,
                        citations = EXCLUDED.citations,
                        corpus_epoch = EXCLUDED.corpus_epoch,
                        created_at = CURRENT_TIMESTAMP;
                    """,
                    (query_text, _vec(query_embedding), answer, json.dumps(citations), epoch, access_key),
                )
            conn.commit()

    def cleanup_semantic_cache(self):
        """Removes stale-epoch and expired cache rows. Run periodically by the worker."""
        self.connect()
        epoch = self.get_corpus_epoch()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM semantic_cache
                    WHERE corpus_epoch < %s
                       OR created_at < now() - make_interval(hours => %s);
                    """,
                    (epoch, settings.cache_ttl_hours),
                )
            conn.commit()

    # ------------------------------------------------------------- ingest jobs

    def enqueue_ingest_job(self, filename: str, stored_path: str, enrich: bool = False,
                           allowed_groups: Optional[List[str]] = None,
                           uploaded_by: Optional[int] = None) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingest_jobs (filename, stored_path, enrich, status, allowed_groups, uploaded_by)
                    VALUES (%s, %s, %s, 'queued', %s, %s) RETURNING id;
                    """,
                    (filename, stored_path, enrich, allowed_groups or [], uploaded_by),
                )
                return cur.fetchone()["id"]

    def claim_next_ingest_job(self) -> Optional[Dict[str, Any]]:
        """Atomically claims one queued job (safe with multiple workers)."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_jobs SET status = 'processing', started_at = now()
                    WHERE id = (
                        SELECT id FROM ingest_jobs WHERE status = 'queued'
                        ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
                    )
                    RETURNING id, filename, stored_path, enrich, allowed_groups, uploaded_by;
                    """
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def finish_ingest_job(self, job_id: int, status: str, error: Optional[str] = None,
                          chunks_count: Optional[int] = None, document_id: Optional[int] = None):
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_jobs
                    SET status = %s, error = %s, chunks_count = %s, document_id = %s, finished_at = now()
                    WHERE id = %s;
                    """,
                    (status, error, chunks_count, document_id, job_id),
                )
            conn.commit()

    def get_ingest_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, enrich, status, error, chunks_count, document_id,
                           allowed_groups, uploaded_by, created_at, started_at, finished_at
                    FROM ingest_jobs WHERE id = %s;
                    """,
                    (job_id,),
                )
                return cur.fetchone()

    def save_query_audit(self, question: str, answer: str, score: float, faithful: bool, reason: str,
                         user_id: Optional[int] = None):
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_audits (question, answer, faithfulness_score, faithful, reason, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (question, answer, score, faithful, reason, user_id),
                )
            conn.commit()

    def count_queued_jobs(self) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM ingest_jobs WHERE status IN ('queued', 'processing');")
                return cur.fetchone()["n"]

_store_instance: Optional[PostgresVectorStore] = None
_store_lock = threading.Lock()

def get_vector_store() -> BaseVectorStore:
    """Process-wide singleton so the connection pool is shared."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = PostgresVectorStore()
    return _store_instance
