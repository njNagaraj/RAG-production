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
    def insert_chunks(self, document_id: int, chunks: List[Dict[str, Any]],
                      embeddings: List[List[float]], org_id: int):
        pass

    @abc.abstractmethod
    def get_all_documents(self, org_id: int, visible_node_ids: Optional[List[int]] = None,
                          limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def list_sources(self, org_id: int, visible_node_ids: Optional[List[int]] = None) -> List[str]:
        pass

    @abc.abstractmethod
    def dense_search(self, query_vector: List[float], top_k: int, org_id: int,
                     visible_node_ids: Optional[List[int]] = None,
                     metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def lexical_search(self, query_text: str, top_k: int, org_id: int,
                       visible_node_ids: Optional[List[int]] = None,
                       metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def check_semantic_cache(self, query_embedding: List[float], similarity_threshold: float = 0.95,
                             access_key: str = "public") -> Any:
        pass

    @abc.abstractmethod
    def add_to_semantic_cache(self, query_text: str, query_embedding: List[float], answer: str,
                              citations: List[Dict[str, Any]], org_id: int, access_key: str = "public"):
        pass

def _vec(embedding: List[float]) -> str:
    """pgvector text representation."""
    return "[" + ",".join(f"{x:.8g}" for x in embedding) + "]"

class PostgresVectorStore(BaseVectorStore):
    """
    Pooled Postgres + pgvector store. One pool per process; every operation
    borrows a connection and returns it. Schema is managed by versioned SQL
    migrations — never destructive DDL at runtime.

    Multi-tenant: every document/chunk/cache/job/audit row carries org_id;
    document visibility is resolved through the org_units hierarchy.
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
        self._iterative_scan: Optional[bool] = None

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
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(727272);")
                conn.commit()

    # ----------------------------------------------------------- organizations

    def create_organization(self, name: str, slug: str, owner_username: str,
                            owner_key_hash: str) -> Dict[str, Any]:
        """
        Provisions a tenant transactionally: organization + root org unit +
        owner user (role assignment 'owner' at the root). Returns the ids.
        """
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO organizations (name, slug) VALUES (%s, %s) RETURNING id;",
                    (name, slug),
                )
                org_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO org_units (org_id, parent_id, type, name, ancestor_ids)
                    VALUES (%s, NULL, 'org', %s, '{}') RETURNING id;
                    """,
                    (org_id, name),
                )
                root_id = cur.fetchone()["id"]
                cur.execute("UPDATE org_units SET ancestor_ids = ARRAY[%s] WHERE id = %s;", (root_id, root_id))

                cur.execute(
                    """
                    INSERT INTO users (username, api_key_hash, role, org_id, home_unit_id, active)
                    VALUES (%s, %s, 'owner', %s, %s, TRUE) RETURNING id;
                    """,
                    (owner_username, owner_key_hash, org_id, root_id),
                )
                owner_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO role_assignments (org_id, user_id, unit_id, role)
                    VALUES (%s, %s, %s, 'owner');
                    """,
                    (org_id, owner_id, root_id),
                )
            conn.commit()
        return {"org_id": org_id, "root_unit_id": root_id, "owner_user_id": owner_id}

    def get_organization(self, org_id: int) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, slug, plan, active, created_at FROM organizations WHERE id = %s;",
                            (org_id,))
                return cur.fetchone()

    def list_organizations(self) -> List[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, slug, plan, active, created_at FROM organizations ORDER BY id;")
                return cur.fetchall()

    def ensure_default_org(self) -> Dict[str, Any]:
        """Find-or-create the 'default' org + root unit. Used by dev-mode auth bypass."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM organizations WHERE slug = 'default';")
                row = cur.fetchone()
                if row:
                    org_id = row["id"]
                    cur.execute("SELECT id FROM org_units WHERE org_id = %s AND type = 'org' LIMIT 1;", (org_id,))
                    root_id = cur.fetchone()["id"]
                else:
                    cur.execute("INSERT INTO organizations (name, slug) VALUES ('Default Organization', 'default') RETURNING id;")
                    org_id = cur.fetchone()["id"]
                    cur.execute(
                        "INSERT INTO org_units (org_id, parent_id, type, name, ancestor_ids) "
                        "VALUES (%s, NULL, 'org', 'Default Organization', '{}') RETURNING id;",
                        (org_id,),
                    )
                    root_id = cur.fetchone()["id"]
                    cur.execute("UPDATE org_units SET ancestor_ids = ARRAY[%s] WHERE id = %s;", (root_id, root_id))
            conn.commit()
        return {"org_id": org_id, "root_unit_id": root_id}

    def get_org_root_unit(self, org_id: int) -> Optional[int]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM org_units WHERE org_id = %s AND type = 'org' LIMIT 1;", (org_id,))
                row = cur.fetchone()
                return row["id"] if row else None

    # -------------------------------------------------------------- org units

    def create_unit(self, org_id: int, parent_id: int, unit_type: str, name: str) -> int:
        """Creates a department/team under parent; materializes ancestor_ids = parent path + self."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ancestor_ids, org_id FROM org_units WHERE id = %s;", (parent_id,))
                parent = cur.fetchone()
                if not parent or parent["org_id"] != org_id:
                    raise ValueError("Parent unit not found in this organization")
                cur.execute(
                    """
                    INSERT INTO org_units (org_id, parent_id, type, name, ancestor_ids)
                    VALUES (%s, %s, %s, %s, '{}') RETURNING id;
                    """,
                    (org_id, parent_id, unit_type, name),
                )
                unit_id = cur.fetchone()["id"]
                new_path = list(parent["ancestor_ids"]) + [unit_id]
                cur.execute("UPDATE org_units SET ancestor_ids = %s WHERE id = %s;", (new_path, unit_id))
            conn.commit()
        return unit_id

    def get_unit(self, unit_id: int) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, org_id, parent_id, type, name, ancestor_ids FROM org_units WHERE id = %s;",
                    (unit_id,),
                )
                return cur.fetchone()

    def get_unit_tree(self, org_id: int) -> List[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, parent_id, type, name, ancestor_ids
                    FROM org_units WHERE org_id = %s ORDER BY array_length(ancestor_ids, 1), name;
                    """,
                    (org_id,),
                )
                return cur.fetchall()

    def update_unit(self, unit_id: int, org_id: int, name: str) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE org_units SET name = %s WHERE id = %s AND org_id = %s RETURNING id;",
                            (name, unit_id, org_id))
                return cur.fetchone() is not None

    def delete_unit(self, unit_id: int, org_id: int) -> bool:
        """Deletes a unit and its subtree (CASCADE). Refuses to delete the org root."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT type FROM org_units WHERE id = %s AND org_id = %s;", (unit_id, org_id))
                row = cur.fetchone()
                if not row:
                    return False
                if row["type"] == "org":
                    raise ValueError("Cannot delete the organization root unit")
                cur.execute("DELETE FROM org_units WHERE id = %s AND org_id = %s RETURNING id;", (unit_id, org_id))
                deleted = cur.fetchone() is not None
            conn.commit()
            return deleted

    def subtree_ids(self, unit_id: int) -> List[int]:
        """All nodes at or below unit_id (uses GIN on ancestor_ids)."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM org_units WHERE %s = ANY(ancestor_ids);", (unit_id,))
                return [r["id"] for r in cur.fetchall()]

    def compute_visible_node_ids(self, org_id: int, assignment_unit_ids: List[int]) -> List[int]:
        """
        Documents inherit downward and authority covers a subtree, so a role
        assignment at node N lets a user read docs attached to
        subtree(N) ∪ ancestors(N). Returns the union over all assignments.
        """
        if not assignment_unit_ids:
            return []
        self.connect()
        assign_set = set(assignment_unit_ids)
        visible: set = set()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # ancestor_ids && assign_ids => the row is in the subtree of some assignment node
                cur.execute(
                    "SELECT id, ancestor_ids FROM org_units WHERE org_id = %s AND ancestor_ids && %s::int[];",
                    (org_id, list(assign_set)),
                )
                for row in cur.fetchall():
                    visible.add(row["id"])                       # subtree member
                    if row["id"] in assign_set:
                        visible.update(row["ancestor_ids"])      # ancestors of the assignment node
        return sorted(visible)

    # -------------------------------------------------------------- role mgmt

    def add_role_assignment(self, org_id: int, user_id: int, unit_id: int, role: str) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO role_assignments (org_id, user_id, unit_id, role)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, unit_id, role) DO UPDATE SET role = EXCLUDED.role
                    RETURNING id;
                    """,
                    (org_id, user_id, unit_id, role),
                )
                rid = cur.fetchone()["id"]
            conn.commit()
            return rid

    def remove_role_assignment(self, assignment_id: int, org_id: int) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM role_assignments WHERE id = %s AND org_id = %s RETURNING id;",
                            (assignment_id, org_id))
                ok = cur.fetchone() is not None
            conn.commit()
            return ok

    def get_user_assignments(self, user_id: int) -> List[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ra.id, ra.unit_id, ra.role, ou.name AS unit_name, ou.type AS unit_type
                    FROM role_assignments ra JOIN org_units ou ON ra.unit_id = ou.id
                    WHERE ra.user_id = %s ORDER BY ra.id;
                    """,
                    (user_id,),
                )
                return cur.fetchall()

    # ------------------------------------------------------------------- users

    def create_user(self, username: str, api_key_hash: str, role: str, org_id: int,
                    home_unit_id: int) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, api_key_hash, role, org_id, home_unit_id, active)
                    VALUES (%s, %s, %s, %s, %s, TRUE) RETURNING id;
                    """,
                    (username, api_key_hash, role, org_id, home_unit_id),
                )
                return cur.fetchone()["id"]

    def get_user_by_key_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, role, org_id, home_unit_id, is_platform_admin, active
                    FROM users WHERE api_key_hash = %s;
                    """,
                    (key_hash,),
                )
                return cur.fetchone()

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, org_id, home_unit_id, is_platform_admin, active FROM users WHERE id = %s;",
                    (user_id,),
                )
                return cur.fetchone()

    def list_users(self, org_id: int) -> List[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, role, home_unit_id, active, created_at
                    FROM users WHERE org_id = %s ORDER BY id;
                    """,
                    (org_id,),
                )
                return cur.fetchall()

    def set_user_active(self, user_id: int, active: bool, org_id: int) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET active = %s WHERE id = %s AND org_id = %s RETURNING id;",
                            (active, user_id, org_id))
                return cur.fetchone() is not None

    def delete_user(self, user_id: int, org_id: int) -> bool:
        """Hard-delete a user within an org. query_audits rows are kept for compliance."""
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s AND org_id = %s RETURNING id;", (user_id, org_id))
                return cur.fetchone() is not None

    def rotate_api_key(self, user_id: int, new_key_hash: str, org_id: int) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET api_key_hash = %s WHERE id = %s AND org_id = %s RETURNING id;",
                    (new_key_hash, user_id, org_id),
                )
                return cur.fetchone() is not None

    def any_users_exist(self) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users LIMIT 1;")
                return cur.fetchone() is not None

    def platform_admin_exists(self) -> bool:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE is_platform_admin AND active LIMIT 1;")
                return cur.fetchone() is not None

    def create_platform_admin(self, username: str, api_key_hash: str) -> Dict[str, Any]:
        """
        Bootstraps the platform operator. Housed in a dedicated 'platform' org so
        the users.org_id FK is satisfied; this account provisions customer orgs.
        """
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM organizations WHERE slug = 'platform';")
                row = cur.fetchone()
                if row:
                    plat_org = row["id"]
                    cur.execute("SELECT id FROM org_units WHERE org_id = %s AND type = 'org' LIMIT 1;", (plat_org,))
                    root_id = cur.fetchone()["id"]
                else:
                    cur.execute("INSERT INTO organizations (name, slug) VALUES ('Platform', 'platform') RETURNING id;")
                    plat_org = cur.fetchone()["id"]
                    cur.execute(
                        "INSERT INTO org_units (org_id, parent_id, type, name, ancestor_ids) "
                        "VALUES (%s, NULL, 'org', 'Platform', '{}') RETURNING id;",
                        (plat_org,),
                    )
                    root_id = cur.fetchone()["id"]
                    cur.execute("UPDATE org_units SET ancestor_ids = ARRAY[%s] WHERE id = %s;", (root_id, root_id))

                cur.execute(
                    """
                    INSERT INTO users (username, api_key_hash, role, org_id, home_unit_id, is_platform_admin, active)
                    VALUES (%s, %s, 'owner', %s, %s, TRUE, TRUE) RETURNING id;
                    """,
                    (username, api_key_hash, plat_org, root_id),
                )
                uid = cur.fetchone()["id"]
            conn.commit()
        return {"user_id": uid, "org_id": plat_org, "root_unit_id": root_id}

    # --------------------------------------------------------------- documents

    def insert_document(self, metadata: Dict[str, Any]) -> int:
        self.connect()
        params = {
            "source": metadata.get("source"),
            "title": metadata.get("title"),
            "total_pages": metadata.get("total_pages"),
            "file_path": metadata.get("file_path"),
            "org_id": metadata.get("org_id"),
            "unit_id": metadata.get("unit_id"),
            "uploaded_by": metadata.get("uploaded_by"),
        }
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (source, title, total_pages, file_path, org_id, unit_id, uploaded_by)
                    VALUES (%(source)s, %(title)s, %(total_pages)s, %(file_path)s, %(org_id)s, %(unit_id)s, %(uploaded_by)s)
                    ON CONFLICT (org_id, source) DO UPDATE SET
                        title = EXCLUDED.title,
                        total_pages = EXCLUDED.total_pages,
                        file_path = EXCLUDED.file_path,
                        unit_id = EXCLUDED.unit_id,
                        uploaded_by = COALESCE(EXCLUDED.uploaded_by, documents.uploaded_by)
                    RETURNING id;
                    """,
                    params,
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("Failed to retrieve document ID after insert")
                return row["id"]

    def insert_chunks(self, document_id: int, chunks: List[Dict[str, Any]],
                      embeddings: List[List[float]], org_id: int):
        """Replaces a document's chunks atomically using COPY (bulk load)."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                cur.execute("SELECT unit_id FROM documents WHERE id = %s;", (document_id,))
                row = cur.fetchone()
                unit_id = row["unit_id"] if row else None
                with cur.copy(
                    "COPY document_chunks (document_id, org_id, unit_id, chunk_index, page_num, content, context, embedding, parent_content, parent_chunk_index) FROM STDIN"
                ) as copy:
                    for chunk, embedding in zip(chunks, embeddings):
                        copy.write_row((
                            document_id,
                            org_id,
                            unit_id,
                            chunk["chunk_index"],
                            chunk["page_num"],
                            chunk["text"],
                            chunk.get("context"),
                            _vec(embedding),
                            chunk.get("parent_content"),
                            chunk.get("parent_chunk_index"),
                        ))
            conn.commit()

    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, source, org_id, unit_id, uploaded_by FROM documents WHERE id = %s;",
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
    def _visibility_sql(org_id: int, visible_node_ids: Optional[List[int]], params: list,
                        alias: str = "d") -> List[str]:
        """
        Tenant + hierarchy visibility, enforced in the WHERE clause (never
        post-filtering). org_id always required. visible_node_ids None means
        org-wide (owner/admin at the org root) — only the org_id filter applies.
        """
        clauses = [f"{alias}.org_id = %s"]
        params.append(org_id)
        if visible_node_ids is not None:
            if not visible_node_ids:
                clauses.append("1 = 0")
            else:
                placeholders = ",".join(["%s"] * len(visible_node_ids))
                clauses.append(f"{alias}.unit_id IN ({placeholders})")
                params.extend(visible_node_ids)
        return clauses

    def get_all_documents(self, org_id: int, visible_node_ids: Optional[List[int]] = None,
                          limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        self.connect()
        params: list = []
        clauses = self._visibility_sql(org_id, visible_node_ids, params)
        where_sql = "WHERE " + " AND ".join(clauses)
        params.extend([min(limit, 500), max(offset, 0)])
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, source, title, total_pages, file_path, unit_id, created_at
                    FROM documents d {where_sql} ORDER BY id DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cur.fetchall()

    def list_sources(self, org_id: int, visible_node_ids: Optional[List[int]] = None) -> List[str]:
        self.connect()
        params: list = []
        clauses = self._visibility_sql(org_id, visible_node_ids, params)
        where_sql = "WHERE " + " AND ".join(clauses)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT source FROM documents d {where_sql};", tuple(params))
                return [row["source"] for row in cur.fetchall() if row["source"]]

    # ------------------------------------------------------------------ search

    @classmethod
    def _filter_sql(cls, org_id: int, visible_node_ids: Optional[List[int]],
                    metadata_filter: Optional[Dict[str, Any]], params: list) -> List[str]:
        # Tenant + hierarchy filter on the chunk's own columns (migration 0006),
        # so the visible set is resolvable from document_chunks_org_unit_idx alone.
        clauses = cls._visibility_sql(org_id, visible_node_ids, params, alias="dc")
        if metadata_filter:
            if "source_doc" in metadata_filter:
                clauses.append("d.source ILIKE %s")
                params.append(f"%{metadata_filter['source_doc']}%")
            if "page_num" in metadata_filter:
                clauses.append("dc.page_num = %s")
                params.append(metadata_filter["page_num"])
        return clauses

    def _supports_iterative_scan(self, cur) -> bool:
        """pgvector >= 0.8 defines hnsw.iterative_scan once its library is loaded."""
        if self._iterative_scan is None:
            cur.execute("SELECT '[1]'::vector;")  # loads the vector library in this session
            cur.execute("SELECT current_setting('hnsw.iterative_scan', true) AS v;")
            self._iterative_scan = cur.fetchone()["v"] is not None
        return self._iterative_scan

    def dense_search(self, query_vector: List[float], top_k: int, org_id: int,
                     visible_node_ids: Optional[List[int]] = None,
                     metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Permission filter lives in the WHERE clause. HNSW applies that filter
        after the graph walk, so for a narrow scope it may return fewer than
        top_k rows. Three guards keep recall up:
        1. Visible set <= exact_scan_max_rows: exact scan of only that set.
        2. pgvector >= 0.8: hnsw.iterative_scan keeps walking the graph.
        3. Older pgvector: one retry with ef_search = hnsw_ef_search_max.
        """
        self.connect()
        vec = _vec(query_vector)
        filter_params: list = []
        where_clauses = self._filter_sql(org_id, visible_node_ids, metadata_filter, filter_params)
        where_sql = "WHERE " + " AND ".join(where_clauses)
        needs_join = bool(metadata_filter and "source_doc" in metadata_filter)
        count_join = "JOIN documents d ON dc.document_id = d.id" if needs_join else ""
        limit = settings.exact_scan_max_rows

        # No ORDER BY inside the CTE, so the planner cannot pick HNSW for it
        exact_sql = f"""
            WITH scoped AS MATERIALIZED (
                SELECT dc.id, dc.chunk_index, dc.page_num,
                       COALESCE(dc.parent_content, dc.content) AS content,
                       dc.content AS child_content, dc.context, d.source AS source_doc,
                       1 - (dc.embedding <=> %s::vector) AS score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                {where_sql}
            )
            SELECT * FROM scoped ORDER BY score DESC LIMIT %s;
        """
        ann_sql = f"""
            SELECT
                dc.id,
                dc.chunk_index,
                dc.page_num,
                COALESCE(dc.parent_content, dc.content) AS content,
                dc.content AS child_content,
                dc.context,
                d.source as source_doc,
                1 - (dc.embedding <=> %s::vector) AS score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            {where_sql}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
        """
        ann_params = tuple([vec] + filter_params + [vec, top_k])

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # 1. Bounded count: reads at most limit+1 index entries
                cur.execute(
                    f"SELECT count(*) AS n FROM (SELECT 1 FROM document_chunks dc {count_join} "
                    f"{where_sql} LIMIT %s) t;",
                    tuple(filter_params + [limit + 1]),
                )
                if cur.fetchone()["n"] <= limit:
                    cur.execute(exact_sql, tuple([vec] + filter_params + [top_k]))
                    return cur.fetchall()

                cur.execute("SET LOCAL hnsw.ef_search = %s;" % max(int(settings.hnsw_ef_search), int(top_k)))
                iterative = settings.hnsw_iterative_scan and self._supports_iterative_scan(cur)
                if iterative:
                    # relaxed_order may return rows slightly out of order; re-sorted below
                    cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order;")
                    cur.execute("SET LOCAL hnsw.max_scan_tuples = %s;" % int(settings.hnsw_max_scan_tuples))
                cur.execute(ann_sql, ann_params)
                rows = cur.fetchall()

                # 3. Starved result on old pgvector: one wider graph walk
                if len(rows) < top_k and not iterative and settings.hnsw_ef_search_max > settings.hnsw_ef_search:
                    cur.execute("SET LOCAL hnsw.ef_search = %s;" % int(settings.hnsw_ef_search_max))
                    cur.execute(ann_sql, ann_params)
                    rows = cur.fetchall()
                return sorted(rows, key=lambda r: r["score"], reverse=True)

    def lexical_search(self, query_text: str, top_k: int, org_id: int,
                       visible_node_ids: Optional[List[int]] = None,
                       metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Postgres full-text leg. A quoted query keeps websearch semantics (every
        term and phrase must match). Any other query ORs its terms and ranks
        by ts_rank normalized for document length: requiring every word of a
        natural-language question matched almost nothing (BEIR SciFact
        nDCG@10 0.072 with AND, 0.606 with OR).
        """
        self.connect()
        if not query_text.strip():
            return []

        if '"' in query_text:
            tsquery = "websearch_to_tsquery('english', %s)"
        else:
            tsquery = "replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery"

        params: list = [query_text]
        where_clauses = ["dc.content_tsvector @@ q.query"]
        where_clauses += self._filter_sql(org_id, visible_node_ids, metadata_filter, params)
        where_sql = "WHERE " + " AND ".join(where_clauses)
        params.append(top_k)

        query_sql = f"""
            SELECT
                dc.id,
                dc.chunk_index,
                dc.page_num,
                COALESCE(dc.parent_content, dc.content) AS content,
                dc.content AS child_content,
                dc.context,
                d.source as source_doc,
                ts_rank(dc.content_tsvector, q.query, 1) AS score
            FROM (SELECT {tsquery} AS query) q
            CROSS JOIN document_chunks dc
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
        # access_key already encodes org_id + permission scope (see auth.compute_access_key),
        # so cache hits can never cross tenants or permission sets.
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
                              citations: List[Dict[str, Any]], org_id: int, access_key: str = "public"):
        self.connect()
        epoch = self.get_corpus_epoch()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_cache (query_text, query_embedding, answer, citations, corpus_epoch, access_key, org_id)
                    VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                    ON CONFLICT (query_text, access_key) DO UPDATE SET
                        query_embedding = EXCLUDED.query_embedding,
                        answer = EXCLUDED.answer,
                        citations = EXCLUDED.citations,
                        corpus_epoch = EXCLUDED.corpus_epoch,
                        org_id = EXCLUDED.org_id,
                        created_at = CURRENT_TIMESTAMP;
                    """,
                    (query_text, _vec(query_embedding), answer, json.dumps(citations), epoch, access_key, org_id),
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

    def enqueue_ingest_job(self, filename: str, stored_path: str, org_id: int, unit_id: int,
                           enrich: bool = False, uploaded_by: Optional[int] = None) -> int:
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingest_jobs (filename, stored_path, enrich, status, org_id, unit_id, uploaded_by)
                    VALUES (%s, %s, %s, 'queued', %s, %s, %s) RETURNING id;
                    """,
                    (filename, stored_path, enrich, org_id, unit_id, uploaded_by),
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
                    RETURNING id, filename, stored_path, enrich, org_id, unit_id, uploaded_by;
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
                           org_id, unit_id, uploaded_by, created_at, started_at, finished_at
                    FROM ingest_jobs WHERE id = %s;
                    """,
                    (job_id,),
                )
                return cur.fetchone()

    def save_query_audit(self, question: str, answer: str, score: float, faithful: bool, reason: str,
                         org_id: Optional[int] = None, user_id: Optional[int] = None):
        self.connect()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_audits (question, answer, faithfulness_score, faithful, reason, org_id, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (question, answer, score, faithful, reason, org_id, user_id),
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
