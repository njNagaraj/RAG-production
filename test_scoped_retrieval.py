"""
Permission-scoped dense search against a live Postgres + pgvector.
A small team's 30 chunks sit inside an org of 3,000 chunks, and the query
vector points at the large department. Plain HNSW + post-filter starves the
team; the scoped search must still return a full top_k from the team only.
"""
import uuid
import numpy as np
import pytest

from pipeline.config import settings
from pipeline.vector_store import get_vector_store

DIM = 384
BIG, SMALL, TOP_K = 3000, 30, 20

@pytest.fixture(scope="module")
def corpus():
    store = get_vector_store()
    try:
        store.initialize_db(dimension=DIM)
    except Exception as e:
        pytest.skip(f"Postgres unavailable: {e}")

    tag = uuid.uuid4().hex[:8]
    org = store.create_organization(f"Scope {tag}", f"scope-{tag}", f"owner-{tag}", f"hash-{tag}")
    org_id, root = org["org_id"], org["root_unit_id"]
    dept = store.create_unit(org_id, root, "department", "Big Dept")
    team = store.create_unit(org_id, root, "team", "Small Team")

    rng = np.random.default_rng(7)
    center = rng.normal(size=DIM)

    def add(unit_id, n, spread, name):
        doc_id = store.insert_document({"source": f"{name}-{tag}.md", "title": name, "total_pages": 1,
                                        "file_path": name, "org_id": org_id, "unit_id": unit_id})
        vecs = [(center + rng.normal(scale=spread, size=DIM)).tolist() for _ in range(n)]
        chunks = [{"chunk_index": i, "page_num": 1, "text": f"{name} chunk {i}"} for i in range(n)]
        store.insert_chunks(doc_id, chunks, vecs, org_id=org_id)
        return doc_id

    add(dept, BIG, 0.3, "dept")     # tight cluster around the query
    team_doc = add(team, SMALL, 3.0, "team")  # far from the query
    return {"store": store, "org_id": org_id, "dept": dept, "team": team,
            "team_doc": team_doc, "query": center.tolist()}

def _team_ids(c):
    with c["store"].pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM document_chunks WHERE document_id = %s;", (c["team_doc"],))
        return {r["id"] for r in cur.fetchall()}

def test_chunks_carry_unit_id(corpus):
    with corpus["store"].pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM document_chunks WHERE document_id = %s AND unit_id = %s;",
                    (corpus["team_doc"], corpus["team"]))
        assert cur.fetchone()["n"] == SMALL

def test_exact_scan_fills_top_k_for_narrow_scope(corpus):
    rows = corpus["store"].dense_search(corpus["query"], TOP_K, corpus["org_id"],
                                        visible_node_ids=[corpus["team"]])
    assert len(rows) == TOP_K
    assert {r["id"] for r in rows} <= _team_ids(corpus)
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)

def test_ann_path_stays_in_scope(corpus, monkeypatch):
    # Force the HNSW path: visible set (30) is above the exact-scan limit
    monkeypatch.setattr(settings, "exact_scan_max_rows", 10)
    rows = corpus["store"].dense_search(corpus["query"], TOP_K, corpus["org_id"],
                                        visible_node_ids=[corpus["team"]])
    assert {r["id"] for r in rows} <= _team_ids(corpus)

def test_empty_scope_returns_nothing(corpus):
    assert corpus["store"].dense_search(corpus["query"], TOP_K, corpus["org_id"], visible_node_ids=[]) == []

def test_document_move_updates_chunk_unit(corpus):
    store = corpus["store"]
    with store.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE documents SET unit_id = %s WHERE id = %s;", (corpus["dept"], corpus["team_doc"]))
        cur.execute("SELECT DISTINCT unit_id FROM document_chunks WHERE document_id = %s;", (corpus["team_doc"],))
        assert [r["unit_id"] for r in cur.fetchall()] == [corpus["dept"]]
        conn.rollback()
