"""
RBAC end-to-end test. Requires the API running with ADMIN_BOOTSTRAP_KEY set,
Postgres up, and the worker running (for the ingest isolation test).

    ADMIN_BOOTSTRAP_KEY=devadmin uvicorn app:app --port 8000
    python worker.py
    python test_rbac.py [base_url] [admin_key]
"""
import io
import sys
import time
import urllib.request
import urllib.error
import json

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
ADMIN_KEY = sys.argv[2] if len(sys.argv) > 2 else "devadmin"

PASS = FAIL = 0

def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

def req(method: str, path: str, key: str = None, body: dict = None, form_file: tuple = None):
    """Returns (status, parsed_json_or_None)."""
    url = BASE + path
    headers = {}
    if key:
        headers["X-API-Key"] = key
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif form_file is not None:
        filename, content, extra = form_file
        boundary = "----ragrbactest"
        parts = []
        for k, v in extra.items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: text/plain\r\n\r\n{content}\r\n--{boundary}--\r\n"
        )
        data = "".join(parts).encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "null")
        except Exception:
            return e.code, None

def wait_job(job_id: int, key: str, timeout_s: int = 120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, job = req("GET", f"/api/jobs/{job_id}", key)
        if status == 200 and job["status"] in ("done", "failed"):
            return job
        time.sleep(2)
    return {"status": "timeout"}

def main():
    ts = int(time.time())
    print("=" * 60)
    print("RBAC END-TO-END TEST")
    print("=" * 60)

    # --- auth basics
    print("[*] Auth basics")
    status, me = req("GET", "/api/me", ADMIN_KEY)
    check("admin /api/me", status == 200 and me and me["role"] == "admin", f"got {status} {me}")
    status, _ = req("GET", "/api/documents", "wrong-key-123")
    check("wrong key -> 401", status == 401, f"got {status}")
    status, _ = req("POST", "/api/query", None, {"question": "hi"})
    check("no key -> 401", status == 401, f"got {status}")

    # --- create users
    print("[*] User management")
    status, fin = req("POST", "/api/users", ADMIN_KEY,
                      {"username": f"fin_user_{ts}", "role": "editor", "groups": ["finance"]})
    check("create finance editor", status == 201 and fin.get("api_key"), f"got {status}")
    status, viewer = req("POST", "/api/users", ADMIN_KEY,
                         {"username": f"viewer_{ts}", "role": "viewer", "groups": []})
    check("create viewer", status == 201 and viewer.get("api_key"), f"got {status}")
    fin_key, viewer_key = fin["api_key"], viewer["api_key"]

    status, _ = req("GET", "/api/users", viewer_key)
    check("viewer list users -> 403", status == 403, f"got {status}")

    # --- role permissions on ingest
    print("[*] Role permissions")
    doc_text = f"CONFIDENTIAL SALARY REPORT {ts}. The zorbital bonus for Q3 is $9,876 for the finance team."
    status, _ = req("POST", "/api/ingest", viewer_key,
                    form_file=(f"salary_{ts}.txt", doc_text, {"enrich": "false", "groups": "finance"}))
    check("viewer ingest -> 403", status == 403, f"got {status}")

    status, job = req("POST", "/api/ingest", fin_key,
                      form_file=(f"salary_{ts}.txt", doc_text, {"enrich": "false", "groups": "finance"}))
    check("editor ingest -> 202", status == 202 and "job_id" in job, f"got {status} {job}")

    final = wait_job(job["job_id"], fin_key)
    check("ingest job done (worker running?)", final["status"] == "done", f"got {final}")

    # --- document listing visibility
    print("[*] Document visibility")
    status, docs_fin = req("GET", "/api/documents", fin_key)
    fin_sees = any(f"salary_{ts}" in d["source"] for d in (docs_fin or []))
    check("finance user sees finance doc in list", status == 200 and fin_sees)

    status, docs_view = req("GET", "/api/documents", viewer_key)
    viewer_sees = any(f"salary_{ts}" in d["source"] for d in (docs_view or []))
    check("viewer does NOT see finance doc in list", status == 200 and not viewer_sees)

    # --- retrieval isolation (the core RAG-RBAC property)
    print("[*] Retrieval isolation")
    question = f"What is the zorbital bonus for Q3?"
    status, ans_fin = req("POST", "/api/query", fin_key, {"question": question})
    fin_cited = status == 200 and any(f"salary_{ts}" in c["source"] for c in ans_fin.get("citations", []))
    check("finance user retrieves finance doc", fin_cited,
          f"citations={[c['source'] for c in (ans_fin or {}).get('citations', [])]}")

    status, ans_view = req("POST", "/api/query", viewer_key, {"question": question})
    view_cited = any(f"salary_{ts}" in c["source"] for c in (ans_view or {}).get("citations", []))
    leaked = "9,876" in (ans_view or {}).get("answer", "")
    check("viewer CANNOT retrieve finance doc", status == 200 and not view_cited and not leaked,
          f"citations={[c['source'] for c in (ans_view or {}).get('citations', [])]}")

    # --- cache isolation: viewer's identical question must not hit finance user's cache
    print("[*] Cache isolation")
    cached_flag = (ans_view or {}).get("cached", False)
    check("viewer identical query was NOT served from finance cache", not cached_flag)

    status, ans_fin2 = req("POST", "/api/query", fin_key, {"question": question})
    check("finance repeat query hits own cache", status == 200 and ans_fin2.get("cached") is True,
          f"cached={ans_fin2.get('cached')}")

    # --- deactivation
    print("[*] Deactivation")
    status, _ = req("PATCH", f"/api/users/{viewer['id']}", ADMIN_KEY, {"active": False})
    check("deactivate viewer", status == 200)
    time.sleep(31)  # auth cache TTL is 30s
    status, _ = req("GET", "/api/me", viewer_key)
    check("deactivated key -> 401", status == 401, f"got {status}")

    print("=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
