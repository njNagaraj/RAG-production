"""
End-to-end multi-tenant RBAC test against a running API.
Usage: python test_tenancy.py [base_url] [platform_key]
Exercises: org provisioning, hierarchy, scoped roles, document inheritance,
tenant isolation, cache isolation, write-scope enforcement.
"""
import sys
import time
import json
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PLATFORM_KEY = sys.argv[2] if len(sys.argv) > 2 else "platadmin"

_passed = 0
_failed = 0

def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}")

def call(method, path, key=None, body=None, files=None, expect=None):
    url = BASE + path
    headers = {}
    if key:
        headers["X-API-Key"] = key
    data = None
    if files:
        boundary = "----tenancyboundary"
        parts = []
        for fname, content in files.items():
            parts.append(f"--{boundary}")
            parts.append(f'Content-Disposition: form-data; name="file"; filename="{fname}"')
            parts.append("Content-Type: text/plain")
            parts.append("")
            parts.append(content)
        for k, v in (body or {}).items():
            parts.append(f"--{boundary}")
            parts.append(f'Content-Disposition: form-data; name="{k}"')
            parts.append("")
            parts.append(str(v))
        parts.append(f"--{boundary}--")
        data = "\r\n".join(parts).encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode()}

def wait_job(key, job_id, timeout=90):
    for _ in range(timeout):
        _, job = call("GET", f"/api/jobs/{job_id}", key=key)
        if job.get("status") in ("done", "failed"):
            return job
        time.sleep(1)
    return {"status": "timeout"}

def main():
    stamp = str(int(time.time()))
    acme_slug = f"acme-{stamp}"
    globex_slug = f"globex-{stamp}"

    print("\n== Provisioning ==")
    st, acme = call("POST", "/api/orgs", key=PLATFORM_KEY,
                    body={"name": "Acme", "slug": acme_slug, "owner_username": f"acme_owner_{stamp}"})
    check("create Acme org (201)", st == 201)
    acme_key = acme.get("owner_api_key")
    acme_root = acme.get("root_unit_id")

    st, globex = call("POST", "/api/orgs", key=PLATFORM_KEY,
                      body={"name": "Globex", "slug": globex_slug, "owner_username": f"globex_owner_{stamp}"})
    check("create Globex org (201)", st == 201)
    globex_key = globex.get("owner_api_key")

    st, _ = call("POST", "/api/orgs", key="wrongkey",
                 body={"name": "X", "slug": "x", "owner_username": "x"})
    check("non-platform cannot create org (401)", st == 401)
    st, _ = call("POST", "/api/orgs", key=acme_key,
                 body={"name": "X", "slug": "x", "owner_username": "x"})
    check("org owner is not platform admin (403)", st == 403)

    print("\n== Hierarchy ==")
    st, fin = call("POST", "/api/units", key=acme_key,
                   body={"parent_id": acme_root, "type": "department", "name": "Finance"})
    check("create Finance dept (201)", st == 201)
    fin_id = fin.get("id")
    st, teamA = call("POST", "/api/units", key=acme_key,
                     body={"parent_id": fin_id, "type": "team", "name": "Team A"})
    team_id = teamA.get("id")
    st, hr = call("POST", "/api/units", key=acme_key,
                  body={"parent_id": acme_root, "type": "department", "name": "HR"})
    hr_id = hr.get("id")
    check("build Finance>TeamA and HR", bool(team_id and hr_id))

    print("\n== Users & scoped roles ==")
    st, fe = call("POST", "/api/users", key=acme_key,
                  body={"username": f"fin_editor_{stamp}", "role": "editor", "unit_id": fin_id})
    fe_key = fe.get("api_key")
    check("create Finance editor (201)", st == 201)
    st, tav = call("POST", "/api/users", key=acme_key,
                   body={"username": f"teamA_viewer_{stamp}", "role": "viewer", "unit_id": team_id})
    tav_key = tav.get("api_key")
    st, hrv = call("POST", "/api/users", key=acme_key,
                   body={"username": f"hr_viewer_{stamp}", "role": "viewer", "unit_id": hr_id})
    hrv_key = hrv.get("api_key")
    check("create Team-A viewer + HR viewer", bool(tav_key and hrv_key))

    # Viewer cannot create users / units (valid payload so we hit authz, not validation)
    st, _ = call("POST", "/api/users", key=tav_key,
                 body={"username": f"nope_{stamp}", "role": "viewer"})
    check("viewer cannot create users (403)", st == 403)

    print("\n== Write-scope enforcement ==")
    st, _ = call("POST", "/api/ingest", key=fe_key,
                 files={"salary.txt": "Q3 bonus pool is 50000 dollars for Finance."},
                 body={"unit_id": hr_id})
    check("Finance editor cannot upload to HR (403)", st == 403)

    st, job = call("POST", "/api/ingest", key=fe_key,
                   files={"finance_policy.txt": "The Q3 bonus pool is 50000 dollars, approved by the CFO."},
                   body={"unit_id": fin_id})
    check("Finance editor uploads to Finance (202)", st == 202)
    j = wait_job(fe_key, job.get("job_id"))
    check("ingest job completes", j.get("status") == "done")

    print("\n== Document inheritance & isolation ==")
    time.sleep(2)
    q = {"question": "What is the Q3 bonus pool?"}
    _, r_tav = call("POST", "/api/query", key=tav_key, body=q)
    check("Team-A viewer retrieves inherited Finance doc",
          any("finance_policy" in c.get("source", "") for c in r_tav.get("citations", [])))

    _, r_hrv = call("POST", "/api/query", key=hrv_key, body=q)
    check("HR viewer does NOT see Finance doc (no citations)",
          not any("finance_policy" in c.get("source", "") for c in r_hrv.get("citations", [])))

    _, r_globex = call("POST", "/api/query", key=globex_key, body=q)
    check("Globex tenant does NOT see Acme doc",
          not any("finance_policy" in c.get("source", "") for c in r_globex.get("citations", [])))

    print("\n== Document listing scope ==")
    _, docs_owner = call("GET", "/api/documents", key=acme_key)
    _, docs_hrv = call("GET", "/api/documents", key=hrv_key)
    _, docs_globex = call("GET", "/api/documents", key=globex_key)
    check("Acme owner sees the doc", any("finance_policy" in d.get("source", "") for d in docs_owner))
    check("HR viewer's document list excludes it", not any("finance_policy" in d.get("source", "") for d in docs_hrv))
    check("Globex sees zero Acme docs", len(docs_globex) == 0)

    print("\n== Cache isolation ==")
    # Team-A viewer asks again -> should be cache hit; HR viewer same question -> NOT that cache entry
    _, r_tav2 = call("POST", "/api/query", key=tav_key, body=q)
    check("Team-A viewer gets cache hit on repeat", r_tav2.get("cached") is True)
    _, r_hrv2 = call("POST", "/api/query", key=hrv_key, body=q)
    check("HR viewer never served Team-A's cached answer",
          not (r_hrv2.get("cached") and "50000" in r_hrv2.get("answer", "")))

    print("\n== Auth basics ==")
    st, _ = call("GET", "/api/documents", key=None)
    check("no key -> 401", st == 401)
    st, _ = call("GET", "/api/users", key=hrv_key)
    check("non-admin cannot list users (403)", st == 403)

    print("\n" + "=" * 50)
    print(f"RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 50)
    return _failed == 0

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
