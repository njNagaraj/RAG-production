import hashlib
import logging
import secrets
import time
import threading
from typing import Optional, List, Dict, Any

from fastapi import Request, HTTPException
from pydantic import BaseModel

from pipeline.config import settings
from pipeline.vector_store import get_vector_store

logger = logging.getLogger("rag.auth")

# Role rank for scoped comparisons
ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2, "owner": 3}

class CurrentUser(BaseModel):
    id: Optional[int] = None
    username: str
    org_id: int
    home_unit_id: Optional[int] = None
    is_platform_admin: bool = False
    role: str = "viewer"                       # legacy primary role (display only)
    assignments: List[Dict[str, Any]] = []      # [{unit_id, role, ...}]
    # Precomputed scopes (populated in _lookup_user):
    visible_node_ids: Optional[List[int]] = None   # None = org-wide (owner/admin at root)
    write_unit_ids: List[int] = []                 # subtree union of editor+ assignments
    manage_unit_ids: List[int] = []                # subtree union of admin+ assignments
    org_wide: bool = False

    def can_write(self, unit_id: int) -> bool:
        return self.org_wide or unit_id in self.write_unit_ids

    def can_manage(self, unit_id: int) -> bool:
        return self.org_wide or unit_id in self.manage_unit_ids

    @property
    def is_org_admin(self) -> bool:
        """True if the user has admin+ authority over the whole org (root)."""
        return self.org_wide

def generate_api_key() -> str:
    return "rag_" + secrets.token_urlsafe(32)

def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

def compute_access_key(user: CurrentUser) -> str:
    """
    Semantic-cache scope. Encodes org_id AND the exact read-visibility set, so a
    cache entry can never be served across tenants or across permission scopes.
    """
    if user.org_wide:
        scope = "all"
    else:
        scope = ",".join(str(n) for n in sorted(user.visible_node_ids or []))
    basis = f"org:{user.org_id}|scope:{scope}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]

# ---- brute-force lockout -----------------------------------------------------

_failed_attempts: Dict[str, List[float]] = {}
_FAIL_WINDOW_S = 60.0
_FAIL_MAX = 10
_LOCKOUT_S = 300.0
_fail_lock = threading.Lock()

def _check_brute_force(ip: str):
    now = time.time()
    with _fail_lock:
        attempts = [t for t in _failed_attempts.get(ip, []) if now - t < _FAIL_WINDOW_S]
        _failed_attempts[ip] = attempts
        if len(attempts) >= _FAIL_MAX:
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed auth attempts from this IP. Try again in {int(_LOCKOUT_S // 60)} minutes.",
            )

def _record_failure(ip: str):
    now = time.time()
    with _fail_lock:
        bucket = _failed_attempts.setdefault(ip, [])
        bucket.append(now)
        _failed_attempts[ip] = [t for t in bucket if now - t < _LOCKOUT_S]

# ---- request-level user cache -------------------------------------------------

_CACHE_TTL_S = 30.0
_user_cache: Dict[str, Any] = {}   # key_hash -> (expires_at, CurrentUser | None)
_cache_lock = threading.Lock()
_auth_disabled_logged = False

def _build_user(row: Dict[str, Any]) -> CurrentUser:
    """Loads assignments and expands the hierarchy into read/write/manage scopes."""
    store = get_vector_store()
    org_id = row["org_id"]
    assignments = store.get_user_assignments(row["id"])
    root_unit = store.get_org_root_unit(org_id)

    # org-wide iff an admin+ assignment sits at the org root
    org_wide = any(
        a["unit_id"] == root_unit and ROLE_RANK.get(a["role"], 0) >= ROLE_RANK["admin"]
        for a in assignments
    )

    if org_wide:
        visible_node_ids = None
    else:
        visible_node_ids = store.compute_visible_node_ids(org_id, [a["unit_id"] for a in assignments])

    # write/manage scopes = subtree unions of editor+/admin+ assignments
    write_units: set = set()
    manage_units: set = set()
    for a in assignments:
        rank = ROLE_RANK.get(a["role"], 0)
        if rank >= ROLE_RANK["editor"]:
            write_units.update(store.subtree_ids(a["unit_id"]))
        if rank >= ROLE_RANK["admin"]:
            manage_units.update(store.subtree_ids(a["unit_id"]))

    primary = max((a["role"] for a in assignments), key=lambda r: ROLE_RANK.get(r, 0), default=row.get("role", "viewer"))
    return CurrentUser(
        id=row["id"], username=row["username"], org_id=org_id,
        home_unit_id=row.get("home_unit_id"), is_platform_admin=row.get("is_platform_admin", False),
        role=primary, assignments=assignments,
        visible_node_ids=visible_node_ids, write_unit_ids=sorted(write_units),
        manage_unit_ids=sorted(manage_units), org_wide=org_wide,
    )

def _lookup_user(key_hash: str) -> Optional[CurrentUser]:
    now = time.time()
    with _cache_lock:
        hit = _user_cache.get(key_hash)
        if hit and hit[0] > now:
            return hit[1]
    row = get_vector_store().get_user_by_key_hash(key_hash)
    user = None
    if row and row["active"]:
        user = _build_user(row)
    with _cache_lock:
        _user_cache[key_hash] = (now + _CACHE_TTL_S, user)
    return user

def invalidate_user_cache():
    """Called after user/role changes so updates apply immediately."""
    with _cache_lock:
        _user_cache.clear()

def auth_enabled() -> bool:
    if settings.platform_bootstrap_key:
        return True
    try:
        return get_vector_store().any_users_exist()
    except Exception:
        return True   # DB down: fail closed for authenticated routes

def _dev_user() -> CurrentUser:
    """Synthetic org-wide owner for dev mode (auth disabled). Uses the default org."""
    ids = get_vector_store().ensure_default_org()
    return CurrentUser(
        id=None, username="dev", org_id=ids["org_id"], home_unit_id=ids["root_unit_id"],
        is_platform_admin=True, role="owner", assignments=[],
        visible_node_ids=None, write_unit_ids=[], manage_unit_ids=[], org_wide=True,
    )

def get_current_user(request: Request) -> CurrentUser:
    global _auth_disabled_logged
    if not auth_enabled():
        if not _auth_disabled_logged:
            logger.warning("AUTH DISABLED: no users exist and PLATFORM_BOOTSTRAP_KEY is unset. "
                           "Development mode only — set PLATFORM_BOOTSTRAP_KEY in production.")
            _auth_disabled_logged = True
        return _dev_user()

    ip = request.client.host if request.client else "unknown"
    _check_brute_force(ip)

    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    user = _lookup_user(hash_key(api_key))
    if user is None:
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return user

# ---- capability dependencies -------------------------------------------------

def require_platform_admin(request: Request) -> CurrentUser:
    user = get_current_user(request)
    if not user.is_platform_admin:
        raise HTTPException(status_code=403, detail="Requires platform administrator")
    return user

def require_org_admin(request: Request) -> CurrentUser:
    """Org-wide admin/owner (authority at the org root)."""
    user = get_current_user(request)
    if not user.is_org_admin:
        raise HTTPException(status_code=403, detail="Requires organization administrator")
    return user

def require_role(*roles: str):
    """
    Back-compat shim from the flat model. Maps the old flat roles onto the new
    scoped model: 'admin' => org admin; 'editor' => has write scope somewhere.
    Prefer the scoped helpers (require_org_admin / can_write / can_manage).
    """
    def checker(request: Request) -> CurrentUser:
        user = get_current_user(request)
        if "admin" in roles and user.is_org_admin:
            return user
        if "editor" in roles and (user.is_org_admin or user.write_unit_ids):
            return user
        if "viewer" in roles:
            return user
        raise HTTPException(status_code=403, detail=f"Requires role: {' or '.join(roles)}")
    return checker

def bootstrap_platform_admin():
    """Creates the platform admin from PLATFORM_BOOTSTRAP_KEY if none exists yet."""
    if not settings.platform_bootstrap_key:
        return
    store = get_vector_store()
    if store.platform_admin_exists():
        return
    store.create_platform_admin("platform-admin", hash_key(settings.platform_bootstrap_key))
    logger.info("Bootstrapped platform admin from PLATFORM_BOOTSTRAP_KEY.")
