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

class CurrentUser(BaseModel):
    id: Optional[int] = None
    username: str
    role: str                      # admin | editor | viewer
    groups: List[str] = []

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def retrieval_groups(self) -> Optional[List[str]]:
        """None means unrestricted (admin); a list restricts document visibility."""
        return None if self.is_admin else self.groups

# Dev-mode pseudo-user when auth is disabled (no users and no bootstrap key)
_DEV_USER = CurrentUser(id=None, username="dev", role="admin", groups=[])

def generate_api_key() -> str:
    return "rag_" + secrets.token_urlsafe(32)

def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

def compute_access_key(user: CurrentUser) -> str:
    """
    Semantic-cache scope. All admins share one all-access scope; everyone else
    is scoped by their exact group set (sorted, so order never splits scopes).
    """
    if user.is_admin:
        basis = "role:admin"
    else:
        basis = "groups:" + ",".join(sorted(set(user.groups)))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]

# --- request authentication ---------------------------------------------------

_CACHE_TTL_S = 30.0
_user_cache: Dict[str, Any] = {}   # key_hash -> (expires_at, CurrentUser | None)
_cache_lock = threading.Lock()
_auth_disabled_logged = False

def _lookup_user(key_hash: str) -> Optional[CurrentUser]:
    now = time.time()
    with _cache_lock:
        hit = _user_cache.get(key_hash)
        if hit and hit[0] > now:
            return hit[1]
    row = get_vector_store().get_user_by_key_hash(key_hash)
    user = None
    if row and row["active"]:
        user = CurrentUser(id=row["id"], username=row["username"],
                           role=row["role"], groups=list(row["groups"] or []))
    with _cache_lock:
        _user_cache[key_hash] = (now + _CACHE_TTL_S, user)
    return user

def invalidate_user_cache():
    """Called after user create/update so key changes apply immediately."""
    with _cache_lock:
        _user_cache.clear()

def auth_enabled() -> bool:
    """Auth is on when any user exists or a bootstrap key is configured."""
    if settings.admin_bootstrap_key:
        return True
    try:
        return get_vector_store().any_users_exist()
    except Exception:
        # DB down: fail closed for authenticated routes; /ready reports separately
        return True

def get_current_user(request: Request) -> CurrentUser:
    global _auth_disabled_logged
    if not auth_enabled():
        if not _auth_disabled_logged:
            logger.warning("AUTH DISABLED: no users exist and ADMIN_BOOTSTRAP_KEY is unset. "
                           "Development mode only — set ADMIN_BOOTSTRAP_KEY in production.")
            _auth_disabled_logged = True
        return _DEV_USER

    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    user = _lookup_user(hash_key(api_key))
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return user

def require_role(*roles: str):
    """Dependency factory: require_role('admin', 'editor')."""
    def checker(request: Request) -> CurrentUser:
        user = get_current_user(request)
        if user.role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {' or '.join(roles)}")
        return user
    return checker

def bootstrap_admin():
    """Creates the initial admin from ADMIN_BOOTSTRAP_KEY if no admin exists yet."""
    if not settings.admin_bootstrap_key:
        return
    store = get_vector_store()
    if store.admin_exists():
        return
    store.create_user("admin", hash_key(settings.admin_bootstrap_key), "admin", [])
    logger.info("Bootstrapped initial 'admin' user from ADMIN_BOOTSTRAP_KEY.")
