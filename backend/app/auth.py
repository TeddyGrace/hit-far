"""Multi-user auth: username + password (scrypt-hashed) -> signed, httpOnly session cookie.

There is no sign-up: an admin adds users on the web app's Users page (`app/routers/users.py`) or
with `python -m app.users` from a shell. The first user (the owner) is an admin.
"""

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models import Dataset, Model, RecordingSession, User

log = logging.getLogger(__name__)

COOKIE_NAME = "hitfar_session"
router = APIRouter(prefix="/api/auth", tags=["auth"])

# scrypt at n=2^14, r=8 (16 MiB, ~50 ms): stdlib only, no extra dependency.
_N, _R, _P = 2**14, 8, 1

MIN_PASSWORD = 8
_USERNAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,49}")


def normalize_username(name: str) -> str:
    """Usernames are case-insensitive: stored lower-case and trimmed."""
    return name.strip().lower()


def username_problem(name: str) -> str | None:
    """Why a (normalized) username can't be used, or None if it's fine."""
    if not _USERNAME_RE.fullmatch(name):
        return "username must be 1-50 characters: letters, digits, '.', '_' or '-'"
    return None


def password_problem(password: str) -> str | None:
    if len(password) < MIN_PASSWORD:
        return f"password must be at least {MIN_PASSWORD} characters"
    return None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${_N}${_R}${_P}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, dk = stored.split("$")
        if algo != "scrypt":
            return False
        want = base64.b64decode(dk)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(want))
    except ValueError:
        return False
    return hmac.compare_digest(got, want)


_DUMMY_HASH = hash_password(secrets.token_hex(16))  # equalises timing for unknown usernames


def _signer(s: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(s.secret_key, salt="session")


def current_user(request: Request, db: Session = Depends(get_db), s: Settings = Depends(get_settings)) -> User:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, "not authenticated")
    try:
        data = _signer(s).loads(token, max_age=s.session_max_age_s)
    except (BadSignature, SignatureExpired):
        raise HTTPException(401, "session expired") from None
    try:
        user = db.get(User, uuid.UUID(data["uid"])) if isinstance(data, dict) and "uid" in data else None
    except ValueError:
        user = None
    if user is None:  # a pre-multi-user cookie, or the user was deleted
        raise HTTPException(401, "session expired")
    return user


# Router-level guard; FastAPI caches it per request, so endpoints can also take `current_user`.
require_auth = current_user


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "only an admin can manage users")
    return user


def bootstrap_users(db: Session, s: Settings) -> None:
    """On API start: with no users yet, create the owner (an admin) from APP_PASSWORD so the existing
    login keeps working; make sure someone is an admin; then give any rows from before users existed
    to the first user."""
    first = db.scalar(select(User).order_by(User.created_at).limit(1))
    if first is None:
        first = User(username=normalize_username(s.owner_username), password_hash=hash_password(s.app_password),
                     is_admin=True)
        db.add(first)
        db.flush()
        log.info("created user %r from APP_PASSWORD", first.username)
    elif db.scalar(select(User.id).where(User.is_admin).limit(1)) is None:
        first.is_admin = True  # nobody could manage users otherwise
    db.execute(update(RecordingSession).where(RecordingSession.user_id.is_(None)).values(user_id=first.id))
    db.execute(update(Dataset).where(Dataset.user_id.is_(None), Dataset.task == "outcome").values(user_id=first.id))
    db.execute(update(Model).where(Model.user_id.is_(None), Model.task.startswith("outcome_"))
               .values(user_id=first.id))
    db.commit()


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginIn, response: Response, db: Session = Depends(get_db),
          s: Settings = Depends(get_settings)) -> dict:
    user = db.scalar(select(User).where(User.username == normalize_username(body.username)))
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok:
        time.sleep(1.0)  # blunt brute-force damping; a handful of users, so latency is fine
        raise HTTPException(401, "wrong username or password")
    response.set_cookie(
        COOKIE_NAME, _signer(s).dumps({"uid": str(user.id)}), max_age=s.session_max_age_s,
        httponly=True, secure=s.cookie_secure, samesite="lax", path="/",
    )
    return {"ok": True, "username": user.username, "is_admin": user.is_admin}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"ok": True, "username": user.username, "is_admin": user.is_admin}
