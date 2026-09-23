"""Multi-user auth: username + password (scrypt-hashed) -> signed, httpOnly session cookie.

Users are managed from the command line (`python -m app.users`); there is no sign-up endpoint.
"""

import base64
import hashlib
import hmac
import logging
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


def bootstrap_users(db: Session, s: Settings) -> None:
    """On API start: with no users yet, create the owner from APP_PASSWORD (so the existing login
    keeps working), then give any rows from before users existed to the first user."""
    first = db.scalar(select(User).order_by(User.created_at).limit(1))
    if first is None:
        first = User(username=s.owner_username.strip().lower(), password_hash=hash_password(s.app_password))
        db.add(first)
        db.flush()
        log.info("created user %r from APP_PASSWORD", first.username)
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
    user = db.scalar(select(User).where(User.username == body.username.strip().lower()))
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok:
        time.sleep(1.0)  # blunt brute-force damping; a handful of users, so latency is fine
        raise HTTPException(401, "wrong username or password")
    response.set_cookie(
        COOKIE_NAME, _signer(s).dumps({"uid": str(user.id)}), max_age=s.session_max_age_s,
        httponly=True, secure=s.cookie_secure, samesite="lax", path="/",
    )
    return {"ok": True, "username": user.username}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"ok": True, "username": user.username}
