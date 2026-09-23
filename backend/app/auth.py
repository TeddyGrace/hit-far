"""Single-user auth: one password (APP_PASSWORD) -> signed, httpOnly session cookie."""

import hmac
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel

from app.config import Settings, get_settings

COOKIE_NAME = "hitfar_session"
router = APIRouter(prefix="/api/auth", tags=["auth"])


def _signer(s: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(s.secret_key, salt="session")


def require_auth(request: Request, s: Settings = Depends(get_settings)) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, "not authenticated")
    try:
        _signer(s).loads(token, max_age=s.session_max_age_s)
    except (BadSignature, SignatureExpired):
        raise HTTPException(401, "session expired") from None


class LoginIn(BaseModel):
    password: str


@router.post("/login")
def login(body: LoginIn, response: Response, s: Settings = Depends(get_settings)) -> dict:
    if not hmac.compare_digest(body.password.encode(), s.app_password.encode()):
        time.sleep(1.0)  # blunt brute-force damping; single user, so latency is fine
        raise HTTPException(401, "wrong password")
    response.set_cookie(
        COOKIE_NAME, _signer(s).dumps({"u": "owner"}), max_age=s.session_max_age_s,
        httponly=True, secure=s.cookie_secure, samesite="lax", path="/",
    )
    return {"ok": True}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me", dependencies=[Depends(require_auth)])
def me() -> dict:
    return {"ok": True}
