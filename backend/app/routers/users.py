"""User management for admins (the web app's Users page). Same rules as `python -m app.users`."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import hash_password, normalize_username, password_problem, require_admin, username_problem
from app.db import get_db
from app.models import RecordingSession, User

router = APIRouter(prefix="/api/users", tags=["users"], dependencies=[Depends(require_admin)])


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    is_admin: bool
    created_at: datetime
    num_sessions: int


class UserCreate(BaseModel):
    username: str
    password: str
    is_admin: bool = False


class UserPatch(BaseModel):
    password: str | None = None
    is_admin: bool | None = None


def _sessions_owned(db: Session, user_id: uuid.UUID) -> int:
    return db.scalar(select(func.count()).select_from(RecordingSession).where(RecordingSession.user_id == user_id))


def _out(db: Session, u: User) -> UserOut:
    return UserOut(id=u.id, username=u.username, is_admin=u.is_admin, created_at=u.created_at,
                   num_sessions=_sessions_owned(db, u.id))


def _get_or_404(db: Session, user_id: uuid.UUID) -> User:
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "user not found")
    return u


def _check_password(pw: str) -> None:
    if problem := password_problem(pw):
        raise HTTPException(422, problem)


@router.get("", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)) -> list[UserOut]:
    return [_out(db, u) for u in db.scalars(select(User).order_by(User.created_at))]


@router.post("", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db)) -> UserOut:
    name = normalize_username(body.username)
    if problem := username_problem(name):
        raise HTTPException(422, problem)
    _check_password(body.password)
    if db.scalar(select(User.id).where(User.username == name)) is not None:
        raise HTTPException(409, f"user {name!r} already exists")
    u = User(username=name, password_hash=hash_password(body.password), is_admin=body.is_admin)
    db.add(u)
    db.commit()
    return _out(db, u)


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: uuid.UUID, body: UserPatch, db: Session = Depends(get_db),
                me: User = Depends(require_admin)) -> UserOut:
    u = _get_or_404(db, user_id)
    if body.password is not None:
        _check_password(body.password)
        u.password_hash = hash_password(body.password)
    if body.is_admin is not None:
        if u.id == me.id and not body.is_admin:
            raise HTTPException(409, "you can't remove your own admin rights")
        u.is_admin = body.is_admin
    db.commit()
    return _out(db, u)


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: uuid.UUID, db: Session = Depends(get_db), me: User = Depends(require_admin)) -> None:
    u = _get_or_404(db, user_id)
    if u.id == me.id:
        raise HTTPException(409, "you can't delete yourself")
    if owned := _sessions_owned(db, u.id):
        raise HTTPException(409, f"{u.username!r} still owns {owned} sessions; delete them first")
    db.delete(u)
    db.commit()
