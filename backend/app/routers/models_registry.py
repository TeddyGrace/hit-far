from fastapi import APIRouter, Depends
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.auth import current_user, require_auth
from app.db import get_db
from app.models import Model, User
from app.schemas import ModelOut

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_auth)])


def visible_models(db: Session, user: User) -> list[Model]:
    """Shared models plus this user's own (outcome) models."""
    return list(db.scalars(select(Model).where(or_(Model.user_id.is_(None), Model.user_id == user.id))
                           .order_by(Model.task, Model.created_at.desc())).all())


@router.get("", response_model=list[ModelOut])
def list_models(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return visible_models(db, user)
