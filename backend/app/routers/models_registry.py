from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.db import get_db
from app.models import Model
from app.schemas import ModelOut

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_auth)])


@router.get("", response_model=list[ModelOut])
def list_models(db: Session = Depends(get_db)):
    return db.scalars(select(Model).order_by(Model.task, Model.created_at.desc())).all()
