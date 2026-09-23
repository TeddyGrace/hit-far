import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.auth import current_user, require_auth
from app.db import get_db
from app.models import Job, JobStatus, Label, Model, ModelStatus, Swing, User
from app.pipeline.run import latest_pose_sequence
from app.routers.models_registry import visible_models
from app.routers.swings import get_swing_or_404, swing_detail
from app.schemas import JobOut, ModelOut, SwingDetail
from app.training.train_events import REVIEW_LABEL_TASK, reviewed_swing_ids

router = APIRouter(prefix="/api", tags=["training"], dependencies=[Depends(require_auth)])


class TrainIn(BaseModel):
    epochs: int = 40
    golfdb: bool = True
    max_golfdb_clips: int | None = None


class TrainingJobOut(JobOut):
    payload: dict


class ReviewIn(BaseModel):
    reviewed: bool


def _active_job(db: Session, type_: str) -> Job | None:
    return db.scalar(select(Job).where(Job.type == type_, Job.status.in_([JobStatus.queued, JobStatus.running]))
                     .limit(1))


@router.post("/training/events", response_model=TrainingJobOut, status_code=201)
def start_event_training(body: TrainIn, db: Session = Depends(get_db)):
    if _active_job(db, jobs.JOB_TRAIN_EVENTS):
        raise HTTPException(409, "an event-model training run is already queued or running")
    if not 1 <= body.epochs <= 200:
        raise HTTPException(400, "epochs must be between 1 and 200")
    job = jobs.enqueue(db, jobs.JOB_TRAIN_EVENTS, {"config": body.model_dump(exclude_none=True)})
    job.max_attempts = 2  # a failed multi-hour run shouldn't loop; pose extraction is cached anyway
    db.commit()
    return job


@router.get("/training/jobs", response_model=list[TrainingJobOut])
def training_jobs(db: Session = Depends(get_db)):
    return db.scalars(select(Job).where(Job.type == jobs.JOB_TRAIN_EVENTS).order_by(Job.created_at.desc())
                      .limit(10)).all()


@router.get("/training/status")
def training_status(db: Session = Depends(get_db)) -> dict:
    return {"reviewed_swings": len(reviewed_swing_ids(db))}


@router.post("/models/{model_id}/promote", response_model=list[ModelOut])
def promote(model_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Make this the active model for its task (for a per-golfer model: for its golfer). The
    previously active model is kept (deprecated) and can be promoted back at any time."""
    m = db.get(Model, model_id)
    if m is None or m.user_id not in (None, user.id):
        raise HTTPException(404, "model not found")
    same_owner = Model.user_id.is_(None) if m.user_id is None else Model.user_id == m.user_id
    for other in db.scalars(select(Model).where(Model.task == m.task, same_owner,
                                                Model.status == ModelStatus.active)).all():
        if other.id != m.id:
            other.status = ModelStatus.deprecated
    m.status = ModelStatus.active
    db.commit()
    return visible_models(db, user)


@router.post("/swings/redetect-events", response_model=dict)
def redetect_all(db: Session = Depends(get_db)):
    """Queue event re-detection with the active model for every swing that has pose."""
    n = 0
    for swing in db.scalars(select(Swing)).all():
        if latest_pose_sequence(db, swing.id) is None:
            continue
        jobs.enqueue(db, jobs.JOB_DETECT_EVENTS, {"swing_id": str(swing.id)}, subject_id=swing.id)
        n += 1
    db.commit()
    return {"queued": n}


@router.put("/swings/{swing_id}/review", response_model=SwingDetail)
def set_reviewed(swing_id: uuid.UUID, body: ReviewIn, db: Session = Depends(get_db),
                 user: User = Depends(current_user)):
    """Mark this swing's 8 events as checked by you - only reviewed swings become training data."""
    swing = get_swing_or_404(db, swing_id, user)
    db.add(Label(task=REVIEW_LABEL_TASK, target_type="swing", target_id=swing.id,
                 corrected_value={"reviewed": body.reviewed}))
    db.commit()
    return swing_detail(db, swing)
