import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import jobs
from app.auth import current_user, require_auth
from app.db import get_db
from app.models import PoseSequence, RecordingSession, ShotOutcome, Swing, User, Video
from app.schemas import (
    JobOut,
    SessionDetail,
    SessionIn,
    SessionOut,
    SessionPatch,
    SessionSummary,
    VideoOut,
    VideoWithStatus,
)
from app.storage import get_storage

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"], dependencies=[Depends(require_auth)])


def get_session_or_404(db: Session, session_id: uuid.UUID, user: User) -> RecordingSession:
    s = db.get(RecordingSession, session_id)
    if s is None or s.user_id != user.id:  # someone else's session looks like a missing one
        raise HTTPException(404, "session not found")
    return s


@router.get("", response_model=list[SessionSummary])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(current_user)):
    nv = select(func.count(Video.id)).where(Video.session_id == RecordingSession.id).scalar_subquery()
    ns = select(func.count(Swing.id)).where(Swing.session_id == RecordingSession.id).scalar_subquery()
    rows = db.execute(select(RecordingSession, nv, ns).where(RecordingSession.user_id == user.id)
                      .order_by(RecordingSession.recorded_at.desc())).all()
    return [
        SessionSummary(**SessionOut.model_validate(s).model_dump(), num_videos=v, num_swings=w) for s, v, w in rows
    ]


@router.post("", response_model=SessionOut, status_code=201)
def create_session(body: SessionIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = RecordingSession(
        user_id=user.id,
        recorded_at=body.recorded_at or datetime.now(timezone.utc),
        name=body.name, location=body.location, club_used=body.club_used, notes=body.notes,
    )
    db.add(s)
    db.commit()
    return s


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = get_session_or_404(db, session_id, user)
    swings = db.scalars(select(Swing).where(Swing.session_id == s.id)).all()
    swing_by_video = {vid: sw.id for sw in swings for vid in sw.video_ids}
    outcomes = {o.swing_id: o for o in db.scalars(
        select(ShotOutcome).where(ShotOutcome.swing_id.in_([sw.id for sw in swings]))).all()} if swings else {}
    videos = []
    for v in s.videos:
        job = jobs.latest_for(db, v.id)
        videos.append(VideoWithStatus(
            **VideoOut.model_validate(v).model_dump(),
            job=JobOut.model_validate(job) if job else None,
            swing_id=swing_by_video.get(v.id),
            outcome=outcomes.get(swing_by_video.get(v.id)),
        ))
    return SessionDetail(**SessionOut.model_validate(s).model_dump(), videos=videos)


@router.patch("/{session_id}", response_model=SessionOut)
def update_session(session_id: uuid.UUID, body: SessionPatch, db: Session = Depends(get_db),
                   user: User = Depends(current_user)):
    s = get_session_or_404(db, session_id, user)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(s, k, v)
    db.commit()
    return s


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = get_session_or_404(db, session_id, user)
    keys: list[str] = []
    for v in s.videos:
        keys += [k for k in (v.file_uri, v.proxy_uri) if k]
    swing_ids = [sw.id for sw in s.swings]
    if swing_ids:
        keys += db.scalars(select(PoseSequence.keypoint_uri).where(PoseSequence.swing_id.in_(swing_ids))).all()
    db.delete(s)
    db.commit()
    st = get_storage()
    for k in keys:
        try:
            st.delete(k)
        except Exception:  # best effort; rows are already gone
            log.warning("could not delete storage object %s", k)
