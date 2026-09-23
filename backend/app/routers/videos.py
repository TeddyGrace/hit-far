import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.auth import require_auth
from app.db import get_db
from app.models import JobStatus, PoseSequence, Swing, Video, VideoStatus
from app.routers.sessions import get_session_or_404
from app.schemas import JobOut, ReprocessIn, UploadIn, UploadOut, VideoOut
from app.storage import get_storage

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["videos"], dependencies=[Depends(require_auth)])

ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def get_video_or_404(db: Session, video_id: uuid.UUID) -> Video:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "video not found")
    return v


@router.post("/sessions/{session_id}/videos", response_model=UploadOut, status_code=201)
def create_upload(session_id: uuid.UUID, body: UploadIn, db: Session = Depends(get_db)):
    get_session_or_404(db, session_id)
    ext = Path(body.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type {ext or '(none)'}")
    vid = uuid.uuid4()
    v = Video(
        id=vid, session_id=session_id, original_filename=body.filename, content_type=body.content_type,
        camera_role=body.camera_role, file_uri=f"uploads/{vid}{ext}", status=VideoStatus.pending_upload,
    )
    db.add(v)
    db.commit()
    url = get_storage().presign_put(v.file_uri, body.content_type)
    return UploadOut(video=VideoOut.model_validate(v), upload_url=url,
                     upload_headers={"Content-Type": body.content_type})


@router.post("/videos/{video_id}/complete", response_model=JobOut)
def complete_upload(video_id: uuid.UUID, db: Session = Depends(get_db)):
    v = get_video_or_404(db, video_id)
    if v.status != VideoStatus.pending_upload:
        raise HTTPException(409, f"video is already {v.status.value}")
    if not get_storage().exists(v.file_uri):
        raise HTTPException(400, "upload not found in storage")
    v.status = VideoStatus.uploaded
    job = jobs.enqueue(db, jobs.JOB_PROCESS_VIDEO, {"video_id": str(v.id)}, subject_id=v.id)
    db.commit()
    return job


@router.post("/videos/{video_id}/reprocess", response_model=JobOut)
def reprocess(video_id: uuid.UUID, body: ReprocessIn, db: Session = Depends(get_db)):
    v = get_video_or_404(db, video_id)
    if v.status == VideoStatus.pending_upload:
        raise HTTPException(409, "upload not completed")
    latest = jobs.latest_for(db, v.id)
    if latest and latest.status in (JobStatus.queued, JobStatus.running):
        raise HTTPException(409, "a job is already queued or running for this video")
    if not get_storage().exists(v.file_uri):
        raise HTTPException(409, "original file is no longer in storage; delete this video and re-upload")
    if v.status == VideoStatus.failed and not v.checksum:
        v.status = VideoStatus.uploaded  # preprocessing never succeeded; retry from the raw upload
        v.error = None
    job = jobs.enqueue(db, jobs.JOB_PROCESS_VIDEO, {"video_id": str(v.id), "force": body.force}, subject_id=v.id)
    db.commit()
    return job


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    v = get_video_or_404(db, video_id)
    keys = [k for k in (v.file_uri, v.proxy_uri) if k]
    swings = db.scalars(select(Swing).where(Swing.video_ids.any(v.id))).all()
    for sw in swings:
        keys += db.scalars(select(PoseSequence.keypoint_uri).where(PoseSequence.swing_id == sw.id)).all()
        db.delete(sw)
    db.delete(v)
    db.commit()
    st = get_storage()
    for k in keys:
        try:
            st.delete(k)
        except Exception:
            log.warning("could not delete storage object %s", k)
