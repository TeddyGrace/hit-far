"""Postgres-backed job queue (no Redis): claim with SELECT ... FOR UPDATE SKIP LOCKED."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.models import Job, JobStatus

JOB_PROCESS_VIDEO = "process_video"


def enqueue(db: Session, type_: str, payload: dict, subject_id: uuid.UUID | None = None) -> Job:
    job = Job(type=type_, payload=payload, subject_id=subject_id)
    db.add(job)
    db.flush()
    return job


def claim(db: Session, lock_timeout_s: int) -> Job | None:
    """Claim the next runnable job. Also reclaims jobs whose worker died mid-run."""
    now = datetime.now(timezone.utc)
    stale = now - timedelta(seconds=lock_timeout_s)
    job = db.scalar(
        select(Job)
        .where(
            or_(
                (Job.status == JobStatus.queued) & (Job.run_after <= now),
                (Job.status == JobStatus.running) & (Job.locked_at < stale),
            )
        )
        .order_by(Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if job is None:
        db.rollback()
        return None
    job.status = JobStatus.running
    job.locked_at = now
    job.attempts += 1
    job.error = None
    db.commit()
    return job


def set_stage(db: Session, job_id: uuid.UUID, stage: str) -> None:
    db.execute(update(Job).where(Job.id == job_id).values(stage=stage))
    db.commit()


def finish(db: Session, job_id: uuid.UUID) -> None:
    db.execute(update(Job).where(Job.id == job_id).values(status=JobStatus.done, stage="done", locked_at=None))
    db.commit()


def fail(db: Session, job_id: uuid.UUID, error: str, retryable: bool = True) -> None:
    job = db.get(Job, job_id)
    if job is None:
        return
    job.error = error[-4000:]
    job.locked_at = None
    if retryable and job.attempts < job.max_attempts:
        job.status = JobStatus.queued
        job.run_after = datetime.now(timezone.utc) + timedelta(seconds=30 * 2 ** (job.attempts - 1))
    else:
        job.status = JobStatus.failed
    db.commit()


def latest_for(db: Session, subject_id: uuid.UUID) -> Job | None:
    return db.scalar(select(Job).where(Job.subject_id == subject_id).order_by(Job.created_at.desc()).limit(1))
