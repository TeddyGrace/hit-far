"""Postgres-backed job queue (no Redis): claim with SELECT ... FOR UPDATE SKIP LOCKED."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select, true, update
from sqlalchemy.orm import Session

from app.models import Job, JobStatus

JOB_PROCESS_VIDEO = "process_video"
JOB_DETECT_EVENTS = "detect_events"
JOB_TRAIN_EVENTS = "train_events"
JOB_RECOMPUTE_METRICS = "recompute_metrics"
JOB_TRAIN_OUTCOMES = "train_outcomes"
JOB_TRACK_CLUB = "track_club"

# Which job types each worker role consumes. Training runs for hours on its own service so it
# never blocks swing processing. Outcome models are small tabular models that train in seconds, so
# the processing worker runs them.
ROLE_JOB_TYPES = {
    "worker": (JOB_PROCESS_VIDEO, JOB_DETECT_EVENTS, JOB_RECOMPUTE_METRICS, JOB_TRAIN_OUTCOMES, JOB_TRACK_CLUB),
    "trainer": (JOB_TRAIN_EVENTS,),
}


def enqueue(db: Session, type_: str, payload: dict, subject_id: uuid.UUID | None = None) -> Job:
    job = Job(type=type_, payload=payload, subject_id=subject_id)
    db.add(job)
    db.flush()
    return job


def claim(db: Session, lock_timeout_s: int, types: tuple[str, ...] | None = None) -> Job | None:
    """Claim the next runnable job. Also reclaims jobs whose worker died mid-run (no heartbeat -
    `set_stage` refreshes the lock - for `lock_timeout_s`)."""
    now = datetime.now(timezone.utc)
    stale = now - timedelta(seconds=lock_timeout_s)
    job = db.scalar(
        select(Job)
        .where(Job.type.in_(types) if types else true())
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
    """Report progress; doubles as the heartbeat that keeps a long job from being reclaimed."""
    db.execute(update(Job).where(Job.id == job_id).values(stage=stage[:50], locked_at=datetime.now(timezone.utc)))
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


def pending(db: Session, type_: str) -> Job | None:
    """A queued or running job of this type, if any."""
    return db.scalar(select(Job).where(Job.type == type_, Job.status.in_([JobStatus.queued, JobStatus.running]))
                     .limit(1))


def enqueue_once(db: Session, type_: str, payload: dict) -> Job | None:
    """Enqueue unless the same job (type and payload) is already waiting to run (running ones don't
    count: they may have read the data before the change that triggered this)."""
    if db.scalar(select(Job).where(Job.type == type_, Job.payload == payload, Job.status == JobStatus.queued)
                 .limit(1)):
        return None
    return enqueue(db, type_, payload)


def latest_for(db: Session, subject_id: uuid.UUID) -> Job | None:
    return db.scalar(select(Job).where(Job.subject_id == subject_id).order_by(Job.created_at.desc()).limit(1))
