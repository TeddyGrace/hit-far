"""Hands-off housekeeping queued at API start-up."""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.models import Job, JobStatus, Model
from app.outcomes.service import maybe_queue_training, users_with_stale_models
from app.pipeline.registry import TASK_EVENTS
from app.pipeline.run import swings_needing_club, swings_needing_metrics

log = logging.getLogger(__name__)


def queue_startup_jobs(db: Session) -> list[str]:
    queued = []
    # A metric formula change (PIPELINE_VERSION bump) leaves existing swings without current
    # metrics; recompute them from cached pose. The worker retrains outcome models afterwards.
    if swings_needing_metrics(db) and jobs.enqueue_once(db, jobs.JOB_RECOMPUTE_METRICS, {}):
        queued.append(jobs.JOB_RECOMPUTE_METRICS)
    # Club tracking for swings processed before the tracker existed (or before a tracker version
    # bump). One job per swing; the last one retrains the outcome models with the club metrics.
    pending = set(db.scalars(select(Job.subject_id).where(
        Job.type == jobs.JOB_TRACK_CLUB, Job.status.in_([JobStatus.queued, JobStatus.running]))).all())
    todo = [s for s in swings_needing_club(db) if s not in pending]
    for n, sid in enumerate(todo, 1):
        jobs.enqueue(db, jobs.JOB_TRACK_CLUB, {"swing_id": str(sid), "retrain_after": n == len(todo)},
                     subject_id=sid)
    if todo:
        queued.append(f"{jobs.JOB_TRACK_CLUB} x{len(todo)}")
    # Outcome models trained under an older problem definition (PROBLEMS_VERSION bump): retrain.
    stale = [uid for uid in users_with_stale_models(db) if maybe_queue_training(db, uid, force=True)]
    if stale:
        queued.append(f"{jobs.JOB_TRAIN_OUTCOMES} x{len(stale)}")
    # First event-model training: once, if nothing trained exists and no run was ever attempted
    # (a failed run is not retried automatically; start it again from the Models page).
    trained = db.scalar(select(Model.id).where(Model.task == TASK_EVENTS, Model.checkpoint_uri.is_not(None)).limit(1))
    attempted = db.scalar(select(Job.id).where(Job.type == jobs.JOB_TRAIN_EVENTS).limit(1))
    if trained is None and attempted is None:
        job = jobs.enqueue(db, jobs.JOB_TRAIN_EVENTS, {"config": {"epochs": 40, "golfdb": True}})
        job.max_attempts = 2
        queued.append(jobs.JOB_TRAIN_EVENTS)
    db.commit()
    if queued:
        log.info("queued at startup: %s", ", ".join(queued))
    return queued
