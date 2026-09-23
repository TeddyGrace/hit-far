"""Hands-off housekeeping queued at API start-up."""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.models import Job, Model
from app.pipeline.registry import TASK_EVENTS
from app.pipeline.run import swings_needing_metrics

log = logging.getLogger(__name__)


def queue_startup_jobs(db: Session) -> list[str]:
    queued = []
    # A metric formula change (PIPELINE_VERSION bump) leaves existing swings without current
    # metrics; recompute them from cached pose. The worker retrains outcome models afterwards.
    if swings_needing_metrics(db) and jobs.enqueue_once(db, jobs.JOB_RECOMPUTE_METRICS, {}):
        queued.append(jobs.JOB_RECOMPUTE_METRICS)
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
